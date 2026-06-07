"""ESP-XFORM 上位机 (PyQt5)

把"复音吉他音色转换"整条 PC 流程封装为图形界面：
  1) 数据/预处理：导入音频(自动转 48k 单声道)→ 组织 clip → 运行预处理。
  2) 训练可视化：配置超参 → 实时 train_loss / val_logmel 曲线 → 训练完成自动导出。
  3) 推理/部署：选 checkpoint + 输入吉他 wav + 目标乐器 → 渲染转换 wav，看波形/频谱对比、试听。

依赖：PyQt5 + matplotlib（均已在环境中）。重计算放 QThread，避免界面卡死。
运行：  python tools/xform_studio.py
"""
from __future__ import annotations

import os
import sys
import json
import traceback
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

try:
    import matplotlib
    matplotlib.use("Qt5Agg")
    # 让内嵌图表能正确显示中文(优先 Windows 常见 CJK 字体, 找不到则回退)
    matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    matplotlib.rcParams["axes.unicode_minus"] = False
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.figure import Figure
    from PyQt5 import QtCore, QtWidgets
except ImportError as _e:
    sys.stderr.write(
        "\n[xform_studio] 缺少依赖: %s\n" % _e
        + "当前 Python 解释器没有 PyQt5/matplotlib。请用项目自带环境启动(任意终端/编辑器均可):\n"
        + "  方式1(推荐): 双击或运行项目根目录的  run_studio.bat\n"
        + "  方式2: 用 conda 环境 tcn_cuda 的解释器运行:\n"
        + "    C:\\Users\\BanGO\\Anaconda3\\envs\\tcn_cuda\\python.exe tools\\xform_studio.py\n\n")
    sys.exit(1)

# 轻量(纯 numpy)依赖，torch 相关模块在各 worker 线程内懒加载
from esp_xform.config import load_instruments
from esp_xform.mask.config import load_mask_config
from esp_xform.audio.io import load_wav, save_wav
from esp_xform.audio.resample import resample_to

SR = 48000


# --------------------------------------------------------------------------- #
# 工具函数
# --------------------------------------------------------------------------- #
def spectral_centroid(y: np.ndarray, sr: int = SR) -> float:
    if len(y) == 0:
        return 0.0
    w = np.hanning(len(y))
    mag = np.abs(np.fft.rfft(y * w))
    freqs = np.fft.rfftfreq(len(y), 1.0 / sr)
    s = mag.sum()
    return float((freqs * mag).sum() / s) if s > 1e-9 else 0.0


def wav_stats(path: str):
    y, sr = load_wav(path, expected_sr=None, mono=True)
    peak = float(np.abs(y).max()) if len(y) else 0.0
    rms = float(np.sqrt(np.mean(y ** 2))) if len(y) else 0.0
    return {"sr": sr, "n": len(y), "dur": len(y) / sr if sr else 0,
            "peak": peak, "rms": rms, "centroid": spectral_centroid(y, sr), "y": y}


def open_in_os(path: str):
    """用系统默认程序打开文件/文件夹(播放 wav / 浏览目录)。"""
    try:
        if sys.platform.startswith("win"):
            os.startfile(path)  # type: ignore[attr-defined]
        elif sys.platform == "darwin":
            os.system(f'open "{path}"')
        else:
            os.system(f'xdg-open "{path}"')
    except Exception as e:
        QtWidgets.QMessageBox.warning(None, "打开失败", str(e))


def next_clip_dir(raw_dir: Path) -> Path:
    n = 0
    for d in raw_dir.glob("clip_*"):
        try:
            n = max(n, int(d.name.split("_")[1]))
        except Exception:
            pass
    return raw_dir / f"clip_{n + 1:04d}"


# --------------------------------------------------------------------------- #
# Worker 线程
# --------------------------------------------------------------------------- #
class PreprocessWorker(QtCore.QThread):
    log = QtCore.pyqtSignal(str)
    done = QtCore.pyqtSignal(dict)
    failed = QtCore.pyqtSignal(str)

    def __init__(self, cfg_path, inst_path):
        super().__init__()
        self.cfg_path, self.inst_path = cfg_path, inst_path

    def run(self):
        try:
            from esp_xform.mask.dataset import preprocess
            cfg = load_mask_config(self.cfg_path)
            inst = load_instruments(self.inst_path)["instruments"]
            self.log.emit(f"乐器映射: {inst}")
            self.log.emit(f"raw={cfg.paths.raw_dir}  ->  proc={cfg.paths.proc_dir}")
            info = preprocess(cfg, inst)
            self.done.emit(info)
        except Exception:
            self.failed.emit(traceback.format_exc())


class TrainWorker(QtCore.QThread):
    epoch = QtCore.pyqtSignal(int, float, float, float)  # ep, train_loss, val(-1=无), best
    log = QtCore.pyqtSignal(str)
    done = QtCore.pyqtSignal(dict)
    failed = QtCore.pyqtSignal(str)

    def __init__(self, cfg_path, inst_path, epochs, lr, batch, device):
        super().__init__()
        self.cfg_path, self.inst_path = cfg_path, inst_path
        self.epochs, self.lr, self.batch, self.device = epochs, lr, batch, device
        self._stop = False

    def request_stop(self):
        self._stop = True

    def run(self):
        try:
            import torch
            from esp_xform.mask.train import train as train_mask
            from esp_xform.mask.model import MaskNet
            from esp_xform.mask.export import export_masknet_artifacts

            cfg = load_mask_config(self.cfg_path)
            cfg.train.epochs = int(self.epochs)
            cfg.train.lr = float(self.lr)
            cfg.train.batch_size = int(self.batch)
            inst = load_instruments(self.inst_path)["instruments"]

            self.log.emit(f"开始训练: device={self.device} epochs={self.epochs} "
                          f"lr={self.lr} batch={self.batch} 乐器={inst}")

            def on_epoch(ep, tr, val, best):
                self.epoch.emit(int(ep), float(tr),
                                float(val) if val is not None else -1.0, float(best))

            info = train_mask(cfg, inst, device=self.device,
                              on_epoch=on_epoch, should_stop=lambda: self._stop)
            self.log.emit(f"训练结束: {info}")

            # 自动导出产物(权重/矩阵/嵌入/统计)
            ckpt = Path(info["ckpt_dir"]) / "best.pt"
            if not ckpt.exists():
                ckpt = Path(info["ckpt_dir"]) / "last.pt"
            state = torch.load(ckpt, map_location="cpu", weights_only=False)
            model = MaskNet(cfg, state["num_instruments"])
            model.load_state_dict(state["model_state"])
            model.eval()
            out = export_masknet_artifacts(model, cfg, cfg.paths.output_dir,
                                           state["instruments"], proc_dir=cfg.paths.proc_dir)
            self.log.emit(f"已导出产物到 {cfg.paths.output_dir}/exports")
            info["export"] = out
            info["ckpt"] = str(ckpt)
            self.done.emit(info)
        except Exception:
            self.failed.emit(traceback.format_exc())


class RenderWorker(QtCore.QThread):
    done = QtCore.pyqtSignal(str, dict, dict)   # out_path, in_stats, out_stats
    failed = QtCore.pyqtSignal(str)

    def __init__(self, ckpt, source, inst_name, inst_id, out_path, device, add_noise,
                 pitch=0.0, gain=0.0, clip_mode="limit"):
        super().__init__()
        self.ckpt, self.source, self.inst_name = ckpt, source, inst_name
        self.inst_id, self.out_path = inst_id, out_path
        self.device, self.add_noise, self.pitch = device, add_noise, float(pitch)
        self.gain = float(gain)
        self.clip_mode = clip_mode

    def run(self):
        try:
            from esp_xform.mask.infer import load_mask_model, render_instrument
            model, cfg, mats, mean, std, _ = load_mask_model(self.ckpt, self.device)
            audio, _ = load_wav(self.source, expected_sr=cfg.stft.sample_rate, mono=True)
            y = render_instrument(model, cfg, mats, mean, std, audio,
                                  int(self.inst_id), self.device, self.add_noise,
                                  pitch_semitones=self.pitch, gain_db=self.gain,
                                  clip_mode=self.clip_mode)
            Path(self.out_path).parent.mkdir(parents=True, exist_ok=True)
            save_wav(self.out_path, y, cfg.stft.sample_rate)
            in_stats = wav_stats(self.source)
            out_stats = wav_stats(self.out_path)
            self.done.emit(self.out_path, in_stats, out_stats)
        except Exception:
            self.failed.emit(traceback.format_exc())


# --------------------------------------------------------------------------- #
# 通用：带浏览按钮的路径选择行
# --------------------------------------------------------------------------- #
class PathRow(QtWidgets.QWidget):
    def __init__(self, label, default="", pick_dir=False, file_filter="所有文件 (*.*)"):
        super().__init__()
        self.pick_dir, self.file_filter = pick_dir, file_filter
        lay = QtWidgets.QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(QtWidgets.QLabel(label))
        self.edit = QtWidgets.QLineEdit(default)
        lay.addWidget(self.edit, 1)
        btn = QtWidgets.QPushButton("浏览…")
        btn.clicked.connect(self._browse)
        lay.addWidget(btn)

    def _browse(self):
        if self.pick_dir:
            p = QtWidgets.QFileDialog.getExistingDirectory(self, "选择目录", self.edit.text() or str(ROOT))
        else:
            p, _ = QtWidgets.QFileDialog.getOpenFileName(self, "选择文件", self.edit.text() or str(ROOT), self.file_filter)
        if p:
            self.edit.setText(p)

    def text(self):
        return self.edit.text().strip()


# --------------------------------------------------------------------------- #
# 主窗口
# --------------------------------------------------------------------------- #
class Studio(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ESP-XFORM 上位机 — 复音吉他音色转换")
        self.resize(1080, 760)

        self._has_cuda = self._detect_cuda()
        self.train_worker = None
        self.pre_worker = None
        self.render_worker = None
        self.ep_x, self.ep_tr, self.ep_val = [], [], []

        tabs = QtWidgets.QTabWidget()
        tabs.addTab(self._build_data_tab(), "1 · 数据 / 预处理")
        tabs.addTab(self._build_train_tab(), "2 · 训练可视化")
        tabs.addTab(self._build_infer_tab(), "3 · 推理 / 输出")
        self.setCentralWidget(tabs)
        self.statusBar().showMessage("就绪" + ("（检测到 CUDA）" if self._has_cuda else "（仅 CPU）"))

    @staticmethod
    def _detect_cuda() -> bool:
        try:
            import torch
            return bool(torch.cuda.is_available())
        except Exception:
            return False

    # ----------------------------- Tab 1: 数据 ----------------------------- #
    def _build_data_tab(self):
        w = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(w)

        self.cfg_row = PathRow("Mask 配置", str(ROOT / "configs" / "mask_style.json"),
                               file_filter="配置 (*.json *.yaml)")
        self.inst_row = PathRow("乐器映射", str(ROOT / "configs" / "instruments.json"),
                                file_filter="乐器配置 (*.json *.yaml)")
        self.raw_row = PathRow("原始数据目录(raw)", str(ROOT / "dataset" / "raw_style"), pick_dir=True)
        for r in (self.cfg_row, self.inst_row, self.raw_row):
            lay.addWidget(r)

        # 导入音频区
        box = QtWidgets.QGroupBox("导入音频到新 clip（自动转 48k / 单声道；源吉他必填，目标可选）")
        g = QtWidgets.QGridLayout(box)
        self.imp_guitar = PathRow("吉他(源)", file_filter="音频 (*.wav *.flac *.aiff *.mp3)")
        self.imp_target = PathRow("目标乐器音频", file_filter="音频 (*.wav *.flac *.aiff *.mp3)")
        self.imp_target_name = QtWidgets.QLineEdit("bass")
        g.addWidget(self.imp_guitar, 0, 0, 1, 3)
        g.addWidget(self.imp_target, 1, 0, 1, 3)
        g.addWidget(QtWidgets.QLabel("目标乐器名"), 2, 0)
        g.addWidget(self.imp_target_name, 2, 1)
        imp_btn = QtWidgets.QPushButton("导入为新 clip")
        imp_btn.clicked.connect(self.on_import)
        g.addWidget(imp_btn, 2, 2)
        lay.addWidget(box)

        # clip 列表
        self.clip_table = QtWidgets.QTableWidget(0, 2)
        self.clip_table.setHorizontalHeaderLabels(["clip", "包含的音频文件"])
        self.clip_table.horizontalHeader().setSectionResizeMode(1, QtWidgets.QHeaderView.Stretch)
        lay.addWidget(self.clip_table, 1)

        row = QtWidgets.QHBoxLayout()
        refresh = QtWidgets.QPushButton("刷新 clip 列表")
        refresh.clicked.connect(self.refresh_clips)
        self.pre_btn = QtWidgets.QPushButton("运行预处理")
        self.pre_btn.clicked.connect(self.on_preprocess)
        row.addWidget(refresh)
        row.addWidget(self.pre_btn)
        row.addStretch(1)
        lay.addLayout(row)

        self.data_log = QtWidgets.QPlainTextEdit()
        self.data_log.setReadOnly(True)
        self.data_log.setMaximumHeight(140)
        lay.addWidget(self.data_log)

        self.refresh_clips()
        return w

    def refresh_clips(self):
        raw = Path(self.raw_row.text())
        self.clip_table.setRowCount(0)
        if not raw.exists():
            return
        for d in sorted(raw.glob("clip_*")):
            if not d.is_dir():
                continue
            wavs = ", ".join(sorted(p.name for p in d.glob("*.wav")))
            r = self.clip_table.rowCount()
            self.clip_table.insertRow(r)
            self.clip_table.setItem(r, 0, QtWidgets.QTableWidgetItem(d.name))
            self.clip_table.setItem(r, 1, QtWidgets.QTableWidgetItem(wavs))

    def _import_one(self, src: str, dst: Path):
        y, sr = load_wav(src, expected_sr=None, mono=True)
        if sr != SR:
            y = resample_to(y, sr, SR)
        save_wav(dst, y, SR)
        return len(y) / SR

    def on_import(self):
        gsrc = self.imp_guitar.text()
        if not gsrc or not Path(gsrc).exists():
            QtWidgets.QMessageBox.warning(self, "缺少吉他音频", "请先选择吉他(源)音频文件。")
            return
        raw = Path(self.raw_row.text())
        raw.mkdir(parents=True, exist_ok=True)
        clip = next_clip_dir(raw)
        clip.mkdir(parents=True, exist_ok=True)
        try:
            insts = ["guitar"]
            dur = self._import_one(gsrc, clip / "guitar.wav")
            tsrc = self.imp_target.text()
            tname = self.imp_target_name.text().strip() or "target"
            targets = []
            if tsrc and Path(tsrc).exists():
                self._import_one(tsrc, clip / f"{tname}.wav")
                insts.append(tname)
                targets.append(tname)
            meta = {"sample_rate": SR, "channels": 1, "instruments": insts,
                    "source": "guitar", "targets": targets, "duration_s": round(dur, 3),
                    "imported": True}
            (clip / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2),
                                            encoding="utf-8")
            self.data_log.appendPlainText(f"[导入] {clip.name}: {insts}  时长 {dur:.2f}s")
            self.refresh_clips()
        except Exception:
            self.data_log.appendPlainText(traceback.format_exc())

    def on_preprocess(self):
        if self.pre_worker and self.pre_worker.isRunning():
            return
        self.pre_btn.setEnabled(False)
        self.data_log.appendPlainText("=== 预处理开始 ===")
        self.pre_worker = PreprocessWorker(self.cfg_row.text(), self.inst_row.text())
        self.pre_worker.log.connect(lambda s: self.data_log.appendPlainText(s))
        self.pre_worker.done.connect(self._pre_done)
        self.pre_worker.failed.connect(self._pre_failed)
        self.pre_worker.start()

    def _pre_done(self, info):
        self.data_log.appendPlainText(f"=== 预处理完成: {info} ===")
        self.pre_btn.setEnabled(True)
        self.statusBar().showMessage(f"预处理完成: {info.get('n_pairs')} 对 / {info.get('n_windows')} 窗")

    def _pre_failed(self, tb):
        self.data_log.appendPlainText("预处理失败:\n" + tb)
        self.pre_btn.setEnabled(True)

    # ----------------------------- Tab 2: 训练 ----------------------------- #
    def _build_train_tab(self):
        w = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(w)

        cfg = QtWidgets.QHBoxLayout()
        self.sp_epochs = QtWidgets.QSpinBox(); self.sp_epochs.setRange(1, 5000); self.sp_epochs.setValue(120)
        self.sp_lr = QtWidgets.QDoubleSpinBox(); self.sp_lr.setDecimals(5); self.sp_lr.setRange(1e-5, 1.0)
        self.sp_lr.setSingleStep(1e-4); self.sp_lr.setValue(3e-4)
        self.sp_batch = QtWidgets.QSpinBox(); self.sp_batch.setRange(1, 512); self.sp_batch.setValue(32)
        self.cb_device = QtWidgets.QComboBox()
        self.cb_device.addItems((["cuda"] if self._has_cuda else []) + ["cpu"])
        for lbl, wid in [("epochs", self.sp_epochs), ("lr", self.sp_lr),
                         ("batch", self.sp_batch), ("device", self.cb_device)]:
            cfg.addWidget(QtWidgets.QLabel(lbl)); cfg.addWidget(wid)
        cfg.addStretch(1)
        self.btn_train = QtWidgets.QPushButton("开始训练")
        self.btn_train.clicked.connect(self.on_train)
        self.btn_stop = QtWidgets.QPushButton("停止"); self.btn_stop.setEnabled(False)
        self.btn_stop.clicked.connect(self.on_stop_train)
        cfg.addWidget(self.btn_train); cfg.addWidget(self.btn_stop)
        lay.addLayout(cfg)

        # 曲线
        self.fig = Figure(figsize=(7, 3.2))
        self.ax = self.fig.add_subplot(111)
        self.ax.set_xlabel("epoch"); self.ax.set_ylabel("loss")
        self.ax.grid(True, alpha=0.3)
        self.canvas = FigureCanvas(self.fig)
        lay.addWidget(self.canvas, 1)

        self.prog = QtWidgets.QProgressBar()
        lay.addWidget(self.prog)

        self.train_log = QtWidgets.QPlainTextEdit(); self.train_log.setReadOnly(True)
        self.train_log.setMaximumHeight(170)
        lay.addWidget(self.train_log)
        return w

    def on_train(self):
        if self.train_worker and self.train_worker.isRunning():
            return
        self.ep_x, self.ep_tr, self.ep_val = [], [], []
        self.ax.clear()
        self.ax.set_xlabel("epoch"); self.ax.set_ylabel("loss"); self.ax.grid(True, alpha=0.3)
        self.canvas.draw()
        self.prog.setMaximum(self.sp_epochs.value()); self.prog.setValue(0)
        self.btn_train.setEnabled(False); self.btn_stop.setEnabled(True)
        self.train_log.appendPlainText("=== 训练开始 ===")
        self.train_worker = TrainWorker(self.cfg_row.text(), self.inst_row.text(),
                                        self.sp_epochs.value(), self.sp_lr.value(),
                                        self.sp_batch.value(), self.cb_device.currentText())
        self.train_worker.epoch.connect(self._on_epoch)
        self.train_worker.log.connect(lambda s: self.train_log.appendPlainText(s))
        self.train_worker.done.connect(self._train_done)
        self.train_worker.failed.connect(self._train_failed)
        self.train_worker.start()

    def on_stop_train(self):
        if self.train_worker:
            self.train_worker.request_stop()
            self.train_log.appendPlainText("（已请求停止，将在本轮结束后停止）")
            self.btn_stop.setEnabled(False)

    def _on_epoch(self, ep, tr, val, best):
        self.ep_x.append(ep); self.ep_tr.append(tr)
        if val >= 0:
            self.ep_val.append(val)
        self.prog.setValue(ep + 1)
        self.ax.clear()
        self.ax.set_xlabel("epoch"); self.ax.set_ylabel("loss"); self.ax.grid(True, alpha=0.3)
        self.ax.plot(self.ep_x, self.ep_tr, label="train_loss", color="#1f77b4")
        if self.ep_val:
            self.ax.plot(self.ep_x[:len(self.ep_val)], self.ep_val, label="val_logmel", color="#d62728")
        self.ax.legend(loc="upper right")
        self.ax.set_title(f"epoch {ep}  train={tr:.4f}" + (f"  val={val:.4f}  best={best:.4f}" if val >= 0 else ""))
        self.canvas.draw()
        if ep % 5 == 0 or val >= 0:
            self.train_log.appendPlainText(
                f"epoch {ep:3d}  train_loss={tr:.4f}" + (f"  val_logmel={val:.4f}" if val >= 0 else ""))

    def _train_done(self, info):
        self.train_log.appendPlainText(f"=== 训练完成: best_val_logmel={info.get('best_val_logmel')} ===")
        self.train_log.appendPlainText(f"checkpoint: {info.get('ckpt')}")
        self.btn_train.setEnabled(True); self.btn_stop.setEnabled(False)
        self.statusBar().showMessage(f"训练完成 best_val_logmel={info.get('best_val_logmel')}")
        # 自动把推理页的 checkpoint 指向刚训练好的 best.pt
        if info.get("ckpt"):
            self.ckpt_row.edit.setText(info["ckpt"])

    def _train_failed(self, tb):
        self.train_log.appendPlainText("训练失败:\n" + tb)
        self.btn_train.setEnabled(True); self.btn_stop.setEnabled(False)

    # ----------------------------- Tab 3: 推理 ----------------------------- #
    def _build_infer_tab(self):
        w = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(w)

        self.ckpt_row = PathRow("Checkpoint", str(ROOT / "outputs" / "mask_style" / "checkpoints" / "best.pt"),
                                file_filter="检查点 (*.pt)")
        self.src_row = PathRow("输入吉他 wav", file_filter="音频 (*.wav *.flac *.aiff)")
        lay.addWidget(self.ckpt_row)
        lay.addWidget(self.src_row)

        row = QtWidgets.QHBoxLayout()
        self.btn_loadckpt = QtWidgets.QPushButton("加载乐器列表")
        self.btn_loadckpt.clicked.connect(self.on_load_ckpt)
        self.cb_inst = QtWidgets.QComboBox()
        self.chk_noise = QtWidgets.QCheckBox("叠加噪声带"); self.chk_noise.setChecked(True)
        self.btn_render = QtWidgets.QPushButton("渲染转换")
        self.btn_render.clicked.connect(self.on_render)
        row.addWidget(self.btn_loadckpt)
        row.addWidget(QtWidgets.QLabel("目标乐器"))
        row.addWidget(self.cb_inst, 1)
        row.addWidget(self.chk_noise)
        row.addWidget(self.btn_render)
        lay.addLayout(row)

        # 移频/变调（运行时可调）：把输出移到目标乐器音区
        prow2 = QtWidgets.QHBoxLayout()
        prow2.addWidget(QtWidgets.QLabel("移频(半音)"))
        self.sl_pitch = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.sl_pitch.setRange(-24, 24); self.sl_pitch.setValue(0)
        self.sl_pitch.setTickPosition(QtWidgets.QSlider.TicksBelow); self.sl_pitch.setTickInterval(6)
        self.sp_pitch = QtWidgets.QSpinBox(); self.sp_pitch.setRange(-24, 24)
        self.sp_pitch.setValue(0); self.sp_pitch.setSuffix(" 半音")
        self.sl_pitch.valueChanged.connect(self.sp_pitch.setValue)
        self.sp_pitch.valueChanged.connect(self.sl_pitch.setValue)
        self.chk_live = QtWidgets.QCheckBox("调节时自动重渲染"); self.chk_live.setChecked(True)
        self.sl_pitch.valueChanged.connect(self._pitch_changed)
        self.btn_reset_pitch = QtWidgets.QPushButton("归零")
        self.btn_reset_pitch.clicked.connect(lambda: self.sp_pitch.setValue(0))
        prow2.addWidget(self.sl_pitch, 1)
        prow2.addWidget(self.sp_pitch)
        prow2.addWidget(self.btn_reset_pitch)
        prow2.addWidget(self.chk_live)
        lay.addLayout(prow2)
        hint = QtWidgets.QLabel("提示：贝斯常用 -12（低八度）、尤克里里常用 +5~+12 调到其自然音区；可边拖边听。")
        hint.setStyleSheet("color: gray;")
        lay.addWidget(hint)

        # 输出增益（dB，运行时可调）：最后施加，带削波保护
        grow = QtWidgets.QHBoxLayout()
        grow.addWidget(QtWidgets.QLabel("输出增益(dB)"))
        self.sl_gain = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.sl_gain.setRange(-48, 48); self.sl_gain.setValue(0)
        self.sl_gain.setTickPosition(QtWidgets.QSlider.TicksBelow); self.sl_gain.setTickInterval(12)
        self.sp_gain = QtWidgets.QSpinBox(); self.sp_gain.setRange(-48, 48)
        self.sp_gain.setValue(0); self.sp_gain.setSuffix(" dB")
        self.sl_gain.valueChanged.connect(self.sp_gain.setValue)
        self.sp_gain.valueChanged.connect(self.sl_gain.setValue)
        self.sl_gain.valueChanged.connect(self._pitch_changed)   # 复用同一防抖重渲染
        self.btn_reset_gain = QtWidgets.QPushButton("归零")
        self.btn_reset_gain.clicked.connect(lambda: self.sp_gain.setValue(0))
        grow.addWidget(self.sl_gain, 1)
        grow.addWidget(self.sp_gain)
        grow.addWidget(self.btn_reset_gain)
        grow.addWidget(QtWidgets.QLabel("削波"))
        self.cb_clip = QtWidgets.QComboBox()
        # (显示文本, 内部模式值)
        for label, mode in [("限幅(干净)", "limit"), ("软饱和", "soft"), ("硬削波", "hard")]:
            self.cb_clip.addItem(label, mode)
        self.cb_clip.currentIndexChanged.connect(self._pitch_changed)  # 复用防抖重渲染
        grow.addWidget(self.cb_clip)
        lay.addLayout(grow)
        ghint = QtWidgets.QLabel("超低频(如 bass -12)听感小但已满幅时：选「软饱和」+ 提增益，加谐波让小喇叭也听得见、更厚。")
        ghint.setStyleSheet("color: gray;")
        lay.addWidget(ghint)

        # 运行时调节用：防抖定时器 + 重渲染状态
        self._pitch_timer = QtCore.QTimer(self); self._pitch_timer.setSingleShot(True)
        self._pitch_timer.timeout.connect(self._auto_rerender)
        self._pending_rerender = False
        self._rendered_pitch = 0.0
        self._rendered_gain = 0.0
        self._rendered_clip = "limit"

        # 频谱对比
        self.ifig = Figure(figsize=(7, 4))
        self.iax_in = self.ifig.add_subplot(211)
        self.iax_out = self.ifig.add_subplot(212)
        self.icanvas = FigureCanvas(self.ifig)
        lay.addWidget(self.icanvas, 1)

        self.infer_info = QtWidgets.QLabel("—")
        self.infer_info.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        lay.addWidget(self.infer_info)

        prow = QtWidgets.QHBoxLayout()
        self.btn_play_out = QtWidgets.QPushButton("播放输出"); self.btn_play_out.setEnabled(False)
        self.btn_play_out.clicked.connect(lambda: open_in_os(self._last_out) if self._last_out else None)
        self.btn_play_in = QtWidgets.QPushButton("播放输入")
        self.btn_play_in.clicked.connect(lambda: open_in_os(self.src_row.text()) if self.src_row.text() else None)
        self.btn_open_dir = QtWidgets.QPushButton("打开输出目录"); self.btn_open_dir.setEnabled(False)
        self.btn_open_dir.clicked.connect(lambda: open_in_os(str(Path(self._last_out).parent)) if self._last_out else None)
        prow.addWidget(self.btn_play_in); prow.addWidget(self.btn_play_out); prow.addWidget(self.btn_open_dir)
        prow.addStretch(1)
        lay.addLayout(prow)

        self._last_out = None
        return w

    def on_load_ckpt(self):
        ckpt = self.ckpt_row.text()
        if not Path(ckpt).exists():
            QtWidgets.QMessageBox.warning(self, "找不到 checkpoint", ckpt)
            return
        try:
            import torch
            state = torch.load(ckpt, map_location="cpu", weights_only=False)
            insts = state.get("instruments", {})
            self.cb_inst.clear()
            for name, idx in sorted(insts.items(), key=lambda kv: kv[1]):
                if name == "guitar":
                    continue  # 源乐器不作为目标
                self.cb_inst.addItem(f"{name} (id={idx})", idx)
            self.statusBar().showMessage(f"已加载乐器: {list(insts.keys())}")
        except Exception:
            QtWidgets.QMessageBox.critical(self, "加载失败", traceback.format_exc())

    def on_render(self):
        if self.render_worker and self.render_worker.isRunning():
            return
        ckpt, src = self.ckpt_row.text(), self.src_row.text()
        if not Path(ckpt).exists() or not Path(src).exists():
            QtWidgets.QMessageBox.warning(self, "缺少输入", "请确认 checkpoint 与输入吉他 wav 均存在。")
            return
        if self.cb_inst.count() == 0:
            self.on_load_ckpt()
            if self.cb_inst.count() == 0:
                return
        inst_name = self.cb_inst.currentText().split(" ")[0]
        inst_id = self.cb_inst.currentData()
        pitch = float(self.sp_pitch.value())
        gain = float(self.sp_gain.value())
        clip_mode = self.cb_clip.currentData() or "limit"
        psfx = f"_p{int(pitch):+d}" if abs(pitch) > 1e-6 else ""
        gsfx = f"_g{int(gain):+d}" if abs(gain) > 1e-6 else ""
        csfx = f"_{clip_mode}" if clip_mode != "limit" else ""
        out = ROOT / "outputs" / "mask_style" / "renders" / f"{Path(src).stem}__to_{inst_name}{psfx}{gsfx}{csfx}.wav"
        self._rendered_pitch = pitch
        self._rendered_gain = gain
        self._rendered_clip = clip_mode
        self.btn_render.setEnabled(False)
        self.statusBar().showMessage(f"渲染中… (移频 {pitch:+.0f} 半音, 增益 {gain:+.0f} dB, {clip_mode})")
        self.render_worker = RenderWorker(ckpt, src, inst_name, inst_id, str(out),
                                          "cpu", self.chk_noise.isChecked(), pitch, gain, clip_mode)
        self.render_worker.done.connect(self._render_done)
        self.render_worker.failed.connect(self._render_failed)
        self.render_worker.start()

    def _pitch_changed(self, _val):
        """移频控件变化：开启了自动重渲染则防抖触发。"""
        if self.chk_live.isChecked() and self.src_row.text() and Path(self.ckpt_row.text()).exists():
            self._pitch_timer.start(450)

    def _auto_rerender(self):
        if self.render_worker and self.render_worker.isRunning():
            self._pending_rerender = True   # 渲染进行中，待完成后用最新值重渲
            return
        if self.cb_inst.count() == 0:
            return
        self.on_render()

    def _spec(self, ax, y, title):
        ax.clear()
        if len(y) > 1100:
            ax.specgram(y, NFFT=1024, Fs=SR, noverlap=768, cmap="magma")
        ax.set_title(title, fontsize=9)
        ax.set_ylabel("Hz")

    def _render_done(self, out_path, ins, outs):
        self._last_out = out_path
        pitch = self._rendered_pitch
        gain = self._rendered_gain
        clip_mode = self._rendered_clip
        self._spec(self.iax_in, ins["y"], f"INPUT guitar  centroid={ins['centroid']:.0f}Hz  rms={ins['rms']:.4f}")
        self._spec(self.iax_out, outs["y"],
                   f"OUTPUT  pitch{pitch:+.0f}st  gain{gain:+.0f}dB  {clip_mode}  centroid={outs['centroid']:.0f}Hz  rms={outs['rms']:.4f}")
        self.ifig.tight_layout()
        self.icanvas.draw()
        self.infer_info.setText(
            f"输出: {out_path}\n"
            f"移频  : {pitch:+.0f} 半音    输出增益: {gain:+.0f} dB    削波模式: {clip_mode}\n"
            f"输入  : sr={ins['sr']} 时长={ins['dur']:.2f}s peak={ins['peak']:.3f} rms={ins['rms']:.4f} 质心={ins['centroid']:.0f}Hz\n"
            f"输出  : sr={outs['sr']} 时长={outs['dur']:.2f}s peak={outs['peak']:.3f} rms={outs['rms']:.4f} 质心={outs['centroid']:.0f}Hz")
        self.btn_render.setEnabled(True)
        self.btn_play_out.setEnabled(True); self.btn_open_dir.setEnabled(True)
        self.statusBar().showMessage(f"渲染完成 (移频 {pitch:+.0f} 半音, 增益 {gain:+.0f} dB, {clip_mode}): " + out_path)
        # 渲染期间若用户又调了移频/增益/削波模式，用最新值补渲一次
        if self._pending_rerender:
            self._pending_rerender = False
            if (abs(float(self.sp_pitch.value()) - self._rendered_pitch) > 1e-6
                    or abs(float(self.sp_gain.value()) - self._rendered_gain) > 1e-6
                    or (self.cb_clip.currentData() or "limit") != self._rendered_clip):
                QtCore.QTimer.singleShot(0, self._auto_rerender)

    def _render_failed(self, tb):
        QtWidgets.QMessageBox.critical(self, "渲染失败", tb)
        self.btn_render.setEnabled(True)
        self.statusBar().showMessage("渲染失败")


def main():
    app = QtWidgets.QApplication(sys.argv)
    win = Studio()
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
