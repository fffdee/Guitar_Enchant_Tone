"""ESP-GT-XFORM 数据预处理 / 训练 上位机 (PyQt5)。

按钮一键运行各流水线脚本, QProcess 非阻塞、实时显示日志, 可停止/清空。
覆盖: 数据集质检 -> 预处理 -> 训练 -> 渲染/导出。

运行: python tools/preprocess_gui.py   (用 conda 环境的 python, 含 PyQt5 + torch)
"""
from __future__ import annotations

import sys
from pathlib import Path

from PyQt5.QtCore import QProcess, QProcessEnvironment, Qt
from PyQt5.QtGui import QFont, QTextCursor
from PyQt5.QtWidgets import (
    QApplication, QComboBox, QFileDialog, QFormLayout, QGroupBox, QHBoxLayout,
    QLabel, QLineEdit, QMainWindow, QPlainTextEdit, QPushButton, QSpinBox,
    QSplitter, QTabWidget, QVBoxLayout, QWidget,
)

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable  # 当前解释器(应为 conda 环境) -> 子脚本同环境运行


def path_row(default: str, mode: str = "dir", filt: str = "All (*.*)"):
    """返回 (容器widget, QLineEdit)。mode: dir/open/save。"""
    w = QWidget()
    h = QHBoxLayout(w)
    h.setContentsMargins(0, 0, 0, 0)
    le = QLineEdit(default)
    btn = QPushButton("浏览…")
    btn.setFixedWidth(64)

    def browse():
        if mode == "dir":
            p = QFileDialog.getExistingDirectory(w, "选择目录", str(ROOT))
        elif mode == "save":
            p, _ = QFileDialog.getSaveFileName(w, "选择输出文件", str(ROOT), filt)
        else:
            p, _ = QFileDialog.getOpenFileName(w, "选择文件", str(ROOT), filt)
        if p:
            le.setText(p)

    btn.clicked.connect(browse)
    h.addWidget(le)
    h.addWidget(btn)
    return w, le


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ESP-GT-XFORM 数据预处理上位机")
        self.resize(960, 720)
        self.proc: QProcess | None = None
        self._cur_title = ""

        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)

        info = QLabel(f"项目: {ROOT}\nPython: {PY}")
        info.setStyleSheet("color:#888")
        root.addWidget(info)

        splitter = QSplitter(Qt.Vertical)
        root.addWidget(splitter, 1)

        tabs = QTabWidget()
        tabs.addTab(self._tab_dataset(), "① 数据集质检")
        tabs.addTab(self._tab_preprocess(), "② 预处理")
        tabs.addTab(self._tab_train(), "③ 训练")
        tabs.addTab(self._tab_render(), "④ 渲染/导出")
        splitter.addWidget(tabs)

        # 日志区
        logbox = QGroupBox("运行日志")
        lv = QVBoxLayout(logbox)
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setFont(QFont("Consolas", 9))
        self.log.setMaximumBlockCount(5000)
        lv.addWidget(self.log)
        ctl = QHBoxLayout()
        self.status = QLabel("就绪")
        self.btn_stop = QPushButton("停止")
        self.btn_stop.setEnabled(False)
        self.btn_stop.clicked.connect(self.stop)
        btn_clear = QPushButton("清空日志")
        btn_clear.clicked.connect(lambda: self.log.clear())
        ctl.addWidget(self.status, 1)
        ctl.addWidget(self.btn_stop)
        ctl.addWidget(btn_clear)
        lv.addLayout(ctl)
        splitter.addWidget(logbox)
        splitter.setSizes([300, 420])

        self._run_buttons: list[QPushButton] = []
        for b in self.findChildren(QPushButton):
            if b.property("run"):
                self._run_buttons.append(b)

    # ----------------------- Tabs ----------------------- #
    def _tab_dataset(self):
        w = QWidget(); f = QFormLayout(w)
        self.qa_raw_w, self.qa_raw = path_row("dataset/raw", "dir")
        self.qa_src = QLineEdit("guitar")
        self.qa_sr = QSpinBox(); self.qa_sr.setRange(8000, 192000); self.qa_sr.setValue(48000)
        f.addRow("raw 目录", self.qa_raw_w)
        f.addRow("源文件名", self.qa_src)
        f.addRow("期望采样率", self.qa_sr)
        btn = QPushButton("扫描并质检数据集")
        btn.setProperty("run", True)
        btn.clicked.connect(self.run_qa)
        f.addRow(btn)
        return w

    def _tab_preprocess(self):
        w = QWidget(); f = QFormLayout(w)
        self.pp_cfg_w, self.pp_cfg = path_row("configs/mask.json", "open", "JSON/YAML (*.json *.yaml *.yml)")
        self.pp_raw_w, self.pp_raw = path_row("dataset/raw", "dir")
        self.pp_proc_w, self.pp_proc = path_row("dataset/proc", "dir")
        self.pp_limit = QLineEdit(""); self.pp_limit.setPlaceholderText("留空=全部；填数字=仅前 N 个片段(调试)")
        f.addRow("配置", self.pp_cfg_w)
        f.addRow("raw 目录", self.pp_raw_w)
        f.addRow("proc 输出", self.pp_proc_w)
        f.addRow("limit", self.pp_limit)
        btn = QPushButton("运行预处理 (STFT/梅尔/掩码标签/切窗)")
        btn.setProperty("run", True)
        btn.clicked.connect(self.run_preprocess)
        f.addRow(btn)
        return w

    def _tab_train(self):
        w = QWidget(); f = QFormLayout(w)
        self.tr_cfg_w, self.tr_cfg = path_row("configs/mask.json", "open", "JSON/YAML (*.json *.yaml *.yml)")
        self.tr_proc_w, self.tr_proc = path_row("dataset/proc", "dir")
        self.tr_epochs = QSpinBox(); self.tr_epochs.setRange(1, 5000); self.tr_epochs.setValue(50)
        self.tr_device = QComboBox(); self.tr_device.addItems(["auto", "cuda", "cpu"])
        f.addRow("配置", self.tr_cfg_w)
        f.addRow("proc 目录", self.tr_proc_w)
        f.addRow("epochs", self.tr_epochs)
        f.addRow("device", self.tr_device)
        btn = QPushButton("开始训练 (含导出)")
        btn.setProperty("run", True)
        btn.clicked.connect(self.run_train)
        f.addRow(btn)
        return w

    def _tab_render(self):
        w = QWidget(); v = QVBoxLayout(w)
        g1 = QGroupBox("渲染 (用 checkpoint 把吉他转成目标乐器)")
        f1 = QFormLayout(g1)
        self.rd_ckpt_w, self.rd_ckpt = path_row("outputs/mask/checkpoints/best.pt", "open", "ckpt (*.pt)")
        self.rd_src_w, self.rd_src = path_row("dataset/raw/clip_0001/guitar.wav", "open", "wav (*.wav)")
        self.rd_instr = QLineEdit("bass")
        self.rd_out_w, self.rd_out = path_row("outputs/mask/renders/render.wav", "save", "wav (*.wav)")
        f1.addRow("checkpoint", self.rd_ckpt_w)
        f1.addRow("源吉他 wav", self.rd_src_w)
        f1.addRow("目标乐器", self.rd_instr)
        f1.addRow("输出 wav", self.rd_out_w)
        btn1 = QPushButton("渲染")
        btn1.setProperty("run", True)
        btn1.clicked.connect(self.run_render)
        f1.addRow(btn1)
        v.addWidget(g1)

        g2 = QGroupBox("导出部署产物 (BNNW 权重 + 矩阵/嵌入 .h)")
        f2 = QFormLayout(g2)
        self.ex_ckpt_w, self.ex_ckpt = path_row("outputs/mask/checkpoints/best.pt", "open", "ckpt (*.pt)")
        self.ex_out_w, self.ex_out = path_row("outputs/mask", "dir")
        f2.addRow("checkpoint", self.ex_ckpt_w)
        f2.addRow("输出目录", self.ex_out_w)
        btn2 = QPushButton("导出产物")
        btn2.setProperty("run", True)
        btn2.clicked.connect(self.run_export)
        f2.addRow(btn2)
        v.addWidget(g2)
        v.addStretch(1)
        return w

    # ----------------------- 动作 ----------------------- #
    def run_qa(self):
        self.run([str(ROOT / "tools/qa_dataset.py"), "--raw", self.qa_raw.text(),
                  "--sr", str(self.qa_sr.value()), "--source", self.qa_src.text()], "质检")

    def run_preprocess(self):
        args = [str(ROOT / "scripts/preprocess_mask.py"), "--config", self.pp_cfg.text(),
                "--raw", self.pp_raw.text(), "--proc", self.pp_proc.text()]
        if self.pp_limit.text().strip():
            args += ["--limit", self.pp_limit.text().strip()]
        self.run(args, "预处理")

    def run_train(self):
        args = [str(ROOT / "scripts/train_mask.py"), "--config", self.tr_cfg.text(),
                "--proc", self.tr_proc.text(), "--epochs", str(self.tr_epochs.value())]
        if self.tr_device.currentText() != "auto":
            args += ["--device", self.tr_device.currentText()]
        self.run(args, "训练")

    def run_render(self):
        self.run([str(ROOT / "scripts/render_mask.py"), "--ckpt", self.rd_ckpt.text(),
                  "--source", self.rd_src.text(), "--instrument", self.rd_instr.text(),
                  "--out", self.rd_out.text()], "渲染")

    def run_export(self):
        self.run([str(ROOT / "scripts/export_mask.py"), "--ckpt", self.ex_ckpt.text(),
                  "--out", self.ex_out.text()], "导出")

    # ----------------------- 进程控制 ----------------------- #
    def run(self, args: list[str], title: str):
        if self.proc is not None:
            self.log_line(f"[!] 已有任务在运行({self._cur_title})，请先停止。")
            return
        self._cur_title = title
        self.proc = QProcess(self)
        self.proc.setWorkingDirectory(str(ROOT))
        self.proc.setProcessChannelMode(QProcess.MergedChannels)
        env = QProcessEnvironment.systemEnvironment()
        env.insert("PYTHONIOENCODING", "utf-8")
        env.insert("PYTHONUNBUFFERED", "1")
        self.proc.setProcessEnvironment(env)
        self.proc.readyReadStandardOutput.connect(self._on_output)
        self.proc.finished.connect(self._on_finished)
        self.proc.errorOccurred.connect(self._on_error)

        self.log_line(f"\n==== [{title}] 开始 ====")
        self.log_line("> " + " ".join(f'"{a}"' if " " in a else a for a in [PY] + args))
        self._set_running(True)
        self.proc.start(PY, args)

    def _on_output(self):
        if not self.proc:
            return
        data = bytes(self.proc.readAllStandardOutput()).decode("utf-8", errors="replace")
        self.log.moveCursor(QTextCursor.End)
        self.log.insertPlainText(data)
        self.log.moveCursor(QTextCursor.End)

    def _on_error(self, err):
        self.log_line(f"[进程错误] {err}")

    def _on_finished(self, code, _status):
        self.log_line(f"==== [{self._cur_title}] 结束 (exit={code}) ====")
        self.proc = None
        self._set_running(False)

    def stop(self):
        if self.proc is not None:
            self.log_line("[停止] 正在终止任务…")
            self.proc.kill()

    def _set_running(self, running: bool):
        self.status.setText(f"运行中: {self._cur_title}…" if running else "就绪")
        self.btn_stop.setEnabled(running)
        for b in self._run_buttons:
            b.setEnabled(not running)

    def log_line(self, s: str):
        self.log.moveCursor(QTextCursor.End)
        self.log.insertPlainText(s + "\n")
        self.log.moveCursor(QTextCursor.End)


def main():
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
