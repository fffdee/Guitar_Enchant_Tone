"""WAV 读写（仅用 Python 标准库 wave + numpy）。

规格：项目固定 48 kHz / 16-bit / mono。为稳健起见，读取时兼容 8/16/24/32-bit
与多声道（自动混为单声道），写出统一为 16-bit PCM。
"""

from __future__ import annotations

import wave
from pathlib import Path
from typing import Tuple

import numpy as np


def _bytes_to_float(raw: bytes, sampwidth: int, n_channels: int) -> np.ndarray:
    """将 PCM 字节解码为 float32 数组，形状 (n_samples, n_channels)，范围约 [-1, 1)。"""
    if sampwidth == 1:  # 8-bit PCM 为无符号
        data = np.frombuffer(raw, dtype=np.uint8).astype(np.float32)
        data = (data - 128.0) / 128.0
    elif sampwidth == 2:  # 16-bit
        data = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    elif sampwidth == 3:  # 24-bit packed little-endian
        a = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3).astype(np.int32)
        val = a[:, 0] | (a[:, 1] << 8) | (a[:, 2] << 16)
        neg = val >= (1 << 23)
        val[neg] -= 1 << 24
        data = val.astype(np.float32) / float(1 << 23)
    elif sampwidth == 4:  # 32-bit int
        data = np.frombuffer(raw, dtype="<i4").astype(np.float32) / float(1 << 31)
    else:
        raise ValueError(f"不支持的采样位宽: {sampwidth * 8} bit")

    if n_channels > 1:
        data = data.reshape(-1, n_channels)
    else:
        data = data.reshape(-1, 1)
    return data


def load_wav(
    path: str | Path,
    expected_sr: int | None = None,
    mono: bool = True,
) -> Tuple[np.ndarray, int]:
    """读取 WAV，返回 (float32 音频[-1,1], 采样率)。

    - mono=True 时多声道取均值。
    - expected_sr 给定且不一致时抛错（本框架不做重采样，数据须 48 kHz）。
    """
    path = Path(path)
    with wave.open(str(path), "rb") as wf:
        n_channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        sr = wf.getframerate()
        n_frames = wf.getnframes()
        raw = wf.readframes(n_frames)

    audio = _bytes_to_float(raw, sampwidth, n_channels)  # (N, C)
    if mono:
        audio = audio.mean(axis=1)
    else:
        audio = audio.squeeze()

    if expected_sr is not None and sr != expected_sr:
        raise ValueError(
            f"{path.name} 采样率为 {sr} Hz，期望 {expected_sr} Hz。"
            " 请在 DAW 中以 48 kHz 重新导出，本框架不做重采样。"
        )
    return audio.astype(np.float32), sr


def save_wav(
    path: str | Path,
    audio: np.ndarray,
    sample_rate: int,
    subtype_bits: int = 16,
) -> None:
    """写出 16-bit PCM mono WAV。audio 为 float[-1,1]，自动裁剪防爆音。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    audio = np.asarray(audio, dtype=np.float32).squeeze()
    if audio.ndim > 1:  # 多声道→单声道
        audio = audio.mean(axis=-1)
    audio = np.clip(audio, -1.0, 1.0)

    if subtype_bits != 16:
        raise NotImplementedError("当前仅支持写出 16-bit PCM。")
    pcm = (audio * 32767.0).astype("<i2")

    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(int(sample_rate))
        wf.writeframes(pcm.tobytes())


__all__ = ["load_wav", "save_wav"]
