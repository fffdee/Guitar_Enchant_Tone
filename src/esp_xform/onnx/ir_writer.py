"""bnn 图 IR 二进制写入器 (纯 stdlib, 无 torch/onnx 依赖)。

格式与 C 端严格一致: MCU/tiny_nn/components/tinynn/graph/include/bnn_graph/bnn_graph_ir.h
  Header 32B: magic('BGIR') ver n_nodes n_outputs weights_off weights_len rsv rsv
  Node 84B: type[16] kind cfg[9](i32) param(f32) ndep_or_ndim a[4] extra0
  Outputs: n_outputs 个 i32 (IR 节点序号)
  Weights(可选): BNNW 字节流 (magic('BNNW') ver count(u64) f32[]), 位于 weights_off

mask 导出器与 ONNX 转换器共用此模块, 保证两条产线产出同一 IR 格式。
"""
from __future__ import annotations

import struct
from typing import List, Sequence

IR_MAGIC = 0x52494742    # 'BGIR'
IR_VERSION = 1
NODE_BYTES = 84

BNNW_MAGIC = 0x57574E42  # 'BNNW'
BNNW_VER = 1

# 激活码 (与 C layer_activation 一致)
ACT_NONE, ACT_RELU, ACT_SIGMOID, ACT_TANH, ACT_SOFTPLUS = 0, 1, 2, 3, 4


def ir_node(type_name: str, kind: int, *, inf=0, outf=0, inc=0, outc=0, k=0, st=0,
            pad=0, dil=0, act=0, param=0.0, ndep=0, a=(0, 0, 0, 0), extra0=0) -> bytes:
    """打包一个 84 字节 IR 节点 (字段顺序/偏移与 C bnn_graph_ir.c 严格一致)。"""
    t = type_name.encode("ascii")[:16].ljust(16, b"\x00")
    aa = (list(a) + [0, 0, 0, 0])[:4]
    return (t
            + struct.pack("<10i", kind, inf, outf, inc, outc, k, st, pad, dil, act)
            + struct.pack("<f", float(param))
            + struct.pack("<6i", ndep, aa[0], aa[1], aa[2], aa[3], extra0))


def input_node(shape: Sequence[int]) -> bytes:
    """输入节点 (kind=0): ndim + shape (最多 4 维)。"""
    nd = len(shape)
    return ir_node("input", 0, ndep=nd, a=tuple(shape))


def bnnw_bytes(flat_f32: bytes, count: int) -> bytes:
    """把 float32 权重字节流封装为 BNNW (供图内嵌权重)。"""
    return struct.pack("<IIQ", BNNW_MAGIC, BNNW_VER, int(count)) + flat_f32


def pack_ir(nodes: List[bytes], outputs: Sequence[int], weights: bytes = b"") -> bytes:
    """组装完整 IR. nodes 为 ir_node()/input_node() 列表, outputs 为 IR 节点序号,
    weights 为可选的 BNNW 字节流 (嵌入图内, 设备端建图时自动加载)。"""
    body = b"".join(nodes) + struct.pack("<%di" % len(outputs), *outputs)
    wlen = len(weights)
    woff = (32 + len(body)) if wlen else 0
    header = struct.pack("<8I", IR_MAGIC, IR_VERSION, len(nodes), len(outputs), woff, wlen, 0, 0)
    return header + body + weights
