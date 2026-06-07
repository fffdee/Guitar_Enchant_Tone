"""ONNX -> bnn 图 IR 转换器 (混合 ONNX 路线 PC 侧)。

读取标准 .onnx, 把支持的算子映射为 bnn 层, 产出自包含的图 IR(.bin, 含权重),
设备端用 bnn_graph_build_from_ir 动态建图运行 —— 无需为每个网络改固件。

用法:
  python tools/onnx_to_bnn_ir.py model.onnx -o model_ir.bin

依赖: pip install onnx numpy
支持算子: Conv(1D/2D) / Relu / Sigmoid / Tanh / Softplus / Add(残差) / Gemm / MatMul / Flatten
不支持的算子会明确报错并列出。

权重顺序: 按节点拓扑序、层内 W 后 b, 与 C bnn_graph 收集参数的顺序一致。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from esp_xform.onnx import ir_writer as iw  # noqa: E402

_ACT = {"Relu": iw.ACT_RELU, "Sigmoid": iw.ACT_SIGMOID, "Tanh": iw.ACT_TANH, "Softplus": iw.ACT_SOFTPLUS}


def _attr(node, name, default=None):
    for a in node.attribute:
        if a.name == name:
            if a.ints:    return list(a.ints)
            if a.floats:  return list(a.floats)
            if a.HasField("i"): return a.i
            if a.HasField("f"): return a.f
            if a.HasField("s"): return a.s
    return default


def convert(onnx_path: str, out_path: str) -> None:
    import onnx
    from onnx import numpy_helper

    model = onnx.load(onnx_path)
    g = model.graph

    inits = {t.name: numpy_helper.to_array(t) for t in g.initializer}   # 权重
    input_names = [i.name for i in g.input if i.name not in inits]      # 真正的图输入

    # tensor 名 -> 产出它的 IR 节点序号
    name2idx: dict[str, int] = {}
    nodes: list[bytes] = []
    weights: list[np.ndarray] = []
    unsupported: list[str] = []

    # 1) 输入节点
    for iname in input_names:
        vi = next((v for v in g.input if v.name == iname), None)
        shape = []
        if vi is not None and vi.type.tensor_type.shape.dim:
            for d in vi.type.tensor_type.shape.dim:
                shape.append(int(d.dim_value) if d.dim_value > 0 else 1)
        if not shape:
            shape = [1]
        name2idx[iname] = len(nodes)
        nodes.append(iw.input_node(shape[:4]))

    def dep_indices(node):
        deps = []
        for inp in node.input:
            if inp in inits or inp == "":
                continue
            if inp not in name2idx:
                raise ValueError(f"算子 {node.op_type} 的输入 '{inp}' 未知 (非初始化器/非已知节点)")
            deps.append(name2idx[inp])
        return deps

    # 2) 算子节点
    for node in g.node:
        op = node.op_type
        deps = dep_indices(node)

        if op == "Conv":
            w = inits[node.input[1]]                     # [out, in/group, kH, (kW)]
            b = inits[node.input[2]] if len(node.input) > 2 and node.input[2] in inits else None
            ksh = _attr(node, "kernel_shape") or list(w.shape[2:])
            strides = _attr(node, "strides") or [1] * len(ksh)
            pads = _attr(node, "pads") or [0] * (2 * len(ksh))
            dil = _attr(node, "dilations") or [1] * len(ksh)
            is1d = (w.ndim == 3) or (len(ksh) == 1)
            ltype = "conv1d" if is1d else "conv2d"
            nodes.append(iw.ir_node(ltype, 1, inc=int(w.shape[1]), outc=int(w.shape[0]),
                                    k=int(ksh[0]), st=int(strides[0]), pad=int(pads[0]),
                                    dil=int(dil[0]), ndep=len(deps), a=tuple(deps)))
            weights.append(w.astype("<f4").reshape(-1))
            weights.append((b if b is not None else np.zeros(w.shape[0])).astype("<f4").reshape(-1))

        elif op in _ACT:
            nodes.append(iw.ir_node("activation", 1, act=_ACT[op], ndep=len(deps), a=tuple(deps)))

        elif op == "Add" and len(deps) == 2:
            nodes.append(iw.ir_node("residual", 1, ndep=2, a=tuple(deps)))

        elif op in ("Gemm", "MatMul"):
            w = inits[node.input[1]]                     # [in,out] 或 [out,in]
            transB = _attr(node, "transB", 0)
            out_f, in_f = (w.shape[0], w.shape[1]) if transB else (w.shape[1], w.shape[0])
            b = inits[node.input[2]] if len(node.input) > 2 and node.input[2] in inits else None
            nodes.append(iw.ir_node("dense", 1, inf=int(in_f), outf=int(out_f),
                                    ndep=len(deps), a=tuple(deps)))
            wmat = w if transB else w.T                  # 统一存为 [out,in]
            weights.append(np.ascontiguousarray(wmat).astype("<f4").reshape(-1))
            weights.append((b if b is not None else np.zeros(out_f)).astype("<f4").reshape(-1))

        elif op == "Flatten":
            nodes.append(iw.ir_node("flatten", 1, ndep=len(deps), a=tuple(deps)))

        else:
            unsupported.append(op)
            continue

        for o in node.output:
            name2idx[o] = len(nodes) - 1

    if unsupported:
        raise SystemExit(f"[onnx_to_bnn_ir] 不支持的算子: {sorted(set(unsupported))}\n"
                         f"  已支持: Conv/Relu/Sigmoid/Tanh/Softplus/Add/Gemm/MatMul/Flatten")

    # 3) 输出节点
    outputs = []
    for o in g.output:
        if o.name not in name2idx:
            raise SystemExit(f"[onnx_to_bnn_ir] 输出 '{o.name}' 未映射")
        outputs.append(name2idx[o.name])

    # 4) 权重打包 (BNNW, 顺序与节点拓扑序一致)
    flat = np.concatenate(weights).astype("<f4") if weights else np.zeros(0, "<f4")
    bnnw = iw.bnnw_bytes(flat.tobytes(), int(flat.size))
    ir = iw.pack_ir(nodes, outputs, bnnw)

    Path(out_path).write_bytes(ir)
    print(f"[onnx_to_bnn_ir] 写出 {out_path}: 节点={len(nodes)} 输出={len(outputs)} "
          f"权重={flat.size} floats ({len(ir)} bytes)")


def main():
    ap = argparse.ArgumentParser(description="ONNX -> bnn 图 IR(.bin)")
    ap.add_argument("onnx", help="输入 .onnx 路径")
    ap.add_argument("-o", "--out", required=True, help="输出 IR .bin 路径")
    args = ap.parse_args()
    convert(args.onnx, args.out)


if __name__ == "__main__":
    main()
