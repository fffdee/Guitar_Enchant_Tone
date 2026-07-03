# tinynn — 纯 C 轻量神经网络推理框架 (通用层)
#
# 框架只包含与项目无关的通用能力:
#   utils/    内存池 / tensor / workspace / log
#   layer/    通用层 (dense / conv1d / conv2d / film / embedding / activation / ...)
#   graph/    计算图 IR (数据驱动建图)
#   operator/ 基础算子 (compute 后端 / DSP 后端 / NN 算子)
#
# 项目特有的代码 (model / synth / frontend / effect / specmat / polyphony)
# 放在 bsp/07_tinynn_app/ 中, 与框架解耦。
get_filename_component(TINYNN_ROOT "${CMAKE_CURRENT_LIST_DIR}" ABSOLUTE)

set(TINYNN_SRCS
    # utils: 内存 / 日志 / tensor / workspace
    ${TINYNN_ROOT}/utils/src/bnn_mem.c
    ${TINYNN_ROOT}/utils/src/bnn_log.c
    ${TINYNN_ROOT}/utils/src/bnn_tensor.c
    ${TINYNN_ROOT}/utils/src/bnn_workspace.c
    # operator: 基础算子 (compute / dsp / nn)
    ${TINYNN_ROOT}/operator/src/bnn_op.c
    ${TINYNN_ROOT}/operator/src/bnn_op_cpu.c
    ${TINYNN_ROOT}/operator/src/bnn_dsp_cpu.c
    ${TINYNN_ROOT}/operator/src/bnn_nn.c
    # layer: 通用层
    ${TINYNN_ROOT}/layer/src/bnn_layer.c
    ${TINYNN_ROOT}/layer/src/layer_activation.c
    ${TINYNN_ROOT}/layer/src/layer_conv1d.c
    ${TINYNN_ROOT}/layer/src/layer_film.c
    ${TINYNN_ROOT}/layer/src/layer_embedding.c
    # graph: 计算图 IR
    ${TINYNN_ROOT}/graph/src/bnn_graph.c
    ${TINYNN_ROOT}/graph/src/bnn_graph_ir.c
)

# 双核并行辅助模块 (仅在 BNN_PERF_DUAL_CORE=1 时有实际代码)
if(BNN_PERF_DUAL_CORE)
    list(APPEND TINYNN_SRCS ${TINYNN_ROOT}/operator/src/bnn_dual_core.c)
endif()

idf_component_register(
    SRCS ${TINYNN_SRCS}
    INCLUDE_DIRS
        ${TINYNN_ROOT}/utils/include
        ${TINYNN_ROOT}/operator/include
        ${TINYNN_ROOT}/layer/include
        ${TINYNN_ROOT}/graph/include
    # 各 layer 通过 __attribute__((constructor)) 自注册, 这些 constructor 无显式引用,
    # 会被 --gc-sections 回收导致运行时 "layer not found". WHOLE_ARCHIVE 强制整库链入,
    # 保留所有 constructor, 使 conv1d/activation/film/embedding 正常注册.
    WHOLE_ARCHIVE
)

target_compile_definitions(${COMPONENT_LIB} PUBLIC BNN_PLATFORM_MCU)

# 当目标为 ESP32-P4 时启用 INT8 ESP-NN conv 分支
# 依赖: bsp 组件提供 bnn_nn_set_backend(nn_accel_nn_backend()) 注册
if(CONFIG_IDF_TARGET_ESP32P4)
    target_compile_definitions(${COMPONENT_LIB} PUBLIC BNN_CONV1D_INT8_ACCEL)
endif()

# ============================================================
# 性能优化开关 (通过 CMake 变量或 sdkconfig 控制)
# ============================================================
# 启用方式: 在顶层 CMakeLists.txt 或 idf.py 命令行设置
#   idf.py -DBNN_PERF_DUAL_CORE=1 -DBNN_PERF_WEIGHT_PREFETCH=1 build
# 或在 sdkconfig.h 中定义对应宏
#
# 默认全部关闭 (原始单核 F32 路径), 方便回退对比

if(DEFINED BNN_PERF_DUAL_CORE AND BNN_PERF_DUAL_CORE)
    target_compile_definitions(${COMPONENT_LIB} PUBLIC BNN_PERF_DUAL_CORE=1)
    message(STATUS "[bnn_perf] DUAL_CORE=ON")
endif()

if(DEFINED BNN_PERF_WEIGHT_PREFETCH AND BNN_PERF_WEIGHT_PREFETCH)
    target_compile_definitions(${COMPONENT_LIB} PUBLIC BNN_PERF_WEIGHT_PREFETCH=1)
    message(STATUS "[bnn_perf] WEIGHT_PREFETCH=ON")
endif()

if(DEFINED BNN_PERF_FUSE_BIAS_RELU AND BNN_PERF_FUSE_BIAS_RELU)
    target_compile_definitions(${COMPONENT_LIB} PUBLIC BNN_PERF_FUSE_BIAS_RELU=1)
    message(STATUS "[bnn_perf] FUSE_BIAS_RELU=ON")
endif()

target_compile_options(${COMPONENT_LIB} PRIVATE -Wno-error=format)
