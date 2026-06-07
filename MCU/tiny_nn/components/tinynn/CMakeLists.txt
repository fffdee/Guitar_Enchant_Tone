# bnn 框架已融合进本组件 (tinynn): 源码就在组件内, 自包含, 不再外部引用。
get_filename_component(TINYNN_ROOT "${CMAKE_CURRENT_LIST_DIR}" ABSOLUTE)

set(TINYNN_SRCS
    ${TINYNN_ROOT}/utils/src/bnn_mem.c
    ${TINYNN_ROOT}/utils/src/bnn_log.c
    ${TINYNN_ROOT}/utils/src/bnn_tensor.c
    ${TINYNN_ROOT}/utils/src/bnn_workspace.c
    ${TINYNN_ROOT}/operator/src/bnn_op.c
    ${TINYNN_ROOT}/operator/src/bnn_op_cpu.c
    ${TINYNN_ROOT}/operator/src/bnn_dsp_cpu.c
    ${TINYNN_ROOT}/operator/src/bnn_specmat.c
    ${TINYNN_ROOT}/operator/src/bnn_polyphony.c
    ${TINYNN_ROOT}/layer/src/bnn_layer.c
    ${TINYNN_ROOT}/layer/src/layer_activation.c
    ${TINYNN_ROOT}/layer/src/layer_conv1d.c
    ${TINYNN_ROOT}/layer/src/layer_film.c
    ${TINYNN_ROOT}/layer/src/layer_embedding.c
    ${TINYNN_ROOT}/graph/src/bnn_graph.c
    ${TINYNN_ROOT}/graph/src/bnn_graph_ir.c
    ${TINYNN_ROOT}/frontend/src/bnn_specfront.c
    ${TINYNN_ROOT}/synth/src/bnn_bands.c
    ${TINYNN_ROOT}/synth/src/bnn_specsynth.c
    ${TINYNN_ROOT}/model/src/bnn_masknet.c
    ${TINYNN_ROOT}/effect/src/bnn_fx.c
)

idf_component_register(
    SRCS ${TINYNN_SRCS}
    INCLUDE_DIRS
        ${TINYNN_ROOT}/utils/include
        ${TINYNN_ROOT}/operator/include
        ${TINYNN_ROOT}/layer/include
        ${TINYNN_ROOT}/graph/include
        ${TINYNN_ROOT}/frontend/include
        ${TINYNN_ROOT}/synth/include
        ${TINYNN_ROOT}/model/include
        ${TINYNN_ROOT}/effect/include
    # 各 layer 通过 __attribute__((constructor)) 自注册, 这些 constructor 无显式引用,
    # 会被 --gc-sections 回收导致运行时 "layer not found". WHOLE_ARCHIVE 强制整库链入,
    # 保留所有 constructor, 使 conv1d/activation/film/embedding 正常注册.
    WHOLE_ARCHIVE
)

target_compile_definitions(${COMPONENT_LIB} PUBLIC BNN_PLATFORM_MCU)
target_compile_options(${COMPONENT_LIB} PRIVATE -Wno-error=format)
