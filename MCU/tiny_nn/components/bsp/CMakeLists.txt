# BSP: 仿 banux 分层的板级支持 (ESP-IDF 原生驱动)
#   02_device     设备驱动 (SD 卡 / I2S)
#   03_framework  驱动框架 (设备注册 / 初始化序列)
#   05_component  组件 (FATFS 挂载 / WAV 读写 / 模型产物加载 / DSP/NN 加速)
#   06_app        应用 (音色转换 / I2S 实时推理 / CLI)
#   07_tinynn_app 项目特有的 tinynn 扩展 (model / synth / frontend / effect / specmat / polyphony)
#                 与通用推理框架 (tinynn) 解耦, 方便框架独立复用

get_filename_component(APP_ROOT "${CMAKE_CURRENT_LIST_DIR}/07_tinynn_app" ABSOLUTE)

idf_component_register(
    SRCS
        # 02_device: 设备驱动
        "02_device/sd_card.c"
        "02_device/usb_msc.c"
        "02_device/i2s_driver.c"
        "02_device/wm8978.c"
        "02_device/es8311.c"
        # 03_framework: 驱动框架
        "03_framework/drv_device.c"
        "03_framework/drv_init.c"
        # 05_component: 组件
        "05_component/fs_mount.c"
        "05_component/wav.c"
        "05_component/model_store.c"
        "05_component/dsp_accel.c"
        "05_component/nn_accel.c"
        "05_component/audio_post.c"
        # 06_app: 应用
        "06_app/audio_xform.c"
        "06_app/infer_worker.c"
        "06_app/i2s_xform.c"
        "06_app/fx_chain.c"
        "06_app/fx_modules.c"
        "06_app/cli.c"
        # 07_tinynn_app: 项目特有的 tinynn 扩展 (从框架中解耦)
        ${APP_ROOT}/model/src/bnn_masknet.c
        ${APP_ROOT}/model/src/bnn_xform.c
        ${APP_ROOT}/synth/src/bnn_bands.c
        ${APP_ROOT}/synth/src/bnn_specsynth.c
        ${APP_ROOT}/synth/src/bnn_synth.c
        ${APP_ROOT}/frontend/src/bnn_frontend.c
        ${APP_ROOT}/frontend/src/bnn_specfront.c
        ${APP_ROOT}/effect/src/bnn_fx.c
        ${APP_ROOT}/bnn_specmat.c
        ${APP_ROOT}/bnn_polyphony.c
    INCLUDE_DIRS
        "include"
        "02_device"
        "03_framework"
        "05_component"
        "06_app"
        # 07_tinynn_app: 头文件搜索路径
        ${APP_ROOT}/include
        ${APP_ROOT}/model/include
        ${APP_ROOT}/synth/include
        ${APP_ROOT}/frontend/include
        ${APP_ROOT}/effect/include
    REQUIRES
        tinynn fatfs sdmmc esp_driver_sdmmc esp_driver_i2c vfs driver esp_psram esp_timer
        espressif__esp-dsp espressif__esp-nn console
        espressif__esp_tinyusb espressif__esp_codec_dev
)

# 把 -Werror=format 降为警告 (uint32_t=long unsigned int 在 xtensa/riscv 工具链差异)
target_compile_options(${COMPONENT_LIB} PRIVATE -Wno-error=format)
