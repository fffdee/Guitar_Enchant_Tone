# BSP: 仿 banux 分层的板级支持 (ESP-IDF 原生驱动)
#   02_device   设备驱动 (SD 卡)
#   03_framework 驱动框架 (设备注册 / 初始化序列)
#   05_component 组件 (FATFS 挂载 / WAV 读写 / 模型产物加载)
#   06_app      应用 (音色转换: 读wav→推理→写wav)
idf_component_register(
    SRCS
        "02_device/sd_card.c"
        "03_framework/drv_device.c"
        "03_framework/drv_init.c"
        "05_component/fs_mount.c"
        "05_component/wav.c"
        "05_component/model_store.c"
        "05_component/dsp_accel.c"
        "05_component/nn_accel.c"
        "05_component/audio_post.c"
        "06_app/audio_xform.c"
        "06_app/infer_worker.c"
        "06_app/cli.c"
    INCLUDE_DIRS
        "include"
        "02_device"
        "03_framework"
        "05_component"
        "06_app"
    REQUIRES
        tinynn fatfs sdmmc esp_driver_sdmmc vfs driver esp_psram esp_timer
        espressif__esp-dsp espressif__esp-nn console
)

# 把 -Werror=format 降为警告 (uint32_t=long unsigned int 在 xtensa/riscv 工具链差异)
target_compile_options(${COMPONENT_LIB} PRIVATE -Wno-error=format)
