#include "cli.h"
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <dirent.h>
#include <sys/stat.h>
#include "esp_console.h"
#include "esp_log.h"
#include "esp_heap_caps.h"
#include "esp_timer.h"
#include <unistd.h>
#include <fcntl.h>
#include <errno.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "bsp.h"
#include "wav.h"
#include "audio_xform.h"
#include "infer_worker.h"
#include "i2s_driver.h"
#include "i2s_xform.h"
#include "fx_chain.h"
#include "fx_modules.h"
#include "wm8978.h"
#include "es8311.h"
#include "usb_msc.h"

static const char *TAG = "cli";

#define DEF_IN_WAV   BSP_SD_MOUNT "/in/guitar.wav"
#define DEF_OUT_DIR  BSP_SD_MOUNT "/out"

/* ---------- model: 打印模型信息 ---------- */
static int cmd_model(int argc, char **argv)
{
    (void)argc; (void)argv;
    if (!audio_xform_loaded()) { printf("模型未加载 (检查 %s/model/xform_model.bin)\n", BSP_SD_MOUNT); return 0; }
    audio_xform_print_model();
    return 0;
}

/* ---------- status: worker 状态 + 堆 ---------- */
static int cmd_status(int argc, char **argv)
{
    (void)argc; (void)argv;
    char st[160];
    infer_worker_status(st, sizeof(st));
    printf("%s\n", st);
    /* I2S 实时推理状态 */
    if (i2s_xform_running()) {
        char ist[160];
        i2s_xform_status(ist, sizeof(ist));
        printf("%s\n", ist);
    }
    /* 效果器链状态 */
    if (fx_chain_running()) {
        char fst[160];
        fx_chain_status(fst, sizeof(fst));
        printf("%s\n", fst);
    }
    printf("PSRAM free=%uKB  内部RAM free=%uKB\n",
           (unsigned)(heap_caps_get_free_size(MALLOC_CAP_SPIRAM) / 1024),
           (unsigned)(heap_caps_get_free_size(MALLOC_CAP_INTERNAL) / 1024));
    return 0;
}

/* ---------- stats [reset] ---------- */
static int cmd_stats(int argc, char **argv)
{
    if (argc >= 2 && !strcmp(argv[1], "reset")) {
        audio_xform_reset_stats();
        printf("推理统计已清零\n");
        return 0;
    }
    audio_xform_print_stats();
    return 0;
}

/* ---------- ls [dir] ---------- */
static int cmd_ls(int argc, char **argv)
{
    const char *path = (argc >= 2) ? argv[1] : BSP_SD_MOUNT;
    DIR *d = opendir(path);
    if (!d) { printf("打不开目录: %s\n", path); return 1; }
    printf("目录 %s:\n", path);
    struct dirent *e;
    while ((e = readdir(d)) != NULL) {
        char full[512];
        snprintf(full, sizeof(full), "%s/%s", path, e->d_name);
        struct stat s;
        if (stat(full, &s) == 0)
            printf("  %s%s  %ld bytes\n", e->d_name, (e->d_type == DT_DIR ? "/" : ""), (long)s.st_size);
        else
            printf("  %s%s\n", e->d_name, (e->d_type == DT_DIR ? "/" : ""));
    }
    closedir(d);
    return 0;
}

/* ---------- sdinfo ---------- */
static int cmd_sdinfo(int argc, char **argv)
{
    (void)argc; (void)argv;
    struct stat s;
    if (stat(BSP_SD_MOUNT, &s) == 0) printf("SD 已挂载于 %s\n", BSP_SD_MOUNT);
    else printf("SD 未挂载\n");
    return 0;
}

/* ---------- usb app|host|status ---------- */
static int cmd_usb(int argc, char **argv)
{
    if (argc < 2 || !strcmp(argv[1], "status")) {
        printf("USB: ready=%d attached=%d cdc=%d busy=%d port=%d mount=%s\n",
               usb_msc_is_ready() ? 1 : 0,
               usb_msc_is_attached() ? 1 : 0,
               usb_msc_console_ready() ? 1 : 0,
               usb_msc_is_busy() ? 1 : 0,
               usb_msc_port(),
               usb_msc_mount_name());
        printf("用法: usb app | usb host | usb status\n");
        return 0;
    }

    esp_err_t err;
    if (!strcmp(argv[1], "app")) {
        if (!usb_msc_is_ready()) {
            printf("TinyUSB 未初始化, 当前只能尝试普通 APP /sdcard 模式\n");
            return 0;
        }
        err = usb_msc_mount_app();
        if (err == ESP_OK) {
            printf("SD 已切换到 APP 模式: MCU 可访问 %s\n", BSP_SD_MOUNT);
            return 0;
        }
        printf("切换到 APP 失败: %s\n", esp_err_to_name(err));
        return 1;
    }

    if (!strcmp(argv[1], "host")) {
        if (infer_worker_busy()) {
            printf("正在推理, 不能切换到 USB Host\n");
            return 1;
        }
        if (!usb_msc_is_ready()) {
            printf("TinyUSB 未初始化, 不能切换到 USB Host\n");
            return 1;
        }
        err = usb_msc_mount_usb();
        if (err == ESP_OK) {
            if (usb_msc_is_attached()) {
                printf("SD 已切换到 USB Host 模式: PC 可访问 U 盘; 完成后请安全弹出并执行 usb app\n");
            } else {
                printf("SD 已切换到 USB Host 模式, 但 TinyUSB 设备尚未被 PC 枚举(attached=0); 请确认连接的是 USB-OTG/HS 口而不是下载/JTAG 口\n");
            }
            return 0;
        }
        printf("切换到 USB Host 失败: %s\n", esp_err_to_name(err));
        return 1;
    }

    printf("未知 usb 子命令: %s (用法: usb app | usb host | usb status)\n", argv[1]);
    return 1;
}

static audio_clip_mode_t parse_clip(const char *s)
{
    if (!s) return AUDIO_CLIP_LIMIT;
    if (!strcmp(s, "soft")) return AUDIO_CLIP_SOFT;
    if (!strcmp(s, "hard")) return AUDIO_CLIP_HARD;
    return AUDIO_CLIP_LIMIT;
}

/* ---------- infer <instrument> [-i in][-o out][-p pitch][-g gain][-c clip] ---------- */
static int cmd_infer(int argc, char **argv)
{
    if (argc < 2) {
        printf("用法: infer <乐器> [-i in][-o out][-p 变调][-g 增益dB][-c limit|soft|hard][-N 开噪声][-8 INT8加速]\n");
        printf("例:   infer bass -g 9 -c soft -N   (对齐上位机 guitar__to_bass_g+9_soft)\n");
        printf("      infer bass -8               (INT8 conv, 需 bin 含 weights_i8)\n");
        printf("      infer bass                   (默认 F32)\n");
        printf("验证: 串口 mask mean 应≈3.5~4.0; 若≈2.0 且 noise_t 全 0.693 → 需刷 F32 固件+正确 bin\n");
        return 1;
    }
    if (!audio_xform_loaded()) { printf("模型未加载, 无法推理\n"); return 1; }

    infer_job_t job;
    memset(&job, 0, sizeof(job));
    strncpy(job.instrument, argv[1], sizeof(job.instrument) - 1);

    const char *in = DEF_IN_WAV;
    const char *out = NULL;
    for (int i = 2; i < argc; ) {
        if (!strcmp(argv[i], "-N")) {
            /* 无值标志: 开启噪声注入 */
            job.opt.add_noise = 1;
            i++; continue;
        }
        if (!strcmp(argv[i], "-8")) {
            job.opt.use_int8 = 1;
            i++; continue;
        }
        if (i + 1 >= argc) { printf("参数 %s 缺少值\n", argv[i]); return 1; }
        if      (!strcmp(argv[i], "-i")) in = argv[i + 1];
        else if (!strcmp(argv[i], "-o")) out = argv[i + 1];
        else if (!strcmp(argv[i], "-p")) job.opt.pitch_semitones = strtof(argv[i + 1], NULL);
        else if (!strcmp(argv[i], "-g")) job.opt.gain_db = strtof(argv[i + 1], NULL);
        else if (!strcmp(argv[i], "-c")) job.opt.clip_mode = parse_clip(argv[i + 1]);
        else { printf("未知参数: %s\n", argv[i]); return 1; }
        i += 2;
    }
    strncpy(job.in_path, in, sizeof(job.in_path) - 1);
    if (out) strncpy(job.out_path, out, sizeof(job.out_path) - 1);
    else snprintf(job.out_path, sizeof(job.out_path), "%s/%s.wav", DEF_OUT_DIR, job.instrument);

    if (infer_worker_submit(&job) != ESP_OK) { printf("入队失败 (队列满?)\n"); return 1; }
    printf("已入队: %s -> %s (pitch %+.1f, gain %+.1fdB, clip %d%s). 用 status 查看进度.\n",
           job.instrument, job.out_path, job.opt.pitch_semitones, job.opt.gain_db,
           (int)job.opt.clip_mode, job.opt.use_int8 ? ", INT8" : "");
    return 0;
}

/* ---------- mode sd|i2s|fx ---------- */
static int cmd_mode(int argc, char **argv)
{
    if (argc < 2) {
        printf("用法: mode sd | mode i2s | mode fx\n");
        printf("  sd  - SD 卡文件模式 (读WAV -> 推理 -> 写WAV)\n");
        printf("  i2s - I2S 实时音色转换 (I2S RX -> 神经网络推理 -> I2S TX)\n");
        printf("  fx  - I2S 实时效果器链 (I2S RX -> DSP效果器链 -> I2S TX)\n");
        return 1;
    }

    if (!strcmp(argv[1], "sd")) {
        /* 停止 I2S 实时推理与效果器链 (如果正在运行) */
        if (i2s_xform_running()) {
            i2s_xform_stop();
            printf("I2S 实时推理已停止\n");
        }
        if (fx_chain_running()) {
            fx_chain_stop();
            printf("效果器链已停止\n");
        }
        /* 确保 SD 推理 worker 已启动 */
        if (!infer_worker_busy()) {
            infer_worker_start();
        }
        printf("已切换到 SD 卡文件模式\n");
        return 0;
    }

    if (!strcmp(argv[1], "i2s")) {
        if (!audio_xform_loaded()) { printf("模型未加载, 无法使用 I2S 模式\n"); return 1; }
        /* 互斥: 停止效果器链 */
        if (fx_chain_running()) {
            fx_chain_stop();
            printf("效果器链已停止 (与音色转换互斥)\n");
        }
        /* 初始化 I2S 驱动 (如果尚未初始化) */
        if (!i2s_driver_is_ready()) {
            esp_err_t err = i2s_driver_init(0);
            if (err != ESP_OK) {
                printf("I2S 初始化失败: %s\n", esp_err_to_name(err));
                return 1;
            }
        }
        printf("已切换到 I2S 实时音色转换模式, 使用 'live <乐器>' 开始\n");
        return 0;
    }

    if (!strcmp(argv[1], "fx")) {
        /* 互斥: 停止音色转换 */
        if (i2s_xform_running()) {
            i2s_xform_stop();
            printf("I2S 实时推理已停止 (与效果器链独立模式互斥)\n");
        }
        /* 初始化 I2S 驱动 (如果尚未初始化) */
        if (!i2s_driver_is_ready()) {
            esp_err_t err = i2s_driver_init(0);
            if (err != ESP_OK) {
                printf("I2S 初始化失败: %s\n", esp_err_to_name(err));
                return 1;
            }
        }
        printf("已切换到 I2S 实时效果器链模式 (独立)\n");
        printf("  'fx add <类型>' 添加效果器, 'fx preset <名称>' 加载预设\n");
        printf("  'fx start' 启动独立实时处理, 'fx list' 查看链, 'fx stop' 停止\n");
        printf("  串联模式: 'live <乐器> -x' 或 'live -B -x' (音色转换+效果器链)\n");
        return 0;
    }

    printf("未知模式: %s (用法: mode sd | mode i2s | mode fx)\n", argv[1]);
    return 1;
}

/* ---------- live <instrument|stop|bypass|postfx> [-p pitch][-g gain][-c clip] ---------- */
static int cmd_live(int argc, char **argv)
{
    if (argc < 2) {
        printf("用法: live <乐器> [-p 变调][-g 增益dB][-c limit|soft|hard][-N 开噪声][-8 INT8][-x 开效果器链]\n");
        printf("      live stop                  停止实时推理\n");
        printf("      live instrument <名称>     运行时切换乐器\n");
        printf("      live bypass on|off         音色转换旁通开关\n");
        printf("      live postfx on|off         效果器链后处理开关\n");
        printf("例:   live bass -p -12 -g 9 -c soft -x\n");
        printf("      live bypass on             (旁通音色转换, 纯效果器链)\n");
        return 1;
    }

    /* live stop: 停止实时推理 */
    if (!strcmp(argv[1], "stop")) {
        if (!i2s_xform_running()) { printf("实时推理未在运行\n"); return 0; }
        i2s_xform_stop();
        printf("实时推理已停止\n");
        return 0;
    }

    /* live bypass on|off: 音色转换旁通 */
    if (!strcmp(argv[1], "bypass")) {
        if (argc < 3) {
            printf("当前音色转换: %s\n", i2s_xform_running() ? "(未知)" : "未运行");
            printf("用法: live bypass on|off\n");
            return 1;
        }
        if (!i2s_xform_running()) { printf("实时推理未在运行\n"); return 1; }
        int on = (!strcmp(argv[2], "on") || !strcmp(argv[2], "1")) ? 1 : 0;
        i2s_xform_set_bypass(on);
        printf("音色转换 %s\n", on ? "已旁通 (输入直通效果器链)" : "已启用");
        return 0;
    }

    /* live postfx on|off: 效果器链后处理开关 */
    if (!strcmp(argv[1], "postfx")) {
        if (argc < 3) {
            printf("用法: live postfx on|off\n");
            return 1;
        }
        if (!i2s_xform_running()) { printf("实时推理未在运行\n"); return 1; }
        int on = (!strcmp(argv[2], "on") || !strcmp(argv[2], "1")) ? 1 : 0;
        i2s_xform_set_post_fx(on);
        printf("效果器链后处理 %s\n", on ? "已启用" : "已关闭");
        return 0;
    }

    /* live instrument <名称>: 运行时切换乐器 */
    if (!strcmp(argv[1], "instrument") || !strcmp(argv[1], "inst")) {
        if (argc < 3) { printf("用法: live instrument <乐器名>\n"); return 1; }
        if (!i2s_xform_running()) { printf("实时推理未在运行, 请先 'live <乐器>' 启动\n"); return 1; }
        esp_err_t err = i2s_xform_set_instrument(argv[2]);
        if (err == ESP_ERR_NOT_FOUND) {
            printf("乐器 '%s' 不在模型中\n", argv[2]);
            return 1;
        }
        printf("已切换到: %s\n", argv[2]);
        return 0;
    }

    /* live <乐器>: 启动实时推理 */
    /* 旁通模式不要求模型加载; 非旁通模式要求模型 */
    int bypass_mode = 0;
    int post_fx = 0;
    /* 先扫描 -x 和 -B 标志 (无值参数) */
    for (int i = 2; i < argc; i++) {
        if (!strcmp(argv[i], "-x")) post_fx = 1;
        if (!strcmp(argv[i], "-B")) bypass_mode = 1;
    }
    if (!bypass_mode && !audio_xform_loaded()) { printf("模型未加载 (旁通模式可用 -B)\n"); return 1; }
    if (i2s_xform_running()) {
        /* 已在运行, 切换乐器 */
        esp_err_t err = i2s_xform_set_instrument(argv[1]);
        if (err == ESP_ERR_NOT_FOUND) {
            printf("乐器 '%s' 不在模型中\n", argv[1]);
            return 1;
        }
        printf("已切换到: %s\n", argv[1]);
        return 0;
    }

    /* 确保 I2S 已初始化 */
    if (!i2s_driver_is_ready()) {
        esp_err_t err = i2s_driver_init(0);
        if (err != ESP_OK) {
            printf("I2S 初始化失败: %s\n", esp_err_to_name(err));
            return 1;
        }
    }

    i2s_xform_cfg_t cfg;
    memset(&cfg, 0, sizeof(cfg));
    strncpy(cfg.instrument, argv[1], sizeof(cfg.instrument) - 1);
    cfg.clip_mode = AUDIO_CLIP_LIMIT;
    cfg.bypass = bypass_mode;
    cfg.post_fx = post_fx;

    for (int i = 2; i < argc; ) {
        if (!strcmp(argv[i], "-N")) {
            cfg.add_noise = 1;
            i++; continue;
        }
        if (!strcmp(argv[i], "-8")) {
            cfg.use_int8 = 1;
            i++; continue;
        }
        if (!strcmp(argv[i], "-x")) {
            /* 已扫描过 */
            i++; continue;
        }
        if (!strcmp(argv[i], "-B")) {
            /* 已扫描过 */
            i++; continue;
        }
        if (i + 1 >= argc) { printf("参数 %s 缺少值\n", argv[i]); return 1; }
        if      (!strcmp(argv[i], "-p")) cfg.pitch_semitones = strtof(argv[i + 1], NULL);
        else if (!strcmp(argv[i], "-g")) cfg.gain_db = strtof(argv[i + 1], NULL);
        else if (!strcmp(argv[i], "-c")) cfg.clip_mode = parse_clip(argv[i + 1]);
        else { printf("未知参数: %s\n", argv[i]); return 1; }
        i += 2;
    }

    esp_err_t err = i2s_xform_start(&cfg);
    if (err != ESP_OK) {
        printf("启动失败: %s\n", esp_err_to_name(err));
        return 1;
    }
    printf("实时推理已启动: %s%s (pitch %+.1f, gain %+.1fdB, clip %d%s%s)\n",
           cfg.bypass ? "(旁通)" : cfg.instrument,
           cfg.post_fx ? " +FX" : "",
           cfg.pitch_semitones, cfg.gain_db, (int)cfg.clip_mode,
           cfg.use_int8 ? ", INT8" : "", cfg.add_noise ? ", 噪声" : "");
    printf("  'live stop' 停止, 'live bypass on|off' 旁通音色转换, 'live postfx on|off' 效果器链\n");
    printf("  'fx add|set|preset' 管理效果器链 (效果器链在两种模式下都可用)\n");
    return 0;
}

/* ---------- i2s init|deinit|info ---------- */
static int cmd_i2s(int argc, char **argv)
{
    if (argc < 2 || !strcmp(argv[1], "info")) {
        printf("I2S: ready=%d  running=%d\n",
               i2s_driver_is_ready() ? 1 : 0,
               i2s_xform_running() ? 1 : 0);
        printf("引脚: MCLK=%d BCLK=%d WS=%d DOUT=%d DIN=%d\n",
               I2S_MCLK, I2S_BCLK, I2S_WS, I2S_DOUT, I2S_DIN);
        return 0;
    }
    if (!strcmp(argv[1], "init")) {
        int sr = 0;
        if (argc >= 3) sr = atoi(argv[2]);
        esp_err_t err = i2s_driver_init(sr);
        if (err == ESP_OK) printf("I2S 初始化成功 (sr=%d)\n", sr > 0 ? sr : 48000);
        else printf("I2S 初始化失败: %s\n", esp_err_to_name(err));
        return (err == ESP_OK) ? 0 : 1;
    }
    if (!strcmp(argv[1], "deinit")) {
        if (i2s_xform_running()) i2s_xform_stop();
        i2s_driver_deinit();
        printf("I2S 已释放\n");
        return 0;
    }
    printf("用法: i2s init [sr] | i2s deinit | i2s info\n");
    return 1;
}

/* ---------- fx <subcommand> [args] : 效果器链控制 ---------- */
static int cmd_fx(int argc, char **argv)
{
    if (argc < 2 || !strcmp(argv[1], "list")) {
        fx_chain_print();
        return 0;
    }

    /* fx types : 列出可用效果器类型 */
    if (!strcmp(argv[1], "types")) {
        char buf[128];
        fx_list_types(buf, sizeof(buf));
        printf("可用效果器: %s\n", buf);
        return 0;
    }

    /* fx presets : 列出可用预设 */
    if (!strcmp(argv[1], "presets")) {
        char buf[128];
        fx_chain_list_presets(buf, sizeof(buf));
        printf("可用预设: %s\n", buf);
        return 0;
    }

    /* fx start [-g in_gain_db] [-G out_gain_db] : 启动实时处理 */
    if (!strcmp(argv[1], "start")) {
        if (fx_chain_running()) { printf("效果器链已在运行\n"); return 0; }
        if (i2s_xform_running()) {
            printf("音色转换正在运行, 请先 'live stop'\n");
            return 1;
        }
        if (!i2s_driver_is_ready()) {
            esp_err_t err = i2s_driver_init(0);
            if (err != ESP_OK) { printf("I2S 初始化失败: %s\n", esp_err_to_name(err)); return 1; }
        }
        fx_chain_cfg_t cfg;
        memset(&cfg, 0, sizeof(cfg));
        cfg.sample_rate = I2S_DEFAULT_SAMPLE_RATE;
        cfg.hop_size = FX_HOP_DEFAULT;
        for (int i = 2; i < argc; ) {
            if (i + 1 >= argc) { printf("参数 %s 缺少值\n", argv[i]); return 1; }
            if      (!strcmp(argv[i], "-g")) cfg.input_gain_db  = strtof(argv[i + 1], NULL);
            else if (!strcmp(argv[i], "-G")) cfg.output_gain_db = strtof(argv[i + 1], NULL);
            else { printf("未知参数: %s\n", argv[i]); return 1; }
            i += 2;
        }
        esp_err_t err = fx_chain_start(&cfg);
        if (err != ESP_OK) { printf("启动失败: %s\n", esp_err_to_name(err)); return 1; }
        printf("效果器链已启动 (in=%+.1fdB out=%+.1fdB)\n", cfg.input_gain_db, cfg.output_gain_db);
        printf("  'fx stop' 停止, 'fx list' 查看链, 'fx set <idx> <参数> <值>' 调参\n");
        return 0;
    }

    /* fx stop : 停止 */
    if (!strcmp(argv[1], "stop")) {
        if (!fx_chain_running()) { printf("效果器链未运行\n"); return 0; }
        fx_chain_stop();
        printf("效果器链已停止\n");
        return 0;
    }

    /* fx add <type> : 添加效果器到链尾 */
    if (!strcmp(argv[1], "add")) {
        if (argc < 3) { printf("用法: fx add <类型>\n"); return 1; }
        fx_type_t t = fx_type_from_name(argv[2]);
        if (t == FX_NONE) { printf("未知类型: %s (用 'fx types' 查看可用类型)\n", argv[2]); return 1; }
        int idx = fx_chain_add(t);
        if (idx < 0) { printf("添加失败 (链满?)\n"); return 1; }
        printf("已添加 [%d] %s\n", idx, fx_type_name(t));
        return 0;
    }

    /* fx insert <idx> <type> : 在指定位置插入 */
    if (!strcmp(argv[1], "insert")) {
        if (argc < 4) { printf("用法: fx insert <索引> <类型>\n"); return 1; }
        int idx = atoi(argv[2]);
        fx_type_t t = fx_type_from_name(argv[3]);
        if (t == FX_NONE) { printf("未知类型: %s\n", argv[3]); return 1; }
        int new_idx = fx_chain_insert(idx, t);
        if (new_idx < 0) { printf("插入失败\n"); return 1; }
        printf("已插入 [%d] %s\n", new_idx, fx_type_name(t));
        return 0;
    }

    /* fx del <idx> : 删除指定效果器 */
    if (!strcmp(argv[1], "del") || !strcmp(argv[1], "rm")) {
        if (argc < 3) { printf("用法: fx del <索引>\n"); return 1; }
        int idx = atoi(argv[2]);
        esp_err_t err = fx_chain_remove(idx);
        if (err != ESP_OK) { printf("删除失败: %s\n", esp_err_to_name(err)); return 1; }
        printf("已删除 [%d]\n", idx);
        return 0;
    }

    /* fx clear : 清空链 */
    if (!strcmp(argv[1], "clear")) {
        fx_chain_clear();
        printf("链已清空\n");
        return 0;
    }

    /* fx bypass <idx> | fx enable <idx> : 旁通/启用单个效果器
     * fx bypass on|off : 全局旁通整个链 */
    if (!strcmp(argv[1], "bypass") || !strcmp(argv[1], "enable")) {
        if (argc < 3) {
            if (!strcmp(argv[1], "bypass")) {
                printf("当前全局旁通: %s\n", fx_chain_get_bypass() ? "是" : "否");
                printf("用法: fx bypass on|off        全局旁通\n");
                printf("      fx bypass <索引>        旁通单个效果器\n");
            } else {
                printf("用法: fx enable <索引>\n");
            }
            return 0;
        }
        /* 判断是 on/off 还是索引 */
        if (!strcmp(argv[1], "bypass") &&
            (!strcmp(argv[2], "on") || !strcmp(argv[2], "off") ||
             !strcmp(argv[2], "1")   || !strcmp(argv[2], "0"))) {
            int on = (!strcmp(argv[2], "on") || !strcmp(argv[2], "1")) ? 1 : 0;
            fx_chain_set_bypass(on);
            printf("效果器链 %s\n", on ? "已全局旁通" : "已启用");
            return 0;
        }
        /* 否则按索引处理 */
        int idx = atoi(argv[2]);
        int en = !strcmp(argv[1], "enable") ? 1 : 0;
        esp_err_t err = fx_chain_set_enabled(idx, en);
        if (err != ESP_OK) { printf("操作失败: %s\n", esp_err_to_name(err)); return 1; }
        printf("[%d] %s\n", idx, en ? "已启用" : "已旁通");
        return 0;
    }

    /* fx set <idx> <参数名> <值> : 设置参数 */
    if (!strcmp(argv[1], "set")) {
        if (argc < 5) { printf("用法: fx set <索引> <参数名> <值>\n"); return 1; }
        int idx = atoi(argv[2]);
        float val = strtof(argv[4], NULL);
        esp_err_t err = fx_chain_set_param(idx, argv[3], val);
        if (err != ESP_OK) { printf("设置失败 (索引或参数名无效)\n"); return 1; }
        printf("[%d] %s = %.3f\n", idx, argv[3], val);
        return 0;
    }

    /* fx reset : 重置所有效果器内部状态 */
    if (!strcmp(argv[1], "reset")) {
        fx_chain_reset();
        printf("所有效果器状态已重置\n");
        return 0;
    }

    /* fx preset <名称> : 加载预设 */
    if (!strcmp(argv[1], "preset")) {
        if (argc < 3) {
            char buf[128];
            fx_chain_list_presets(buf, sizeof(buf));
            printf("用法: fx preset <名称>\n可用预设: %s\n", buf);
            return 1;
        }
        esp_err_t err = fx_chain_load_preset(argv[2]);
        if (err != ESP_OK) { printf("加载预设失败: %s\n", esp_err_to_name(err)); return 1; }
        fx_chain_print();
        return 0;
    }

    printf("用法:\n");
    printf("  fx list                   查看当前效果器链\n");
    printf("  fx types                  列出可用效果器类型\n");
    printf("  fx presets                列出可用预设\n");
    printf("  fx add <类型>             添加效果器到链尾\n");
    printf("  fx insert <索引> <类型>   在指定位置插入\n");
    printf("  fx del <索引>             删除指定效果器\n");
    printf("  fx clear                  清空链\n");
    printf("  fx set <索引> <参数> <值> 设置参数\n");
    printf("  fx bypass <索引>          旁通单个效果器\n");
    printf("  fx enable <索引>          启用单个效果器\n");
    printf("  fx bypass on|off          全局旁通整个链\n");
    printf("  fx reset                  重置所有效果器状态\n");
    printf("  fx preset <名称>          加载预设\n");
    printf("  fx start [-g in_db] [-G out_db]  启动独立实时处理\n");
    printf("  fx stop                   停止独立实时处理\n");
    printf("注: 串联模式下 (live -x) 无需 fx start, 效果器链由音色转换任务调用\n");
    return 1;
}

/* ---------- codec [wm8978|es8311] init|vol|gain|mute|info ---------- */
static int cmd_codec(int argc, char **argv)
{
    /* 子命令路由: codec wm8978 ... 或 codec es8311 ... */
    if (argc >= 2 && !strcmp(argv[1], "es8311")) {
        /* ── ES8311 子命令 ── */
        if (argc < 3 || !strcmp(argv[2], "info")) {
            printf("ES8311: ready=%d  running=%d\n",
                   es8311_is_ready() ? 1 : 0, 0);
            printf("I2C: SDA=%d SCL=%d addr=0x%02X\n",
                   ES8311_I2C_SDA, ES8311_I2C_SCL, ES8311_I2C_ADDR);
            printf("I2S: MCLK=%d BCLK=%d WS=%d DOUT=%d DIN=%d\n",
                   ES8311_I2S_MCLK, ES8311_I2S_BCLK, ES8311_I2S_WS,
                   ES8311_I2S_DOUT, ES8311_I2S_DIN);
            printf("用法:\n");
            printf("  codec es8311 init [sr] [mclk] [bclk] [ws] [dout] [din] [pa]  初始化\n");
            printf("  codec es8311 vol <0-100>        设置输出音量\n");
            printf("  codec es8311 gain <dB>          设置麦克风增益 (0~42)\n");
            printf("  codec es8311 deinit             释放 ES8311\n");
            printf("  codec es8311 info               查看状态\n");
            return 0;
        }

        if (!strcmp(argv[2], "init")) {
            int sr = ES8311_DEFAULT_SAMPLE_RATE;
            int mclk = ES8311_I2S_MCLK, bclk = ES8311_I2S_BCLK;
            int ws = ES8311_I2S_WS, dout = ES8311_I2S_DOUT, din = ES8311_I2S_DIN;
            int pa = ES8311_PA_PIN;
            if (argc >= 4)  sr   = atoi(argv[3]);
            if (argc >= 5)  mclk = atoi(argv[4]);
            if (argc >= 6)  bclk = atoi(argv[5]);
            if (argc >= 7)  ws   = atoi(argv[6]);
            if (argc >= 8)  dout = atoi(argv[7]);
            if (argc >= 9)  din  = atoi(argv[8]);
            if (argc >= 10) pa   = atoi(argv[9]);

            esp_err_t err = es8311_init(NULL, sr,
                                        (gpio_num_t)mclk, (gpio_num_t)bclk,
                                        (gpio_num_t)ws, (gpio_num_t)dout,
                                        (gpio_num_t)din,
                                        (gpio_num_t)pa, ES8311_I2C_ADDR);
            if (err == ESP_OK) {
                es8311_start();
                printf("ES8311 初始化成功 (sr=%d)\n", sr);
            } else {
                printf("ES8311 初始化失败: %s\n", esp_err_to_name(err));
            }
            return (err == ESP_OK) ? 0 : 1;
        }

        if (!strcmp(argv[2], "deinit")) {
            if (i2s_xform_running()) i2s_xform_stop();
            es8311_deinit();
            printf("ES8311 已释放\n");
            return 0;
        }

        if (!es8311_is_ready()) {
            printf("ES8311 未初始化, 请先 'codec es8311 init'\n");
            return 1;
        }

        if (!strcmp(argv[2], "vol") && argc >= 4) {
            int v = atoi(argv[3]);
            es8311_set_volume(v);
            printf("ES8311 输出音量: %d\n", v);
            return 0;
        }
        if (!strcmp(argv[2], "gain") && argc >= 4) {
            float g = strtof(argv[3], NULL);
            es8311_set_mic_gain(g);
            printf("ES8311 麦克风增益: %.1fdB\n", g);
            return 0;
        }

        printf("未知 ES8311 子命令: %s\n", argv[2]);
        return 1;
    }

    /* ── WM8978 子命令 (默认, 向后兼容) ── */
    if (argc < 2 || !strcmp(argv[1], "info") || !strcmp(argv[1], "wm8978")) {
        int wm_idx = (!strcmp(argv[1], "wm8978")) ? 2 : 1;
        if (argc < wm_idx + 1 || !strcmp(argv[wm_idx], "info")) {
            printf("WM8978: ready=%d\n", wm8978_is_ready() ? 1 : 0);
            printf("I2C: SDA=%d SCL=%d addr=0x%02X\n",
                   WM8978_I2C_SDA, WM8978_I2C_SCL, WM8978_I2C_ADDR);
            printf("用法:\n");
            printf("  codec [wm8978] init [sr]      初始化 WM8978\n");
            printf("  codec [wm8978] hp <0-63>      设置耳机音量\n");
            printf("  codec [wm8978] spk <0-63>     设置扬声器音量\n");
            printf("  codec [wm8978] gain <0-63>    设置输入增益\n");
            printf("  codec [wm8978] dac <0-255>    设置 DAC 音量\n");
            printf("  codec [wm8978] mute|unmute    静音/取消静音\n");
            printf("  codec [wm8978] input <1|2|3|4> 选择输入\n");
            printf("  codec [wm8978] regs           打印WM8978寄存器(调试)\n");
            printf("  codec es8311 ...              ES8311 编解码器控制\n");
            return 0;
        }
        /* 重定向到 WM8978 子命令 */
        int sub = wm_idx;
        if (!strcmp(argv[sub], "init")) {
            int sr = 0;
            if (argc >= sub + 2) sr = atoi(argv[sub + 1]);
            esp_err_t err = wm8978_init(sr);
            if (err == ESP_OK) {
                wm8978_start();
                printf("WM8978 初始化成功 (sr=%d)\n", sr > 0 ? sr : 48000);
            } else {
                printf("WM8978 初始化失败: %s\n", esp_err_to_name(err));
            }
            return (err == ESP_OK) ? 0 : 1;
        }

        if (!wm8978_is_ready()) {
            printf("WM8978 未初始化, 请先 'codec init'\n");
            return 1;
        }

        if (!strcmp(argv[sub], "hp") && argc >= sub + 2) {
            int v = atoi(argv[sub + 1]);
            wm8978_set_hp_volume(v);
            printf("耳机音量: %d OK\n", v);
            return 0;
        }
        if (!strcmp(argv[sub], "spk") && argc >= sub + 2) {
            int v = atoi(argv[sub + 1]);
            wm8978_set_spk_volume(v);
            printf("扬声器音量: %d OK\n", v);
            return 0;
        }
        if (!strcmp(argv[sub], "gain") && argc >= sub + 2) {
            int v = atoi(argv[sub + 1]);
            wm8978_set_input_gain(v);
            printf("输入增益: %d OK\n", v);
            return 0;
        }
        if (!strcmp(argv[sub], "dac") && argc >= sub + 2) {
            int v = atoi(argv[sub + 1]);
            wm8978_set_dac_volume(v);
            printf("DAC 音量: %d OK\n", v);
            return 0;
        }
        if (!strcmp(argv[sub], "mute")) {
            wm8978_mute(1);
            printf("已静音\n");
            return 0;
        }
        if (!strcmp(argv[sub], "unmute")) {
            wm8978_mute(0);
            printf("已取消静音\n");
            return 0;
        }
        if (!strcmp(argv[sub], "input") && argc >= sub + 2) {
            int ch = atoi(argv[sub + 1]);
            wm8978_input_t inp;
            switch (ch) {
            case 1: inp = WM8978_IN_LINPUT1; break;
            case 2: inp = WM8978_IN_LINPUT2; break;
            case 3: inp = WM8978_IN_LINPUT3; break;
            case 4: inp = WM8978_IN_DIFF;    break;
            default: printf("1=LINPUT1 2=LINPUT2 3=AUX 4=差分\n"); return 1;
            }
            wm8978_set_input(inp);
            printf("输入通道: %d OK\n", ch);
            return 0;
        }
        /* codec [wm8978] regs : 打印 WM8978 寄存器 (调试用) */
        if (!strcmp(argv[sub], "regs") || !strcmp(argv[sub], "dump")) {
            if (!wm8978_is_ready()) { printf("WM8978 未初始化\n"); return 1; }
            wm8978_dump_regs();
            return 0;
        }
        printf("未知 WM8978 子命令: %s\n", argv[sub]);
        return 1;
    }

    printf("未知 codec 类型: %s (支持 wm8978, es8311)\n", argv[1]);
    return 1;
}

/* ---------- debug <instrument> [-i in] : 打印首块首帧中间值供 PC 对拍 ---------- */
static int cmd_debug(int argc, char **argv)
{
    if (argc < 2) {
        printf("用法: debug <乐器> [-i in.wav]\n");
        printf("  默认输入 %s, MCU 默认推理参数: gain=0 clip=limit add_noise=0\n", DEF_IN_WAV);
        return 1;
    }
    if (!audio_xform_loaded()) { printf("模型未加载\n"); return 1; }
    const char *in = DEF_IN_WAV;
    for (int i = 2; i < argc; ) {
        if (i + 1 >= argc) { printf("参数 %s 缺少值\n", argv[i]); return 1; }
        if (!strcmp(argv[i], "-i")) in = argv[i + 1];
        else { printf("未知参数: %s\n", argv[i]); return 1; }
        i += 2;
    }
    esp_err_t err = audio_xform_debug(in, argv[1]);
    if (err == ESP_OK) printf("debug 日志已输出到串口 (logmel/mask/gain_lin[0:8])\n");
    else printf("debug 失败: %s\n", esp_err_to_name(err));
    return (err == ESP_OK) ? 0 : 1;
}

/* ---------- fxrender [-i in][-o out][-g in_db][-G out_db] : 效果器链 SD 卡渲染 + 实时性评估
 *
 * 模仿 infer 命令, 但用效果器链 (失真/EQ/延迟/混响等) 代替神经网络推理。
 * 读输入 wav -> 按帧通过效果器链处理 -> 写输出 wav, 同时统计每帧处理耗时,
 * 计算 RT 因子 (音频时长/处理耗时), 评估能否满足实时要求。
 *
 * 前提: 已用 'fx add' / 'fx preset' 配置好效果器链。
 * 帧长使用 FX_HOP_DEFAULT (240 采样 @48k = 5ms), 与 I2S 实时模式一致。
 */
static int cmd_fxrender(int argc, char **argv)
{
    const char *in  = DEF_IN_WAV;
    const char *out = NULL;
    float in_gain_db  = 0.0f;
    float out_gain_db = 0.0f;

    for (int i = 1; i < argc; ) {
        if (i + 1 >= argc) { printf("参数 %s 缺少值\n", argv[i]); return 1; }
        if      (!strcmp(argv[i], "-i")) in  = argv[i + 1];
        else if (!strcmp(argv[i], "-o")) out = argv[i + 1];
        else if (!strcmp(argv[i], "-g")) in_gain_db  = strtof(argv[i + 1], NULL);
        else if (!strcmp(argv[i], "-G")) out_gain_db = strtof(argv[i + 1], NULL);
        else { printf("未知参数: %s\n", argv[i]); return 1; }
        i += 2;
    }

    if (!out) out = BSP_SD_MOUNT "/out/fx.wav";

    /* 检查效果器链非空 */
    if (fx_chain_get_bypass()) {
        printf("效果器链全局旁通已开启, 输出=输入。用 'fx bypass off' 关闭。\n");
    }
    char st[128];
    fx_chain_status(st, sizeof(st));
    printf("链状态: %s\n", st);

    /* SD 占用标记 (与 infer 一致, 阻止 USB Host 切换) */
    usb_msc_set_busy(true);
    esp_err_t msc_rc = ESP_OK;
    if (usb_msc_is_ready()) {
        msc_rc = usb_msc_mount_app();
        if (msc_rc != ESP_OK)
            ESP_LOGW(TAG, "SD 切回 APP 模式失败: %s", esp_err_to_name(msc_rc));
    }

    /* 1. 读取输入 wav */
    float *x = NULL; int n = 0, sr = 0;
    esp_err_t err = wav_read_mono_f32(in, &x, &n, &sr);
    if (err != ESP_OK) {
        printf("读取 %s 失败: %s\n", in, esp_err_to_name(err));
        usb_msc_set_busy(false);
        return 1;
    }
    printf("输入: %s  %d采样 %dHz  %.2fs\n", in, n, sr, (double)n / sr);

    /* 2. 分配输出缓冲 (PSRAM), 复制输入 (in-place 效果器需要原数据) */
    float *y = (float *)heap_caps_malloc((size_t)n * sizeof(float), MALLOC_CAP_SPIRAM);
    if (!y) {
        printf("PSRAM 分配失败 (%d 样本)\n", n);
        free(x);
        usb_msc_set_busy(false);
        return 1;
    }
    float in_gain  = powf(10.0f, in_gain_db / 20.0f);
    float out_gain = powf(10.0f, out_gain_db / 20.0f);
    for (int i = 0; i < n; i++)
        y[i] = x[i] * in_gain;
    free(x);

    /* 3. 重置效果器链状态 (清延迟/混响缓冲) */
    fx_chain_reset();

    /* 4. 按帧处理, 统计每帧耗时 */
    int hop = FX_HOP_DEFAULT;
    int n_frames = (n + hop - 1) / hop;
    int64_t total_us = 0;
    int64_t min_us = INT64_MAX;
    int64_t max_us = 0;
    int over_budget = 0;   /* 超过帧预算 (hop/sr 秒) 的帧数 */

    int64_t t_start = esp_timer_get_time();
    for (int f = 0; f < n_frames; f++) {
        int off = f * hop;
        int len = (off + hop <= n) ? hop : (n - off);

        int64_t t0 = esp_timer_get_time();
        fx_chain_process(y + off, len);
        int64_t dt = esp_timer_get_time() - t0;

        total_us += dt;
        if (dt < min_us) min_us = dt;
        if (dt > max_us) max_us = dt;
        /* 帧预算: hop/sr 秒 = hop*1e6/sr 微秒; 超过即无法实时 */
        int64_t budget_us = (int64_t)hop * 1000000 / sr;
        if (dt > budget_us) over_budget++;
    }
    int64_t total_ms = (esp_timer_get_time() - t_start) / 1000;

    /* 5. 输出增益 */
    for (int i = 0; i < n; i++)
        y[i] *= out_gain;

    /* 6. 写出 wav */
    err = wav_write_mono_f32(out, y, n, sr);
    free(y);
    usb_msc_set_busy(false);

    if (err != ESP_OK) {
        printf("写出 %s 失败: %s\n", out, esp_err_to_name(err));
        return 1;
    }

    /* 7. 打印实时性评估 */
    double audio_dur_ms = 1000.0 * n / sr;
    double rt_factor = (total_ms > 0) ? audio_dur_ms / (double)total_ms : 0.0;
    double avg_us = (n_frames > 0) ? (double)total_us / n_frames : 0.0;
    int64_t budget_us = (int64_t)hop * 1000000 / sr;

    printf("输出: %s  %d采样 %dHz\n", out, n, sr);
    printf("────────── 实时性评估 ──────────\n");
    printf("帧长: %d 采样 (%.1fms @%dHz)\n", hop, (double)hop * 1000 / sr, sr);
    printf("总帧数: %d\n", n_frames);
    printf("音频时长: %.2fs\n", audio_dur_ms / 1000.0);
    printf("处理耗时: %.2fs (%lldms)\n", (double)total_ms / 1000.0, (long long)total_ms);
    printf("RT 因子: %.2fx %s\n", rt_factor,
           rt_factor >= 1.0 ? "(实时OK)" : "(不满足实时!)");
    printf("每帧耗时: avg=%.0fus  min=%lldus  max=%lldus  (预算 %lldus)\n",
           avg_us, (long long)min_us, (long long)max_us, (long long)budget_us);
    printf("超预算帧: %d / %d (%.1f%%)\n", over_budget, n_frames,
           n_frames > 0 ? 100.0 * over_budget / n_frames : 0.0);
    if (max_us > budget_us)
        printf("⚠ 最差帧 %.2fms 超预算 %.2fms → 实时模式会 underrun\n",
               (double)max_us / 1000.0, (double)budget_us / 1000.0);
    else
        printf("✓ 所有帧均在预算内, 可满足实时\n");
    return 0;
}

/* ---------- i2srec [-o out.wav] [-t secs] : I2S 输入测试 — 从 RX 录制 WAV 到 SD
 *
 * 调试用命令。启动 I2S 双工 → 从 RX 按实时速率录制 → 写出 WAV → 停止+释放。
 * 每次调用独立管理 I2S 生命周期 (init → start → 录制 → stop → deinit)。
 * 如果 I2S pipeline 正在运行则拒绝 (互斥)。
 */
static int cmd_i2srec(int argc, char **argv)
{
    const char *out = NULL;
    int duration_sec = 10;
    for (int i = 1; i < argc; ) {
        if (i + 1 >= argc) { printf("参数 %s 缺少值\n", argv[i]); return 1; }
        if      (!strcmp(argv[i], "-o")) out = argv[i + 1];
        else if (!strcmp(argv[i], "-t")) duration_sec = atoi(argv[i + 1]);
        else { printf("未知参数: %s\n", argv[i]); return 1; }
        i += 2;
    }
    if (!out) out = BSP_SD_MOUNT "/out/i2s_rec.wav";
    if (duration_sec < 1 || duration_sec > 60) {
        printf("录制时长需 1-60 秒\n"); return 1;
    }

    /* 互斥检查 */
    if (i2s_xform_running()) {
        printf("I2S 实时推理正在运行, 请先 'live stop'\n"); return 1;
    }
    if (fx_chain_running()) {
        printf("效果器链实时处理正在运行, 请先 'fx stop'\n"); return 1;
    }

    /* 释放残留 I2S 状态 (上次 test 留下的 stopped-but-ready 状态) */
    if (i2s_driver_is_ready()) {
        i2s_driver_stop();
        i2s_driver_deinit();
    }

    /* 初始化 I2S */
    esp_err_t err = i2s_driver_init(0);
    if (err != ESP_OK) {
        printf("I2S 初始化失败: %s\n", esp_err_to_name(err)); return 1;
    }

    /* 确保 WM8978 可用 (输入链路: LINPUT1→ADC→I2S DIN) */
    if (!wm8978_is_ready()) {
        err = wm8978_init(48000);
        if (err != ESP_OK) {
            printf("WM8978 初始化失败: %s\n", esp_err_to_name(err));
            i2s_driver_deinit();
            return 1;
        }
        wm8978_start();
        printf("(已自动初始化 WM8978)\n");
    } else {
        /* 确保 WM8978 处于启动状态 (ADC/DAC 上电, 取消静音) */
        wm8978_start();
    }

    /* 调试: 打印 WM8978 关键寄存器 */
    wm8978_dump_regs();

    /* 启动 I2S 双工 */
    err = i2s_driver_start();
    if (err != ESP_OK) {
        printf("I2S 启动失败: %s\n", esp_err_to_name(err));
        i2s_driver_deinit();
        return 1;
    }

    int sr = I2S_DEFAULT_SAMPLE_RATE;
    int total_samples = duration_sec * sr;
    int chunk = 256;  /* 匹配 DMA dma_frame_num=256 */

    /* 分配 PSRAM 缓冲 */
    float *buf = (float *)heap_caps_malloc((size_t)total_samples * sizeof(float),
                                            MALLOC_CAP_SPIRAM);
    if (!buf) {
        printf("PSRAM 分配失败 (%d 样本)\n", total_samples);
        i2s_driver_stop(); i2s_driver_deinit();
        return 1;
    }

    int16_t *rx_buf = (int16_t *)heap_caps_malloc((size_t)chunk * sizeof(int16_t),
                                                   MALLOC_CAP_INTERNAL | MALLOC_CAP_32BIT);
    if (!rx_buf) {
        printf("内部缓冲分配失败\n"); free(buf);
        i2s_driver_stop(); i2s_driver_deinit();
        return 1;
    }

    printf("I2S 录制 %ds @ %dHz → %s ...\n", duration_sec, sr, out);
    int collected = 0, underruns = 0;
    int64_t t0 = esp_timer_get_time();
    while (collected < total_samples) {
        int need = (total_samples - collected > chunk) ? chunk : (total_samples - collected);
        int got = i2s_driver_read(rx_buf, need, 500);
        if (got > 0) {
            for (int i = 0; i < got; i++)
                buf[collected + i] = (float)rx_buf[i] / 32768.0f;
            collected += got;
        } else {
            /* 读失败: 填充静音 */
            underruns++;
            for (int i = 0; i < need; i++)
                buf[collected + i] = 0.0f;
            collected += need;
        }
    }
    int64_t elapsed_ms = (esp_timer_get_time() - t0) / 1000;
    free(rx_buf);

    /* 停止 + 释放 I2S (为下次 test 留干净状态) */
    i2s_driver_stop();
    i2s_driver_deinit();

    /* 写出 WAV */
    err = wav_write_mono_f32(out, buf, total_samples, sr);
    free(buf);
    if (err != ESP_OK) {
        printf("写出 %s 失败: %s\n", out, esp_err_to_name(err));
        return 1;
    }

    printf("录制完成: %s  %d采样 %dHz  %.1fs\n", out, total_samples, sr,
           (double)total_samples / sr);
    printf("录制耗时: %lldms  underruns=%d\n", (long long)elapsed_ms, underruns);
    if (underruns > 0)
        printf("⚠ I2S RX 读取有 %d 次欠载 → 检查硬件连接/时钟\n", underruns);
    else
        printf("✓ I2S RX 工作正常 (无欠载)\n");
    printf("提示: 把 SD 卡插回 PC, 用 Audacity 打开 %s 查看波形\n", out);
    return 0;
}

/* ---------- i2stone [-f freq] [-t secs] [-a amplitude] : I2S 输出测试 — 播放正弦波
 *
 * 调试用命令。生成正弦波 → 启动 I2S → 按实时速率逐帧写入 TX → 停止+释放。
 * 每次调用独立管理 I2S 生命周期 (init → start → 播放 → stop → deinit)。
 * 如果 I2S pipeline 正在运行则拒绝 (互斥)。
 *
 * 关键设计: 写入 pacing 由 i2s_channel_write 超时自然提供 —
 * 写满 DMA 缓冲后, 下一次 write 阻塞直到 DMA 消耗完一帧, 速率由硬件 clocks 决定。
 */
static int cmd_i2stone(int argc, char **argv)
{
    float freq = 440.0f;       /* A4 */
    int duration_sec = 10;
    float amplitude = 0.5f;    /* -6dBFS */
    for (int i = 1; i < argc; ) {
        if (i + 1 >= argc) { printf("参数 %s 缺少值\n", argv[i]); return 1; }
        if      (!strcmp(argv[i], "-f")) freq  = strtof(argv[i + 1], NULL);
        else if (!strcmp(argv[i], "-t")) duration_sec = atoi(argv[i + 1]);
        else if (!strcmp(argv[i], "-a")) amplitude = strtof(argv[i + 1], NULL);
        else { printf("未知参数: %s\n", argv[i]); return 1; }
        i += 2;
    }
    if (duration_sec < 1 || duration_sec > 60) {
        printf("输出时长需 1-60 秒\n"); return 1;
    }
    if (freq < 20.0f || freq > 20000.0f) {
        printf("频率需 20-20000 Hz\n"); return 1;
    }
    if (amplitude <= 0.0f || amplitude > 1.0f) {
        printf("幅度需 0-1\n"); return 1;
    }

    /* 互斥检查 */
    if (i2s_xform_running()) {
        printf("I2S 实时推理正在运行, 请先 'live stop'\n"); return 1;
    }
    if (fx_chain_running()) {
        printf("效果器链实时处理正在运行, 请先 'fx stop'\n"); return 1;
    }

    /* 释放残留 I2S 状态 */
    if (i2s_driver_is_ready()) {
        i2s_driver_stop();
        i2s_driver_deinit();
    }

    /* 初始化 I2S */
    esp_err_t err = i2s_driver_init(0);
    if (err != ESP_OK) {
        printf("I2S 初始化失败: %s\n", esp_err_to_name(err)); return 1;
    }

    /* 确保 WM8978 可用 + 取消静音 */
    if (!wm8978_is_ready()) {
        err = wm8978_init(48000);
        if (err != ESP_OK) {
            printf("WM8978 初始化失败: %s\n", esp_err_to_name(err));
            i2s_driver_deinit();
            return 1;
        }
        wm8978_start();
        wm8978_set_hp_volume(40);
        printf("(已自动初始化 WM8978, hp=40)\n");
    } else {
        wm8978_mute(0);
    }

    /* 启动 I2S 双工 */
    err = i2s_driver_start();
    if (err != ESP_OK) {
        printf("I2S 启动失败: %s\n", esp_err_to_name(err));
        i2s_driver_deinit();
        return 1;
    }

    int sr = I2S_DEFAULT_SAMPLE_RATE;
    int total_samples = duration_sec * sr;
    int chunk = 256;  /* 匹配 DMA dma_frame_num=256 */

    /* 生成正弦波 (float 便于计算) */
    float *sine = (float *)heap_caps_malloc((size_t)total_samples * sizeof(float),
                                             MALLOC_CAP_SPIRAM);
    if (!sine) {
        printf("PSRAM 分配失败 (%d 样本)\n", total_samples);
        i2s_driver_stop(); i2s_driver_deinit();
        return 1;
    }

    printf("生成 %.1fHz 正弦波 (%d秒 @%dHz, 幅度=%.2f) ...\n", freq, duration_sec, sr, amplitude);
    for (int i = 0; i < total_samples; i++) {
        float t = (float)i / (float)sr;
        float val = amplitude * sinf(2.0f * (float)M_PI * freq * t);
        /* 淡入淡出 50ms */
        float env = 1.0f;
        int fade = sr / 20;
        if (i < fade) env = (float)i / (float)fade;
        else if (i >= total_samples - fade) env = (float)(total_samples - 1 - i) / (float)fade;
        sine[i] = val * env;
    }

    /* 分配 int16 发送缓冲 (复用, 一次一帧) */
    int16_t *tx_buf = (int16_t *)heap_caps_malloc((size_t)chunk * sizeof(int16_t),
                                                   MALLOC_CAP_INTERNAL | MALLOC_CAP_32BIT);
    if (!tx_buf) {
        printf("内部缓冲分配失败\n"); free(sine);
        i2s_driver_stop(); i2s_driver_deinit();
        return 1;
    }

    printf("I2S TX: %.1fHz / %.2famp / %ds ... 请确认耳机/扬声器有输出\n",
           freq, amplitude, duration_sec);

    int sent = 0, underruns = 0;
    int64_t t0 = esp_timer_get_time();
    while (sent < total_samples) {
        int need = (total_samples - sent > chunk) ? chunk : (total_samples - sent);

        /* float → int16 */
        for (int i = 0; i < need; i++) {
            float v = sine[sent + i];
            if (v > 1.0f) v = 1.0f; else if (v < -1.0f) v = -1.0f;
            tx_buf[i] = (int16_t)(v * 32767.0f);
        }

        int done = i2s_driver_write(tx_buf, need, 2000);
        if (done == need) {
            sent += done;
        } else if (done > 0) {
            /* 部分写入 (罕见) */
            sent += done;
            underruns++;
        } else {
            /* 完全失败 */
            underruns++;
            vTaskDelay(pdMS_TO_TICKS(5));  /* 等一下再重试 */
        }
    }
    int64_t elapsed_ms = (esp_timer_get_time() - t0) / 1000;
    free(tx_buf);
    free(sine);

    /* 等待 DMA 排空 + 停止 */
    vTaskDelay(pdMS_TO_TICKS(50));
    i2s_driver_stop();
    i2s_driver_deinit();
    wm8978_mute(1);

    printf("播放完成: %.1fHz x %ds\n", freq, duration_sec);
    printf("播放耗时: %lldms  underruns=%d\n", (long long)elapsed_ms, underruns);
    if (underruns > 0)
        printf("⚠ I2S TX 写入有 %d 次欠载\n", underruns);
    else
        printf("✓ I2S TX 工作正常 (无欠载)\n");
    return 0;
}

/* ---------- i2sloop [-t secs] [-v vol] : I2S 环回测试 (RX->TX 实时直通)
 *
 * 验证完整的音频通路: 模拟输入 → ADC → I2S RX → I2S TX → DAC → 模拟输出。
 * 自动检测已初始化的编解码器 (ES8311 或 WM8978)。
 * 不经过神经网络, 纯粹验证硬件链路。
 *
 * 用法:
 *   i2sloop [-t 10]  环回 10 秒 (默认)
 *   i2sloop -t 30    环回 30 秒
 *   i2sloop -v 60    输出音量 60
 *
 * 互斥检查: 如果 I2S pipeline (音色转换/效果器) 正在运行则拒绝。
 */
static int cmd_i2sloop(int argc, char **argv)
{
    int duration_sec = 10;
    int volume = -1;  /* -1 = 不修改音量 */
    for (int i = 1; i < argc; ) {
        if (i + 1 >= argc) { printf("参数 %s 缺少值\n", argv[i]); return 1; }
        if      (!strcmp(argv[i], "-t")) duration_sec = atoi(argv[i + 1]);
        else if (!strcmp(argv[i], "-v")) volume = atoi(argv[i + 1]);
        else { printf("未知参数: %s\n", argv[i]); return 1; }
        i += 2;
    }
    if (duration_sec < 1 || duration_sec > 300) {
        printf("时长需 1-300 秒\n"); return 1;
    }

    /* 互斥检查 */
    if (i2s_xform_running()) {
        printf("I2S 实时推理正在运行, 请先 'live stop'\n"); return 1;
    }
    if (fx_chain_running()) {
        printf("效果器链正在运行, 请先 'fx stop'\n"); return 1;
    }

    /* ★ 释放残留 I2S 状态 */
    if (i2s_driver_is_ready()) {
        i2s_driver_stop();
        i2s_driver_deinit();
    }

    /* ★ 初始化 I2S */
    esp_err_t i2s_err = i2s_driver_init(0);
    if (i2s_err != ESP_OK) {
        printf("I2S 初始化失败: %s\n", esp_err_to_name(i2s_err));
        return 1;
    }

    /* 检测可用编解码器: ES8311 优先 */
    int use_es8311 = es8311_is_ready();
    int codec_ready = use_es8311 || wm8978_is_ready();

    if (!codec_ready) {
        /* 尝试自动初始化 ES8311 */
        printf("未检测到编解码器, 尝试初始化 ES8311 (GPIO %d-%d)...\n",
               ES8311_I2S_MCLK, ES8311_I2S_DIN);
        esp_err_t err = es8311_init(NULL, ES8311_DEFAULT_SAMPLE_RATE,
                                    ES8311_I2S_MCLK, ES8311_I2S_BCLK,
                                    ES8311_I2S_WS, ES8311_I2S_DOUT,
                                    ES8311_I2S_DIN,
                                    ES8311_PA_PIN, ES8311_I2C_ADDR);
        if (err == ESP_OK) {
            use_es8311 = 1;
            codec_ready = 1;
        } else {
            printf("ES8311 初始化失败: %s\n", esp_err_to_name(err));
            i2s_driver_deinit();
            return 1;
        }
    }

    if (use_es8311) {
        es8311_start();
        if (volume >= 0) es8311_set_volume(volume);
    } else {
        wm8978_start();
        wm8978_mute(0);
        if (volume >= 0) wm8978_set_hp_volume(volume);
        /* 诊断: 打印 WM8978 关键寄存器 */
        printf("── WM8978 寄存器诊断 ──\n");
        wm8978_dump_regs();
    }

    /* ★ 启动 I2S 双工 */
    i2s_err = i2s_driver_start();
    if (i2s_err != ESP_OK) {
        printf("I2S 启动失败: %s\n", esp_err_to_name(i2s_err));
        if (use_es8311) es8311_stop();
        else wm8978_stop();
        i2s_driver_deinit();
        return 1;
    }

    int sr = use_es8311 ? ES8311_DEFAULT_SAMPLE_RATE : I2S_DEFAULT_SAMPLE_RATE;
    /* chunk 必须对齐 DMA dma_frame_num (240), 否则每次读取需等待 2 个描述符 */
    int chunk = 240;
    int total_samples = duration_sec * sr;

    /* 编解码器上电后需要时间稳定 (ADC/DAC 模拟电路启动 ~50ms) */
    vTaskDelay(pdMS_TO_TICKS(80));

    /* 分配内部缓冲 */
    int16_t *buf = (int16_t *)heap_caps_malloc((size_t)chunk * sizeof(int16_t),
                                                MALLOC_CAP_INTERNAL | MALLOC_CAP_32BIT);
    if (!buf) {
        printf("缓冲分配失败\n");
        if (use_es8311) es8311_stop();
        else wm8978_stop();
        return 1;
    }

    printf("I2S 环回测试 %ds @%dHz (%s) ... 对着麦克风说话/弹吉他, 听耳机输出\n",
           duration_sec, sr, use_es8311 ? "ES8311" : "WM8978");
    printf("(按任意键提前终止)\n");

    /* ★ 关键: fgetc(stdin) 在 ESP32 上默认阻塞, 会卡死整个环回循环。
     *     用 fcntl 设为非阻塞, EOF 表示无输入, 非EOF=有按键 → 中断。 */
    int stdin_flags_saved = fcntl(STDIN_FILENO, F_GETFL, 0);
    if (stdin_flags_saved >= 0) {
        fcntl(STDIN_FILENO, F_SETFL, stdin_flags_saved | O_NONBLOCK);
    }

    int processed = 0;
    int und_rx = 0, und_tx = 0;
    int16_t peak = 0;  /* 记录最大采样绝对值, 诊断 ADC 输入 */
    int64_t t0 = esp_timer_get_time();
    while (processed < total_samples) {
        /* 非阻塞检查键盘输入 */
        int ch = fgetc(stdin);
        if (ch != EOF) { printf("\n用户中断\n"); break; }

        int need = (total_samples - processed > chunk) ? chunk : (total_samples - processed);
        int got;
        if (use_es8311) {
            got = es8311_read(buf, need, 100);
        } else {
            got = i2s_driver_read(buf, need, 100);
        }
        if (got <= 0) {
            und_rx++;
            memset(buf, 0, (size_t)need * sizeof(int16_t));
            got = need;
        } else {
            /* 检测输入信号峰值 */
            for (int i = 0; i < got; i++) {
                int16_t abs_val = buf[i] >= 0 ? buf[i] : (int16_t)(-buf[i]);
                if (abs_val > peak) peak = abs_val;
            }
        }

        int written;
        if (use_es8311) {
            written = es8311_write(buf, got, 10);
        } else {
            written = i2s_driver_write(buf, got, 10);
        }
        if (written < got) und_tx++;
        processed += got;

        /* ★ 实时速率对齐: i2sloop 的 read/write 比 I2S 时钟快,
         *   导致 DMA 缓冲被耗尽后连续 timeout。
         *   按实际已处理采样数反算"应该经过的时间", 若 CPU 跑得太快则
         *   忙等补齐, 保证读节奏不超 I2S 时钟。 */
        int64_t target_us = (int64_t)processed * 1000000LL / sr;
        int64_t actual_us = esp_timer_get_time() - t0;
        if (actual_us < target_us)
            esp_rom_delay_us((uint32_t)(target_us - actual_us));
    }
    int64_t elapsed_ms = (esp_timer_get_time() - t0) / 1000;
    free(buf);

    /* 恢复 stdin 阻塞模式 (否则后续 console 无法输入) */
    if (stdin_flags_saved >= 0) {
        fcntl(STDIN_FILENO, F_SETFL, stdin_flags_saved);
    }

    /* ★★ 关键顺序! 先停 I2S (释放 DMA/ISR), 再停编解码器 (I2C 操作)。
     *     若在 I2S 运行期间操作 I2C, I2C 驱动被 DMA ISR 饿死, 导致
     *     ESP_ERR_INVALID_STATE, WM8978 未正确关闭 → 下次运行异常。 */
    i2s_driver_stop();
    i2s_driver_deinit();

    if (use_es8311) es8311_stop();
    else { wm8978_mute(1); wm8978_stop(); }

    printf("环回测试完成: %d采样 %.1fs\n", processed, (double)processed / sr);
    printf("耗时: %lldms  RX欠载=%d  TX溢出=%d\n",
           (long long)elapsed_ms, und_rx, und_tx);
    printf("输入峰值: %d (%.1fdBFS)\n", (int)peak,
           20.0 * log10((double)peak / 32768.0 + 1e-10));
    if (und_rx == 0 && und_tx == 0)
        printf("✓ I2S 环回正常 (无丢帧)\n");
    else
        printf("⚠ I2S 环回有丢帧, 检查硬件连接/时钟\n");
    if (peak < 100 && und_rx == 0)
        printf("⚠ 输入信号峰值极低 (%d), ADC 通路可能有问题\n", (int)peak);
    return 0;
}

/* ---------- i2stest [-f freq] [-t secs] [-a amp] [-v vol] : I2S 综合输出测试
 *
 * 生成正弦波 → 通过当前编解码器输出。
 * 自动检测已初始化的编解码器 (ES8311 或 WM8978), 无需手动指定。
 * 会调用 init→start→播放→stop, 适合快速验证 DAC 通路。
 */
static int cmd_i2stest(int argc, char **argv)
{
    float freq = 440.0f;
    int duration_sec = 3;
    float amplitude = 0.3f;
    int volume = 60;
    for (int i = 1; i < argc; ) {
        if (i + 1 >= argc) { printf("参数 %s 缺少值\n", argv[i]); return 1; }
        if      (!strcmp(argv[i], "-f")) freq  = strtof(argv[i + 1], NULL);
        else if (!strcmp(argv[i], "-t")) duration_sec = atoi(argv[i + 1]);
        else if (!strcmp(argv[i], "-a")) amplitude = strtof(argv[i + 1], NULL);
        else if (!strcmp(argv[i], "-v")) volume = atoi(argv[i + 1]);
        else { printf("未知参数: %s\n", argv[i]); return 1; }
        i += 2;
    }
    if (duration_sec < 1 || duration_sec > 60) {
        printf("时长需 1-60 秒\n"); return 1;
    }

    /* 互斥检查 */
    if (i2s_xform_running()) {
        printf("I2S 实时推理正在运行, 请先 'live stop'\n"); return 1;
    }
    if (fx_chain_running()) {
        printf("效果器链正在运行, 请先 'fx stop'\n"); return 1;
    }

    /* 检测可用编解码器 */
    int use_es8311 = es8311_is_ready();
    if (!use_es8311 && !wm8978_is_ready()) {
        /* 尝试自动初始化 ES8311 */
        printf("自动初始化 ES8311...\n");
        esp_err_t err = es8311_init(NULL, ES8311_DEFAULT_SAMPLE_RATE,
                                    ES8311_I2S_MCLK, ES8311_I2S_BCLK,
                                    ES8311_I2S_WS, ES8311_I2S_DOUT,
                                    ES8311_I2S_DIN,
                                    ES8311_PA_PIN, ES8311_I2C_ADDR);
        if (err == ESP_OK) {
            use_es8311 = 1;
        } else {
            printf("ES8311 初始化失败: %s\n", esp_err_to_name(err));
            return 1;
        }
    }

    if (use_es8311) {
        es8311_start();
        es8311_set_volume(volume);
    } else {
        wm8978_start();
        wm8978_set_hp_volume(volume);
    }

    int sr = use_es8311 ? ES8311_DEFAULT_SAMPLE_RATE : I2S_DEFAULT_SAMPLE_RATE;
    int total_samples = duration_sec * sr;
    int chunk = 256;

    float *sine = (float *)heap_caps_malloc((size_t)total_samples * sizeof(float),
                                             MALLOC_CAP_SPIRAM);
    if (!sine) {
        printf("PSRAM 分配失败\n");
        if (use_es8311) es8311_stop();
        else wm8978_stop();
        return 1;
    }
    for (int i = 0; i < total_samples; i++) {
        float t = (float)i / (float)sr;
        float val = amplitude * sinf(2.0f * (float)M_PI * freq * t);
        int fade = sr / 20;
        float env = 1.0f;
        if (i < fade) env = (float)i / (float)fade;
        else if (i >= total_samples - fade) env = (float)(total_samples - 1 - i) / (float)fade;
        sine[i] = val * env;
    }

    int16_t *tx_buf = (int16_t *)heap_caps_malloc((size_t)chunk * sizeof(int16_t),
                                                   MALLOC_CAP_INTERNAL | MALLOC_CAP_32BIT);
    if (!tx_buf) {
        printf("缓冲分配失败\n"); free(sine);
        if (use_es8311) es8311_stop();
        else wm8978_stop();
        return 1;
    }

    printf("I2S 测试音: %.1fHz / %.1f秒 / vol=%d (%s)\n",
           freq, (double)duration_sec, volume, use_es8311 ? "ES8311" : "WM8978");

    int sent = 0, underruns = 0;
    int64_t t0 = esp_timer_get_time();
    while (sent < total_samples) {
        int need = (total_samples - sent > chunk) ? chunk : (total_samples - sent);
        for (int i = 0; i < need; i++) {
            float v = sine[sent + i];
            if (v > 1.0f) v = 1.0f; else if (v < -1.0f) v = -1.0f;
            tx_buf[i] = (int16_t)(v * 32767.0f);
        }
        int done;
        if (use_es8311) {
            done = es8311_write(tx_buf, need, 2000);
        } else {
            done = i2s_driver_write(tx_buf, need, 2000);
        }
        if (done == need) {
            sent += done;
        } else if (done > 0) {
            sent += done; underruns++;
        } else {
            underruns++;
        }
    }
    int64_t elapsed_ms = (esp_timer_get_time() - t0) / 1000;
    free(tx_buf); free(sine);

    vTaskDelay(pdMS_TO_TICKS(50));
    if (use_es8311) es8311_stop();
    else { wm8978_mute(1); wm8978_stop(); }

    printf("测试完成: %.1fHz x %ds  耗时=%lldms  underruns=%d\n",
           freq, duration_sec, (long long)elapsed_ms, underruns);
    if (underruns == 0)
        printf("✓ 输出正常\n");
    else
        printf("⚠ 有 %d 次欠载\n", underruns);
    return 0;
}

static void register_cmds(void)
{
    const esp_console_cmd_t cmds[] = {
        { .command = "model",  .help = "打印已加载模型信息(采样率/参数量/乐器列表)", .func = &cmd_model },
        { .command = "infer",  .help = "SD卡推理: infer <乐器> [-i in][-o out][-p 变调][-g 增益dB][-c clip][-N 开噪声]", .func = &cmd_infer },
        { .command = "debug",  .help = "数值对拍: debug <乐器> [-i in.wav] 打印 logmel/mask/gain_lin", .func = &cmd_debug },
        { .command = "live",   .help = "I2S实时推理: live <乐器> [-p 变调][-g 增益dB][-c clip][-N] | live stop | live inst <名称>", .func = &cmd_live },
        { .command = "mode",   .help = "切换运行模式: mode sd | mode i2s | mode fx", .func = &cmd_mode },
        { .command = "fx",     .help = "效果器链控制: fx list|add|del|set|bypass|preset|start|stop|...", .func = &cmd_fx },
        { .command = "fxrender", .help = "效果器链SD卡渲染+实时性评估: fxrender [-i in][-o out][-g in_db][-G out_db]", .func = &cmd_fxrender },
        { .command = "i2s",    .help = "I2S 驱动管理: i2s init [sr] | i2s deinit | i2s info", .func = &cmd_i2s },
        { .command = "codec",  .help = "CODEC 控制: codec [wm8978|es8311] init|vol|gain|mute|info", .func = &cmd_codec },
        { .command = "i2srec", .help = "I2S 输入测试: 录制 WAV 到 SD (调试用) — i2srec [-o out.wav] [-t 10]", .func = &cmd_i2srec },
        { .command = "i2stone",.help = "I2S 输出测试: 播放正弦波 (调试用) — i2stone [-f 440] [-t 10] [-a 0.5]", .func = &cmd_i2stone },
        { .command = "i2stest",.help = "I2S 综合输出测试 (ES8311/WM8978自动检测) — i2stest [-f 440] [-t 3] [-a 0.3] [-v 60]", .func = &cmd_i2stest },
        { .command = "i2sloop",.help = "I2S 环回测试 (RX->TX实时直通, ES8311/WM8978自动) — i2sloop [-t 10] [-v 60]", .func = &cmd_i2sloop },
        { .command = "status", .help = "查看推理 worker 状态与内存", .func = &cmd_status },
        { .command = "stats",  .help = "推理时间统计 (次数/avg/min/max/RT); stats reset 清零", .func = &cmd_stats },
        { .command = "ls",     .help = "列目录: ls [路径] (默认 /sdcard)", .func = &cmd_ls },
        { .command = "sdinfo", .help = "SD 卡挂载状态", .func = &cmd_sdinfo },
        { .command = "usb",    .help = "USB 复合设备控制: usb app|host|status", .func = &cmd_usb },
    };
    for (size_t i = 0; i < sizeof(cmds) / sizeof(cmds[0]); ++i)
        ESP_ERROR_CHECK(esp_console_cmd_register(&cmds[i]));
}

esp_err_t cli_start(void)
{
    esp_console_repl_t *repl = NULL;
    esp_console_repl_config_t repl_cfg = ESP_CONSOLE_REPL_CONFIG_DEFAULT();
    repl_cfg.prompt = "xform> ";
    repl_cfg.max_cmdline_length = 128;
    repl_cfg.max_history_len = 16;

    esp_console_dev_uart_config_t uart_cfg = ESP_CONSOLE_DEV_UART_CONFIG_DEFAULT();
    esp_err_t err = esp_console_new_repl_uart(&uart_cfg, &repl_cfg, &repl);
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "REPL 创建失败: %s", esp_err_to_name(err));
        return err;
    }

    esp_console_register_help_command();
    register_cmds();

    err = esp_console_start_repl(repl);
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "REPL 启动失败: %s", esp_err_to_name(err));
        return err;
    }
    ESP_LOGI(TAG, "主控制台命令行就绪 (输入 help 查看命令)");
    return ESP_OK;
}
