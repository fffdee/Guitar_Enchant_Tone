#include "cli.h"
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <dirent.h>
#include <sys/stat.h>
#include "esp_console.h"
#include "esp_log.h"
#include "esp_heap_caps.h"
#include "bsp.h"
#include "audio_xform.h"
#include "infer_worker.h"
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
        printf("用法: infer <乐器> [-i in][-o out][-p 变调][-g 增益dB][-c limit|soft|hard][-N 开噪声注入]\n");
        printf("例:   infer bass -p -12 -g 9 -c soft\n");
        printf("      infer bass -N            (开噪声注入; 默认关闭以避免底噪)\n");
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
    printf("已入队: %s -> %s (pitch %+.1f, gain %+.1fdB, clip %d). 用 status 查看进度.\n",
           job.instrument, job.out_path, job.opt.pitch_semitones, job.opt.gain_db, (int)job.opt.clip_mode);
    return 0;
}

static void register_cmds(void)
{
    const esp_console_cmd_t cmds[] = {
        { .command = "model",  .help = "打印已加载模型信息(采样率/参数量/乐器列表)", .func = &cmd_model },
        { .command = "infer",  .help = "推理: infer <乐器> [-i in][-o out][-p 变调][-g 增益dB][-c clip][-N 开噪声]", .func = &cmd_infer },
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
