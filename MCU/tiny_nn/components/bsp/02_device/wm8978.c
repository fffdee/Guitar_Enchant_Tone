/* WM8978 立体声编解码器驱动实现。
 * 参考 STM32F103 验证过的寄存器映射 (WM8978 V4.5 数据手册)。
 *
 * 初始化流程:
 *   1. I2C 总线初始化
 *   2. 软件复位 WM8978
 *   3. 配置电源管理: BIASEN + VMIDSEL=11
 *   4. 配置音频接口: I2S 格式 (FMT=10), 16-bit (WL=00), 从模式
 *   5. 配置时钟: 被动时钟 (MS=0)
 *   6. 配置输出路径: DAC -> 混音器 -> 耳机
 *   7. 配置输入路径: LINPUT1/RINPUT1 -> PGA -> ADC (录制用)
 *   8. 设置默认音量
 *
 * WM8978 I2C 写时序: 2 字节
 *   Byte1: [(reg_addr & 0x7F) << 1 | (data >> 8) & 0x01]
 *   Byte2: [data & 0xFF]
 */
#include <string.h>
#include "wm8978.h"
#include "esp_log.h"
#include "esp_heap_caps.h"
#include "driver/i2c_master.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

static const char *TAG = "wm8978";

/* I2C 主控句柄 */
static i2c_master_bus_handle_t s_i2c_bus = NULL;
static i2c_master_dev_handle_t s_i2c_dev = NULL;

/* 寄存器缓存 (WM8978 不支持 I2C 读, 需本地缓存) */
static uint16_t s_regs[64] = {0};
static int s_ready = 0;
static int s_sample_rate = 48000;

/* ── 底层 I2C 读写 ──────────────────────────────────────────── */

esp_err_t wm8978_write_reg(uint8_t reg, uint16_t value)
{
    if (!s_i2c_dev) return ESP_ERR_INVALID_STATE;
    if (reg >= 64) return ESP_ERR_INVALID_ARG;

    /* WM8978 寄存器格式: 7-bit 地址 + 9-bit 数据 */
    uint8_t buf[2];
    buf[0] = (uint8_t)((reg << 1) | ((value >> 8) & 0x01));
    buf[1] = (uint8_t)(value & 0xFF);

    esp_err_t err = i2c_master_transmit(s_i2c_dev, buf, 2, -1);
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "写寄存器 R%d(0x%02X) 失败: %s", reg, reg, esp_err_to_name(err));
        return err;
    }
    s_regs[reg] = value;
    return ESP_OK;
}

uint16_t wm8978_read_reg(uint8_t reg)
{
    if (reg >= 64) return 0;
    return s_regs[reg];
}

/* 修改寄存器中的某些位 (读-改-写, 使用缓存值) */
static esp_err_t wm8978_update_bits(uint8_t reg, uint16_t mask, uint16_t val)
{
    uint16_t old = s_regs[reg];
    uint16_t new_val = (old & ~mask) | (val & mask);
    if (new_val == old) return ESP_OK;
    return wm8978_write_reg(reg, new_val);
}

/* ── I2C 总线初始化 ─────────────────────────────────────────── */

static esp_err_t wm8978_i2c_init(void)
{
    if (s_i2c_bus) return ESP_OK;

    i2c_master_bus_config_t bus_cfg = {
        .i2c_port = WM8978_I2C_PORT,
        .sda_io_num = WM8978_I2C_SDA,
        .scl_io_num = WM8978_I2C_SCL,
        .clk_source = I2C_CLK_SRC_DEFAULT,
        .glitch_ignore_cnt = 7,
        .flags.enable_internal_pullup = true,
    };
    esp_err_t err = i2c_new_master_bus(&bus_cfg, &s_i2c_bus);
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "I2C 总线初始化失败: %s", esp_err_to_name(err));
        return err;
    }

    i2c_device_config_t dev_cfg = {
        .dev_addr_length = I2C_ADDR_BIT_LEN_7,
        .device_address = WM8978_I2C_ADDR,
        .scl_speed_hz = WM8978_I2C_FREQ,
    };
    err = i2c_master_bus_add_device(s_i2c_bus, &dev_cfg, &s_i2c_dev);
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "I2C 设备添加失败: %s", esp_err_to_name(err));
        return err;
    }

    ESP_LOGI(TAG, "I2C 初始化: SDA=%d SCL=%d addr=0x%02X freq=%dHz",
             WM8978_I2C_SDA, WM8978_I2C_SCL, WM8978_I2C_ADDR, WM8978_I2C_FREQ);
    return ESP_OK;
}

/* ── 配置音频路径 (参考 STM32F103 验证过的 wm8978_CfgAudioPath) ── */

static esp_err_t wm8978_config_audio_path(int sample_rate)
{
    esp_err_t err;
    uint16_t usReg;

    /* ── R1 电源管理 1 ────────────────────────────────
     *   Bit3    BIASEN = 1      模拟放大器偏置使能 (必须为1)
     *   Bit1:0  VMIDSEL = 11    50kΩ 分压器 (必须非00)
     *   Bit4    MICBEN = 0      不需要 MIC 偏置
     */
    usReg = (1 << 3) | (3 << 0);   /* 0x00B */
    err = wm8978_write_reg(WM8978_REG_PWR1, usReg);
    if (err) return err;

    /* ── R2 电源管理 2 ────────────────────────────────
     *   Bit8    ROUT1EN = 1     右耳机输出使能
     *   Bit7    LOUT1EN = 1     左耳机输出使能
     *   Bit6    SLEEP = 0       正常工作模式
     *   Bit5    BOOSTENR = 1    右通道输入Boost使能 ★ 用到PGA时必须!
     *   Bit4    BOOSTENL = 1    左通道输入Boost使能 ★ 用到PGA时必须!
     *   Bit3:2  INPGA = 11      输入 PGA L+R 上电
     *   Bit1:0  ADCEN = 11      ADC L+R 上电
     *
     *   关键: BOOSTEN 控制输入自举电路, 数据手册注明 "用到PGA放大功能时必须使能"。
     *   缺少 BOOSTEN, 输入信号无法通过 LIP/RIP 到达 PGA!
     */
    usReg = (1 << 8) | (1 << 7)        /* ROUT1EN + LOUT1EN */
          | (1 << 5) | (1 << 4)        /* BOOSTENR + BOOSTENL ★ 关键 */
          | (1 << 3) | (1 << 2)        /* INPGAENR + INPGAENL */
          | (1 << 1) | (1 << 0);       /* ADCENR + ADCENL */
    /* ^ 0x1BF */
    err = wm8978_write_reg(WM8978_REG_PWR2, usReg);
    if (err) return err;

    /* ── R3 电源管理 3 ────────────────────────────────
     *   Bit6:5  LOUT2EN/ROUT2EN = 0  扬声器关闭
     *   Bit3    RMIXEN = 1      右输出混音器使能
     *   Bit2    LMIXEN = 1      左输出混音器使能 ← 之前这里错了! LMIXEN=0 导致左声道无声
     *   Bit1    DACENR = 1      右 DAC 使能
     *   Bit0    DACENL = 1      左 DAC 使能
     */
    usReg = (1 << 3) | (1 << 2) | (1 << 1) | (1 << 0);   /* 0x00F */
    err = wm8978_write_reg(WM8978_REG_PWR3, usReg);
    if (err) return err;

    /* ── R4 音频接口 ──────────────────────────────────
     *   Bit6:5  WL = 00        16-bit 字长
     *   Bit4:3  FMT = 10       I2S 飞利浦标准格式
     *   其余位默认 0 (从模式, 正常极性, 立体声)
     */
    usReg = (2 << 3) | (0 << 5);   /* 0x010 = FMT=10(I2S) + WL=00(16bit) */
    err = wm8978_write_reg(WM8978_REG_AIF, usReg);
    if (err) return err;

    /* ── R6 时钟生成 ──────────────────────────────────
     *   MS = 0  被动时钟模式, MCLK 由 MCU 提供
     */
    err = wm8978_write_reg(WM8978_REG_CLOCK, 0x000);
    if (err) return err;

    /* ── R7 附加控制 ──────────────────────────────────
     *   Bit5  TOCLK = 1      ★ 三态 CLKOUT, MCLK 引脚变为高阻输入
     *                           ESP32 的 MCLK 输出才能驱动该引脚。
     *   Bit3  SR    = 0      48kHz 采样率组
     *   Bit2:0 MCLKDIV = 000 MCLK 不分频
     *
     *   ★ 关键! 默认值 TOCLK=0 使 WM8978 驱动 MCLK 引脚为输出,
     *     与 ESP32 的 MCLK 输出冲突 → ADC/DAC 无时钟 → 完全无声! */
    err = wm8978_write_reg(WM8978_REG_ADD, (1 << 5));  /* TOCLK=1 */
    if (err) return err;

    /* ── R11 DAC 控制 ──────────────────────────────────
     *   SOFTMUTE = 0  取消 DAC 软静音
     *   DACOSR128 = 0 64x 过采样 (低功耗)
     */
    err = wm8978_write_reg(WM8978_REG_DAC, 0x000);
    if (err) return err;

    /* ── R12/R13 DAC 数字音量 ─────────────────────────
     *   0xFF = 0dB (DAC 数字域不衰减)
     *   右声道设 Bit8=1 触发左右同步更新
     */
    err = wm8978_write_reg(WM8978_REG_DACVOL_L, 0xFF);
    if (err) return err;
    err = wm8978_write_reg(WM8978_REG_DACVOL_R, 0xFF | (1 << 8));
    if (err) return err;

    /* ── R14 ADC 控制 ──────────────────────────────────
     *   HPFEN = 0    禁止高通滤波 (保留全部频段)
     *   ADCOSR = 1   128x 过采样 (最佳性能)
     */
    err = wm8978_write_reg(WM8978_REG_ADC, (1 << 3));
    if (err) return err;

    /* ── R15/R16 ADC 数字音量 ─────────────────────────
     *   0xFF = 0dB
     */
    err = wm8978_write_reg(WM8978_REG_ADCVOL_L, 0xFF);
    if (err) return err;
    err = wm8978_write_reg(WM8978_REG_ADCVOL_R, 0xFF | (1 << 8));
    if (err) return err;

    /* ── R32-R34 ALC 自动增益控制 ─────────────────────
     *   全部写 0 禁止 ALC
     */
    wm8978_write_reg(WM8978_REG_ALC1, 0x000);
    wm8978_write_reg(WM8978_REG_ALC2, 0x000);
    wm8978_write_reg(WM8978_REG_ALC3, 0x000);

    /* ── R35 噪声门 ────────────────────────────────────
     *   写默认值禁止噪声门
     */
    wm8978_write_reg(WM8978_REG_NOISEGATE, (3 << 1) | (7 << 0));

    /* ── R44 输入控制 (吉他单端模式) ──────────────────
     *   MBVSEL = 0     MIC 偏置 = 0.9*AVDD (但 MICBEN=0, 偏置未上电)
     *   R2_2INPPGA = 0  不使用 AUX 输入
     *   RIN2INPPGA = 0  右输入负端悬空 (内部接 VMID)
     *   RIP2INPPGA = 1  右输入正端接 RIP
     *   L2_2INPPGA = 0  不使用 AUX 输入
     *   LIN2INPPGA = 0  左输入负端悬空 (内部接 VMID) ← 单端模式
     *   LIP2INPPGA = 1  左输入正端接 LIP
     *
     *   吉他使用单端模式: PGA+ 接收信号, PGA- 由内部偏置到 VMID。
     *   单端模式输入阻抗更高, 更适合被动吉他拾音器的高输出阻抗。
     *   差分模式 (LIN→PGA-) 适合低阻抗话筒, 对被动吉他会导致信号衰减。
     */
    err = wm8978_write_reg(WM8978_REG_INCTRL,
                           (1 << 5) | (1 << 1));
    /* ^ 0x022: 只连接 LIP→PGA+ 和 RIP→PGA+ */
    if (err) return err;

    /* ── R45/R46 输入 PGA 音量 ────────────────────────
     *   Bit8  IPVU = 1     同步更新左右 (仅右声道设)
     *   Bit5:0 = 32 (0x20)  +12dB PGA 增益 (约 4×)
     *
     *   吉他信号: 被动拾音器 ~100-250mV, 主动 ~500mV-1V。
     *   +12dB (4×) 放大: 100mV→400mV, 250mV→1V, 500mV→2V(clip)。
     *   ADC 满量程 ~1Vrms, 这个增益适合大部分被动吉他。
     *   用户可运行时调: codec wm8978 gain <0-63>
     */
    err = wm8978_write_reg(WM8978_REG_LINVOL, 0x20);
    if (err) return err;
    err = wm8978_write_reg(WM8978_REG_RINVOL, 0x20 | (1 << 8));
    if (err) return err;

    /* ── R47/R48 输入 Boost/MIC 增益控制 ─────────────
     *   PGABOOST(bit8) = 0  禁止 MIC +20dB 前置放大 ← 关键!
     *     ★ 吉他不是话筒, 不需要 MIC 前置增益, 否则会让 ADC 严重过载削波 ★
     *   L2_2BOOSTVOL(bit6:4) = 3  AUX/Line 输入 boost +0dB
     *     (0=-12dB, 1=-9dB, 2=-6dB, 3=-3dB, 4=0dB, 5=+3dB, 6=+6dB, 7=+6dB)
     *
     *   注: L2_2BOOSTVOL 仅对 AUX 输入(L2/R2)有效, 对 MIC 输入(LIP/LIN)无效。
     *   设为 3 作为基线, 预留了 ±3dB 的调校空间。
     */
    usReg = (3 << 4);   /* 0x030: 无 PGABOOST, L2_2BOOSTVOL=3 */
    err = wm8978_write_reg(WM8978_REG_INBOOST_L, usReg);
    if (err) return err;
    err = wm8978_write_reg(WM8978_REG_INBOOST_R, usReg);
    if (err) return err;

    /* ── R49 输出控制 (关键!) ─────────────────────────
     *   Bit6  DACL2RMIX = 1  左 DAC → 右输出混音器
     *   Bit5  DACR2LMIX = 1  右 DAC → 左输出混音器
     *   (单声道模式下交叉混音确保两路都有声音)
     *   之前这个寄存器完全没写! DAC 数据无法到达输出混音器
     */
    usReg = (1 << 6) | (1 << 5);   /* 0x060 */
    err = wm8978_write_reg(WM8978_REG_OUTCTRL, usReg);
    if (err) return err;

    /* ── R50/R51 左右输出混音器 ───────────────────────
     *   Bit0  DACL2LMIX/DACR2RMIX = 1  DAC → 输出
     *   之前写到了错误的寄存器地址 (R47/R48 = MIC Boost)
     */
    usReg = (1 << 0);   /* DAC → output mixer */
    err = wm8978_write_reg(WM8978_REG_LOUTMIX, usReg);
    if (err) return err;
    err = wm8978_write_reg(WM8978_REG_ROUTMIX, usReg);
    if (err) return err;

    /* ── R52/R53 耳机音量 (LOUT1/ROUT1) ──────────────
     *   Bit6  MUTE = 0     取消静音
     *   Bit5:0 VOL = 40    约 -9dB 起始音量
     *   先写左声道缓存, 右声道加 Bit8(HPVU) 触发同步更新
     */
    err = wm8978_write_reg(WM8978_REG_LOUT1VOL, 40);
    if (err) return err;
    err = wm8978_write_reg(WM8978_REG_ROUT1VOL, 40 | (1 << 8));
    if (err) return err;

    /* ── R54/R55 扬声器音量 (LOUT2/ROUT2) ────────────
     *   默认静音 (Bit6 MUTE = 1)
     */
    err = wm8978_write_reg(WM8978_REG_LOUT2VOL, (1 << 6));
    if (err) return err;
    err = wm8978_write_reg(WM8978_REG_ROUT2VOL, (1 << 6) | (1 << 8));
    if (err) return err;

    return ESP_OK;
}

/* ── 公共 API 实现 ──────────────────────────────────────────── */

esp_err_t wm8978_init(int sample_rate)
{
    if (s_ready) return ESP_OK;

    s_sample_rate = (sample_rate > 0) ? sample_rate : 48000;

    /* Step 1: I2C 总线初始化 */
    esp_err_t err = wm8978_i2c_init();
    if (err) return err;

    /* Step 2: 软件复位 */
    vTaskDelay(pdMS_TO_TICKS(10));
    err = wm8978_write_reg(WM8978_REG_RESET, 0x000);
    if (err) {
        ESP_LOGE(TAG, "软件复位失败, 检查 I2C 接线和地址");
        return err;
    }
    vTaskDelay(pdMS_TO_TICKS(10));

    /* Step 3: 配置音频路径 */
    err = wm8978_config_audio_path(s_sample_rate);
    if (err) {
        ESP_LOGE(TAG, "音频路径配置失败");
        return err;
    }

    s_ready = 1;
    ESP_LOGI(TAG, "WM8978 初始化完成: %dHz, I2S 16-bit 从模式", s_sample_rate);
    ESP_LOGI(TAG, "  输入: LIP/RIP 单端 -> PGA(x32, +12dB) -> ADC -> I2S DIN");
    ESP_LOGI(TAG, "  输出: I2S DOUT -> DAC -> 混音器 -> LOUT1/ROUT1 (耳机)");
    return ESP_OK;
}

esp_err_t wm8978_start(void)
{
    if (!s_ready) return ESP_ERR_INVALID_STATE;
    esp_err_t err;

    /* R2: 确保输出+输入+BOOST 全部上电 (可能被 stop 关闭过) */
    err = wm8978_update_bits(WM8978_REG_PWR2,
                             (1 << 8) | (1 << 7) | (1 << 5) | (1 << 4)
                             | (3 << 2) | (3 << 0),
                             (1 << 8) | (1 << 7) | (1 << 5) | (1 << 4)
                             | (3 << 2) | (3 << 0));
    if (err) return err;

    /* 确保 R3 中 DACENL/DACENR/LMIXEN/RMIXEN 已上电 */
    err = wm8978_update_bits(WM8978_REG_PWR3,
                             (1 << 3) | (1 << 2) | (1 << 1) | (1 << 0),
                             (1 << 3) | (1 << 2) | (1 << 1) | (1 << 0));
    if (err) return err;

    /* R11: 取消 DAC 软静音 (SOFTMUTE=0, bit6) */
    err = wm8978_update_bits(WM8978_REG_DAC, (1 << 6), 0);
    if (err) return err;

    /* R52/R53: 取消耳机静音 (MUTE=0, bit6) */
    wm8978_update_bits(WM8978_REG_LOUT1VOL, (1 << 6), 0);
    wm8978_update_bits(WM8978_REG_ROUT1VOL, (1 << 6), 0);

    /* R54/R55: 取消扬声器静音 (MUTE=0, bit6) */
    wm8978_update_bits(WM8978_REG_LOUT2VOL, (1 << 6), 0);
    wm8978_update_bits(WM8978_REG_ROUT2VOL, (1 << 6), 0);

    ESP_LOGI(TAG, "WM8978 已启动 (DAC+ADC+混音器上电, 取消静音)");
    return ESP_OK;
}

esp_err_t wm8978_stop(void)
{
    if (!s_ready) return ESP_ERR_INVALID_STATE;

    /* R11: 软静音 DAC */
    wm8978_update_bits(WM8978_REG_DAC, (1 << 6), (1 << 6));

    /* 静音耳机和扬声器 (MUTE=1, bit6) */
    wm8978_update_bits(WM8978_REG_LOUT1VOL, (1 << 6), (1 << 6));
    wm8978_update_bits(WM8978_REG_ROUT1VOL, (1 << 6), (1 << 6));
    wm8978_update_bits(WM8978_REG_LOUT2VOL, (1 << 6), (1 << 6));
    wm8978_update_bits(WM8978_REG_ROUT2VOL, (1 << 6), (1 << 6));

    /* R3: 关闭 DAC + 混音器电源 */
    wm8978_update_bits(WM8978_REG_PWR3,
                       (1 << 3) | (1 << 2) | (1 << 1) | (1 << 0), 0);

    /* R2: 关闭输出+输入+BOOST 路径 (保留 BIASEN 运行) */
    wm8978_update_bits(WM8978_REG_PWR2,
                       (1 << 8) | (1 << 7) | (1 << 5) | (1 << 4)
                       | (3 << 2) | (3 << 0), 0);

    ESP_LOGI(TAG, "WM8978 已停止 (静音 + DAC/ADC/混音器下电)");
    return ESP_OK;
}

esp_err_t wm8978_deinit(void)
{
    if (!s_ready) return ESP_OK;
    wm8978_stop();

    /* 软件复位 */
    wm8978_write_reg(WM8978_REG_RESET, 0x000);
    vTaskDelay(pdMS_TO_TICKS(10));

    /* 释放 I2C */
    if (s_i2c_dev) {
        i2c_master_bus_rm_device(s_i2c_dev);
        s_i2c_dev = NULL;
    }
    if (s_i2c_bus) {
        i2c_del_master_bus(s_i2c_bus);
        s_i2c_bus = NULL;
    }

    s_ready = 0;
    memset(s_regs, 0, sizeof(s_regs));
    ESP_LOGI(TAG, "WM8978 已释放");
    return ESP_OK;
}

esp_err_t wm8978_set_hp_volume(int vol)
{
    if (!s_ready) return ESP_ERR_INVALID_STATE;
    if (vol < 0) vol = 0;
    if (vol > 63) vol = 63;

    /* R52/R53: 音量 Bits[5:0], Bit6=MUTE=0, Bit8=HPVU(同步触发)
     * 先写左声道 (不触发更新), 再写右声道 (Bit8=1 触发左右同步更新) */
    esp_err_t err = wm8978_write_reg(WM8978_REG_LOUT1VOL, (uint16_t)vol);
    if (err) return err;
    return wm8978_write_reg(WM8978_REG_ROUT1VOL, (uint16_t)vol | (1 << 8));
}

esp_err_t wm8978_set_spk_volume(int vol)
{
    if (!s_ready) return ESP_ERR_INVALID_STATE;
    if (vol < 0) vol = 0;
    if (vol > 63) vol = 63;

    /* R54/R55: 扬声器音量, 同耳机逻辑 */
    esp_err_t err = wm8978_write_reg(WM8978_REG_LOUT2VOL, (uint16_t)vol);
    if (err) return err;
    return wm8978_write_reg(WM8978_REG_ROUT2VOL, (uint16_t)vol | (1 << 8));
}

esp_err_t wm8978_set_input_gain(int gain)
{
    if (!s_ready) return ESP_ERR_INVALID_STATE;
    if (gain < 0) gain = 0;
    if (gain > 63) gain = 63;

    /* R45/R46: PGA 增益 Bits[5:0], Bit8=IPVU(同步触发) */
    esp_err_t err = wm8978_write_reg(WM8978_REG_LINVOL, (uint16_t)gain);
    if (err) return err;
    return wm8978_write_reg(WM8978_REG_RINVOL, (uint16_t)gain | (1 << 8));
}

esp_err_t wm8978_set_dac_volume(int vol)
{
    if (!s_ready) return ESP_ERR_INVALID_STATE;
    if (vol < 0) vol = 0;
    if (vol > 255) vol = 255;

    /* R12/R13: DAC 数字音量 Bits[7:0], Bit8=DACVU(同步触发) */
    esp_err_t err = wm8978_write_reg(WM8978_REG_DACVOL_L, (uint16_t)vol);
    if (err) return err;
    return wm8978_write_reg(WM8978_REG_DACVOL_R, (uint16_t)vol | (1 << 8));
}

esp_err_t wm8978_mute(int mute)
{
    if (!s_ready) return ESP_ERR_INVALID_STATE;

    /* R11 Bit6 SOFTMUTE: 1=软静音, 0=正常 */
    uint16_t val = mute ? (1 << 6) : 0;
    return wm8978_update_bits(WM8978_REG_DAC, (1 << 6), val);
}

esp_err_t wm8978_set_input(wm8978_input_t input)
{
    if (!s_ready) return ESP_ERR_INVALID_STATE;
    esp_err_t err;

    switch (input) {
    case WM8978_IN_LINPUT1:
        /* LIP2INPPGA + LIN2INPPGA, RIP2INPPGA + RIN2INPPGA */
        err = wm8978_update_bits(WM8978_REG_INCTRL,
                                 (1 << 5) | (1 << 4) | (1 << 1) | (1 << 0),
                                 (1 << 5) | (1 << 4) | (1 << 1) | (1 << 0));
        break;
    case WM8978_IN_LINPUT2:
        /* L2_2INPPGA + R2_2INPPGA */
        err = wm8978_update_bits(WM8978_REG_INCTRL,
                                 (1 << 6) | (1 << 2),
                                 (1 << 6) | (1 << 2));
        break;
    case WM8978_IN_LINPUT3:
        /* AUX 输入 */
        err = wm8978_update_bits(WM8978_REG_INCTRL,
                                 (1 << 5) | (1 << 4) | (1 << 1) | (1 << 0), 0);
        break;
    case WM8978_IN_DIFF:
        /* 差分输入 */
        err = wm8978_write_reg(WM8978_REG_INCTRL,
                               (1 << 5) | (1 << 4) | (1 << 1) | (1 << 0));
        break;
    default:
        return ESP_ERR_INVALID_ARG;
    }
    return err;
}

int wm8978_is_ready(void)
{
    return s_ready;
}

/* 寄存器名称表 (用于调试输出) */
static const char *reg_names[64] = {
    [0x00]="RESET",     [0x01]="PWR1",      [0x02]="PWR2",
    [0x03]="PWR3",      [0x04]="AIF",       [0x05]="COMP",
    [0x06]="CLOCK",     [0x07]="ADD",       [0x08]="GPIO",
    [0x09]="JACK1",     [0x0A]="JACK2",     [0x0B]="DAC",
    [0x0C]="DACVOL_L",  [0x0D]="DACVOL_R",  [0x0E]="ADC",
    [0x0F]="ADCVOL_L",  [0x10]="ADCVOL_R",  [0x12]="EQ1",
    [0x13]="EQ2",       [0x14]="EQ3",       [0x15]="EQ4",
    [0x16]="EQ5",       [0x18]="DACLIM1",   [0x19]="DACLIM2",
    [0x1B]="NOTCH1",    [0x1C]="NOTCH2",    [0x1D]="NOTCH3",
    [0x1E]="NOTCH4",    [0x20]="ALC1",      [0x21]="ALC2",
    [0x22]="ALC3",      [0x23]="NOISEGATE", [0x24]="PLLN",
    [0x25]="PLLK1",     [0x26]="PLLK2",     [0x27]="PLLK3",
    [0x29]="3D",        [0x2B]="AUXR",      [0x2C]="INCTRL",
    [0x2D]="LINVOL",    [0x2E]="RINVOL",    [0x2F]="INBOOST_L",
    [0x30]="INBOOST_R", [0x31]="OUTCTRL",   [0x32]="LOUTMIX",
    [0x33]="ROUTMIX",   [0x34]="LOUT1VOL",  [0x35]="ROUT1VOL",
    [0x36]="LOUT2VOL",  [0x37]="ROUT2VOL",  [0x38]="OUT3MIX",
    [0x39]="OUT4MIX",
};

void wm8978_dump_regs(void)
{
    if (!s_ready) {
        ESP_LOGI(TAG, "WM8978 未初始化, 无法 dump 寄存器");
        return;
    }
    ESP_LOGI(TAG, "==== WM8978 寄存器 dump ====");
    /* 只打印关键寄存器 (与录音/播放相关的) */
    static const uint8_t key_regs[] = {
        0x01,0x02,0x03,0x04,0x06,0x07,0x0B,0x0E,
        0x2C,0x2D,0x2E,0x2F,0x30,0x31,0x32,0x33,
        0x34,0x35
    };
    for (size_t i = 0; i < sizeof(key_regs); i++) {
        uint8_t r = key_regs[i];
        const char *name = (reg_names[r]) ? reg_names[r] : "???";
        if (s_regs[r] != 0 || r < 0x07) {  /* 电源和接口寄存器始终显示 */
            ESP_LOGI(TAG, "  R%-2d(0x%02X) %-10s = 0x%03X",
                     r, r, name, s_regs[r]);
        }
    }
    ESP_LOGI(TAG, "============================");
}
