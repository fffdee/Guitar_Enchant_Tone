/* WM8978 立体声编解码器驱动 (I2C 控制 + I2S 音频)。
 *
 * WM8978 是 Wolfson(现 Cirrus Logic) 的低功耗高质量立体声 CODEC,
 * 集成麦克风前置放大器、耳机驱动器、扬声器驱动器和 5 段均衡器。
 * 通过 I2C 配置寄存器, 通过 I2S 传输音频数据。
 *
 * 引脚分配 (用户自定义, 通过宏修改):
 *   I2C:
 *     SDA, SCL   (I2C 数据/时钟, 见 WM8978_I2C_SDA/SCL)
 *   I2S (与 i2s_driver 共用):
 *     MCLK (主时钟, WM8978 需要外部 MCLK)
 *     BCLK (位时钟)
 *     WS   (字选择/帧同步)
 *     DOUT (MCU -> WM8978 DAC)
 *     DIN  (WM8978 ADC -> MCU)
 *
 * WM8978 I2C 地址: 0x1A (CSB=0) 或 0x1B (CSB=1)
 * 寄存器格式: 7 位地址 + 9 位数据, 每次 I2C 写 2 字节
 * 寄存器映射对齐 WM8978 V4.5 数据手册与 STM32F103 验证例程
 */
#ifndef WM8978_H
#define WM8978_H

#include "esp_err.h"
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ── 硬件引脚定义 (根据实际接线修改) ── */
#define WM8978_I2C_PORT     0        /* I2C 端口号 */
#define WM8978_I2C_SDA      12       /* SDA 引脚 */
#define WM8978_I2C_SCL      11       /* SCL 引脚 */
#define WM8978_I2C_FREQ     400000   /* 400kHz Fast-mode */
#define WM8978_I2C_ADDR     0x1A     /* 7-bit 地址 (CSB 接地) */

/* ── WM8978 寄存器地址定义 (7-bit, 对齐 WM8978 V4.5 数据手册) ── */
#define WM8978_REG_RESET            0x00  /* R0  软件复位 */
#define WM8978_REG_PWR1             0x01  /* R1  电源管理 1 */
#define WM8978_REG_PWR2             0x02  /* R2  电源管理 2 */
#define WM8978_REG_PWR3             0x03  /* R3  电源管理 3 */
#define WM8978_REG_AIF              0x04  /* R4  音频接口 */
#define WM8978_REG_COMP             0x05  /* R5  压扩控制 */
#define WM8978_REG_CLOCK            0x06  /* R6  时钟生成 */
#define WM8978_REG_ADD              0x07  /* R7  附加控制 */
#define WM8978_REG_GPIO             0x08  /* R8  GPIO 控制 */
#define WM8978_REG_JACK1            0x09  /* R9  插孔检测 1 */
#define WM8978_REG_JACK2            0x0A  /* R10 插孔检测 2 */
#define WM8978_REG_DAC              0x0B  /* R11 DAC 控制 */
#define WM8978_REG_DACVOL_L         0x0C  /* R12 DAC 左声道数字音量 */
#define WM8978_REG_DACVOL_R         0x0D  /* R13 DAC 右声道数字音量 */
#define WM8978_REG_ADC              0x0E  /* R14 ADC 控制 */
#define WM8978_REG_ADCVOL_L         0x0F  /* R15 ADC 左声道数字音量 */
#define WM8978_REG_ADCVOL_R         0x10  /* R16 ADC 右声道数字音量 */
#define WM8978_REG_EQ1              0x12  /* R18 EQ 低频架 */
#define WM8978_REG_EQ2              0x13  /* R19 EQ 频段 1 */
#define WM8978_REG_EQ3              0x14  /* R20 EQ 频段 2 */
#define WM8978_REG_EQ4              0x15  /* R21 EQ 频段 3 */
#define WM8978_REG_EQ5              0x16  /* R22 EQ 高频架 */
#define WM8978_REG_DACLIM1          0x18  /* R24 DAC 限幅器 1 */
#define WM8978_REG_DACLIM2          0x19  /* R25 DAC 限幅器 2 */
#define WM8978_REG_NOTCH1           0x1B  /* R27 陷波滤波器 1 */
#define WM8978_REG_NOTCH2           0x1C  /* R28 陷波滤波器 2 */
#define WM8978_REG_NOTCH3           0x1D  /* R29 陷波滤波器 3 */
#define WM8978_REG_NOTCH4           0x1E  /* R30 陷波滤波器 4 */
#define WM8978_REG_ALC1             0x20  /* R32 ALC 控制 1 */
#define WM8978_REG_ALC2             0x21  /* R33 ALC 控制 2 */
#define WM8978_REG_ALC3             0x22  /* R34 ALC 控制 3 */
#define WM8978_REG_NOISEGATE        0x23  /* R35 噪声门 */
#define WM8978_REG_PLLN             0x24  /* R36 PLL N */
#define WM8978_REG_PLLK1            0x25  /* R37 PLL K1 */
#define WM8978_REG_PLLK2            0x26  /* R38 PLL K2 */
#define WM8978_REG_PLLK3            0x27  /* R39 PLL K3 */
#define WM8978_REG_3D               0x29  /* R41 3D 增强 */
#define WM8978_REG_AUXR             0x2B  /* R43 AUXR 蜂鸣/ROUT2反相控制 */
#define WM8978_REG_INCTRL           0x2C  /* R44 输入控制 */
#define WM8978_REG_LINVOL           0x2D  /* R45 左输入 PGA 音量 */
#define WM8978_REG_RINVOL           0x2E  /* R46 右输入 PGA 音量 */
#define WM8978_REG_INBOOST_L        0x2F  /* R47 左输入 Boost/MIC 增益 */
#define WM8978_REG_INBOOST_R        0x30  /* R48 右输入 Boost/MIC 增益 */
#define WM8978_REG_OUTCTRL          0x31  /* R49 输出控制 (DAC交叉混音/Boost) */
#define WM8978_REG_LOUTMIX          0x32  /* R50 左输出混音器 */
#define WM8978_REG_ROUTMIX          0x33  /* R51 右输出混音器 */
#define WM8978_REG_LOUT1VOL         0x34  /* R52 耳机左音量 (LOUT1) */
#define WM8978_REG_ROUT1VOL         0x35  /* R53 耳机右音量 (ROUT1) */
#define WM8978_REG_LOUT2VOL         0x36  /* R54 扬声器左音量 (LOUT2) */
#define WM8978_REG_ROUT2VOL         0x37  /* R55 扬声器右音量 (ROUT2) */
#define WM8978_REG_OUT3MIX          0x38  /* R56 OUT3 混音器 */
#define WM8978_REG_OUT4MIX          0x39  /* R57 OUT4 混音器 */

/* ── 输入通道选择 ── */
typedef enum {
    WM8978_IN_LINPUT1 = 0,   /* LINPUT1 引脚 */
    WM8978_IN_LINPUT2 = 1,   /* LINPUT2 引脚 */
    WM8978_IN_LINPUT3 = 2,   /* LINPUT3 / AUX */
    WM8978_IN_DIFF    = 3,   /* 差分输入 (LINPUT1-LINPUT2) */
} wm8978_input_t;

/* ── 公共 API ── */

/* 初始化 WM8978: I2C 总线 + 寄存器配置 + 音频路径。
 * sample_rate: 采样率 (8000~48000), 传 0 则使用 48000。
 * 初始化后 CODEC 处于待机状态, 需调用 wm8978_start() 启动。 */
esp_err_t wm8978_init(int sample_rate);

/* 启动 WM8978 (上电 ADC/DAC, 启用输出) */
esp_err_t wm8978_start(void);

/* 停止 WM8978 (进入低功耗) */
esp_err_t wm8978_stop(void);

/* 反初始化, 释放 I2C 资源 */
esp_err_t wm8978_deinit(void);

/* 设置耳机输出音量 (0-63, 0=静音, 63=+6dB) */
esp_err_t wm8978_set_hp_volume(int vol);

/* 设置扬声器输出音量 (0-63, 0=静音, 63=+6dB) */
esp_err_t wm8978_set_spk_volume(int vol);

/* 设置 ADC 输入增益 (0-63, 0=-12dB, 63=+35.25dB) */
esp_err_t wm8978_set_input_gain(int gain);

/* 设置 DAC 输出音量 (0-255, 0=静音, 255=0dB) */
esp_err_t wm8978_set_dac_volume(int vol);

/* 静音控制 (1=静音, 0=取消静音) */
esp_err_t wm8978_mute(int mute);

/* 选择输入通道 (吉他接哪个输入引脚) */
esp_err_t wm8978_set_input(wm8978_input_t input);

/* 写寄存器 (底层接口, 供高级用户使用) */
esp_err_t wm8978_write_reg(uint8_t reg, uint16_t value);

/* 读寄存器 (WM8978 不支持 I2C 读, 此函数返回缓存的值) */
uint16_t wm8978_read_reg(uint8_t reg);

/* WM8978 是否已初始化 */
int wm8978_is_ready(void);

/* 打印所有非零寄存器的当前值 (用于调试 ADC/DAC 配置) */
void wm8978_dump_regs(void);

#ifdef __cplusplus
}
#endif
#endif
