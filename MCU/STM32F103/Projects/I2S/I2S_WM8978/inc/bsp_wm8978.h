/**
  ******************************************************************************
  * @file    bsp_wm8978.h
  * @author  yidianusb
  * @version V1.0.0
  * @date    2026-02-xx
  * @brief   Header for bsp_wm8978.c module
  ******************************************************************************
  * @attention
  *
  * 实验平台: 亿点-起点STM32F103_USB开发板
  * 淘    宝: https://yidianusb.taobao.com
  *
  ******************************************************************************
  */
  
/* Define to prevent recursive inclusion -------------------------------------*/
#ifndef __BSP_WM8978_H__
#define __BSP_WM8978_H__

/* Includes ------------------------------------------------------------------*/
#include "stm32f10x.h"

/*---------------------------------------------------------------------------------------------*/
/*------------------------   I2C控制WM8978配置部分  -------------------------------------------*/
/* WM8978 音频输入通道控制选项, 可以选择多路，比如 MIC_LEFT_ON | LINE_ON */
typedef enum
{
    IN_PATH_OFF     = 0x00, /* 无输入 */
    MIC_LEFT_ON     = 0x01, /* LIN,LIP脚，MIC左声道（接板载咪头）  */
    MIC_RIGHT_ON    = 0x02, /* RIN,RIP脚，MIC右声道（接板载咪头）  */
    LINE_ON         = 0x04, /* L2,R2 立体声输入(接板载耳机插座) */
    AUX_ON          = 0x08, /* AUXL,AUXR 立体声输入（开发板没用到） */
    DAC_ON          = 0x10, /* I2S数据DAC (CPU产生音频信号) */
    ADC_ON          = 0x20  /* 输入的音频馈入WM8978内部ADC （I2S录音) */
} IN_PATH_E;

/* WM8978 音频输出通道控制选项, 可以选择多路 */
typedef enum
{
    OUT_PATH_OFF    = 0x00, /* 无输出 */
    EAR_LEFT_ON     = 0x01, /* LOUT1 耳机左声道(接板载耳机插座) */
    EAR_RIGHT_ON    = 0x02, /* ROUT1 耳机右声道(接板载耳机插座) */
    SPK_ON          = 0x04, /* LOUT2和ROUT2反相输出单声道（开发板没用到）*/
    OUT3_4_ON       = 0x08, /* OUT3 和 OUT4 输出单声道音频（开发板没用到）*/
} OUT_PATH_E;

#define TOTAL_IN_BUF_SIZE              (192*8)
#define AUDIO_IN_PACKET                (192*8)



#define OUTPUT_DEVICE_SPEAKER          1
#define OUTPUT_DEVICE_HEADPHONE        2
#define OUTPUT_DEVICE_BOTH             3
#define OUTPUT_DEVICE_AUTO             4

/* 定义最大音量 */
#define VOLUME_MAX                     63     /* 最大音量 */
#define VOLUME_STEP                    1      /* 音量调节步长 */

/* 定义最大MIC增益 */
#define GAIN_MAX                       63     /* 最大增益 */
#define GAIN_STEP                      1      /* 增益步长 */

/* STM32 I2C 快速模式 */
#define WM8978_I2C_Speed               100000
/* WM8978 I2C从机地址 */
#define WM8978_SLAVE_ADDRESS           0x34

/*I2C接口*/
#define WM8978_I2C                     I2C2
#define WM8978_I2C_CLK                 RCC_APB1Periph_I2C2

#define WM8978_I2C_SCL_PIN             GPIO_Pin_10
#define WM8978_I2C_SCL_GPIO_PORT       GPIOB
#define WM8978_I2C_SCL_GPIO_CLK        RCC_APB2Periph_GPIOB

#define WM8978_I2C_SDA_PIN             GPIO_Pin_11
#define WM8978_I2C_SDA_GPIO_PORT       GPIOB
#define WM8978_I2C_SDA_GPIO_CLK        RCC_APB2Periph_GPIOB

/*---------------------------------------------------------------------------------------------*/
/*------------------------   I2S控制数据传输部分  ---------------------------------------------*/
/**
    * I2S总线传输音频数据口线
    * WM8978_LRC    -> PA15/I2S3_WS
    * WM8978_BCLK   -> PB3/I2S3_CK
    * WM8978_DACDAT -> PB5/I2S3_SD
    * WM8978_MCLK   -> PC7/I2S3_MCK
    */
#define WM8978_CLK                     RCC_APB1Periph_SPI3
#define WM8978_I2Sx_SPI                SPI3

#define WM8978_LRC_GPIO_CLK            RCC_APB2Periph_GPIOA
#define WM8978_LRC_PORT                GPIOA
#define WM8978_LRC_PIN                 GPIO_Pin_15

#define WM8978_BCLK_GPIO_CLK           RCC_APB2Periph_GPIOB
#define WM8978_BCLK_PORT               GPIOB
#define WM8978_BCLK_PIN                GPIO_Pin_3

#define WM8978_DACDAT_GPIO_CLK         RCC_APB2Periph_GPIOB
#define WM8978_DACDAT_PORT             GPIOB
#define WM8978_DACDAT_PIN              GPIO_Pin_5

#define WM8978_MCLK_GPIO_CLK           RCC_APB2Periph_GPIOC
#define WM8978_MCLK_PORT               GPIOC
#define WM8978_MCLK_PIN                GPIO_Pin_7

#define I2Sx_DMA                       DMA2
#define I2Sx_DMA_CLK                   RCC_AHBPeriph_DMA2
#define I2Sx_TX_DMA_CHANNEL            DMA2_Channel2
#define I2Sx_TX_DMA_IT_TCIF            DMA2_IT_TC2
#define I2Sx_TX_DMA_STREAM_IRQn        DMA2_Channel2_IRQn
#define I2Sx_TX_DMA_STREAM_IRQFUN      DMA2_Channel2_IRQHandler


/*等待超时时间*/
#define WM8978_I2C_FLAG_TIMEOUT        ((uint32_t)0x4000)
#define WM8978_I2C_LONG_TIMEOUT        ((uint32_t)(10 * WM8978_I2C_FLAG_TIMEOUT))

/* 供外部引用的函数声明 */
uint8_t wm8978_Init(void);
uint8_t wm8978_Reset(void);
void wm8978_CfgAudioIF(uint16_t _usStandard, uint8_t _ucWordLen);
void wm8978_OutMute(uint8_t _ucMute);
void wm8978_PowerDown(void);
void wm8978_CfgAudioPath(uint16_t _InPath, uint16_t _OutPath);
void wm8978_SetMicGain(uint8_t _ucGain);
void wm8978_SetLineGain(uint8_t _ucGain);
void wm8978_SetOUT2Volume(uint8_t _ucVolume);
void wm8978_SetOUT1Volume(uint8_t _ucVolume);
uint8_t wm8978_ReadOUT1Volume(void);
uint8_t wm8978_ReadOUT2Volume(void);
void wm8978_NotchFilter(uint16_t _NFA0, uint16_t _NFA1);

void I2S_GPIO_Config(void);
void I2S_Stop(void);
void I2Sx_Mode_Config(const uint16_t _usStandard, const uint16_t _usWordLen, const uint32_t _usAudioFreq);
void I2Sx_TX_DMA_Init(void);

#endif /* __BSP_WM8978_H__ */
