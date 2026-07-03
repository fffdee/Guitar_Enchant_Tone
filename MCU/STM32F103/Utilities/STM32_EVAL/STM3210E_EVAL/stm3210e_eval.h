/**
  ******************************************************************************
  * @file    stm3210e_eval.c
  * @author  yidianusb
  * @version V1.0.0
  * @date    2026-02-xx
  * @brief   This file contains definitions for STM3210E_EVAL's Leds, push-buttons
  *          COM ports, sFLASH (on SPI) and Temperature Sensor LM75 (on I2C)
  *          hardware resources.  
  ******************************************************************************
  * @attention
  *
  * 实验平台: 亿点-起点STM32F103_USB开发板
  * 淘    宝: https://yidianusb.taobao.com
  *
  ******************************************************************************
  */

  
/* Define to prevent recursive inclusion -------------------------------------*/
#ifndef __STM3210E_EVAL_H
#define __STM3210E_EVAL_H

#ifdef __cplusplus
 extern "C" {
#endif 

/* Includes ------------------------------------------------------------------*/
#include "stm32f10x.h"
#include "stm32_eval_legacy.h"

/** @addtogroup Utilities
  * @{
  */ 

/** @addtogroup STM32_EVAL
  * @{
  */  
  
/** @addtogroup STM3210E_EVAL
  * @{
  */ 

/** @addtogroup STM3210E_EVAL_LOW_LEVEL
  * @{
  */ 
  
/** @defgroup STM3210E_EVAL_LOW_LEVEL_Exported_Types
  * @{
  */
typedef enum 
{
  LED1 = 0,
  LED2 = 1,
  LED3 = 2
} Led_TypeDef;

typedef enum 
{  
  BUTTON_WAKEUP = 0,
  BUTTON_TAMPER = 1
} Button_TypeDef;

typedef enum 
{  
  BUTTON_MODE_GPIO = 0,
  BUTTON_MODE_EXTI = 1
} ButtonMode_TypeDef;

typedef enum 
{
  COM1 = 0
} COM_TypeDef;
/**
  * @}
  */ 

/** @defgroup STM3210E_EVAL_LOW_LEVEL_Exported_Constants
  * @{
  */ 
/** 
  * @brief  Define for STM3210E_EVAL board  
  */ 


/** @addtogroup STM3210E_EVAL_LOW_LEVEL_LED
  * @{
  */
#define LEDn                             3

#define LED1_PIN                         GPIO_Pin_1
#define LED1_GPIO_PORT                   GPIOC
#define LED1_GPIO_CLK                    RCC_APB2Periph_GPIOC  
  
#define LED2_PIN                         GPIO_Pin_2
#define LED2_GPIO_PORT                   GPIOC
#define LED2_GPIO_CLK                    RCC_APB2Periph_GPIOC  

#define LED3_PIN                         GPIO_Pin_4  
#define LED3_GPIO_PORT                   GPIOB
#define LED3_GPIO_CLK                    RCC_APB2Periph_GPIOB  

/**
  * @}
  */
  
/** @addtogroup STM3210E_EVAL_LOW_LEVEL_BUTTON
  * @{
  */  
#define BUTTONn                          2

/**
 * @brief Wakeup push-button
 */
#define WAKEUP_BUTTON_PIN                      GPIO_Pin_0
#define WAKEUP_BUTTON_GPIO_PORT                GPIOA
#define WAKEUP_BUTTON_GPIO_CLK                 RCC_APB2Periph_GPIOA
#define WAKEUP_BUTTON_EXTI_LINE                EXTI_Line0
#define WAKEUP_BUTTON_EXTI_PORT_SOURCE         GPIO_PortSourceGPIOA
#define WAKEUP_BUTTON_EXTI_PIN_SOURCE          GPIO_PinSource0
#define WAKEUP_BUTTON_EXTI_IRQn                EXTI0_IRQn
#define WAKEUP_BUTTON_EXTI_IRQHandler          EXTI0_IRQHandler 

/**
 * @brief Tamper push-button
 */
#define TAMPER_BUTTON_PIN                      GPIO_Pin_13
#define TAMPER_BUTTON_GPIO_PORT                GPIOC
#define TAMPER_BUTTON_GPIO_CLK                 RCC_APB2Periph_GPIOC
#define TAMPER_BUTTON_EXTI_LINE                EXTI_Line13
#define TAMPER_BUTTON_EXTI_PORT_SOURCE         GPIO_PortSourceGPIOC
#define TAMPER_BUTTON_EXTI_PIN_SOURCE          GPIO_PinSource13
#define TAMPER_BUTTON_EXTI_IRQn                EXTI15_10_IRQn 
#define TAMPER_BUTTON_EXTI_IRQHandler          EXTI15_10_IRQHandler 
    
/**
  * @}
  */ 

/** @addtogroup STM3210E_EVAL_LOW_LEVEL_COM
  * @{
  */
#define COMn                             1

/**
 * @brief Definition for COM port1, connected to USART1
 */ 
#define EVAL_COM1                        USART1
#define EVAL_COM1_CLK                    RCC_APB2Periph_USART1
#define EVAL_COM1_TX_PIN                 GPIO_Pin_9
#define EVAL_COM1_TX_GPIO_PORT           GPIOA
#define EVAL_COM1_TX_GPIO_CLK            RCC_APB2Periph_GPIOA
#define EVAL_COM1_RX_PIN                 GPIO_Pin_10
#define EVAL_COM1_RX_GPIO_PORT           GPIOA
#define EVAL_COM1_RX_GPIO_CLK            RCC_APB2Periph_GPIOA
#define EVAL_COM1_IRQn                   USART1_IRQn

/**
  * @}
  */ 

/** @addtogroup STM3210E_EVAL_LOW_LEVEL_SD_FLASH
  * @{
  */
/**
  * @brief  SD FLASH SDIO Interface
  */ 

#define SD_DETECT_PIN                    GPIO_Pin_3                  /* PC.3 */
#define SD_DETECT_GPIO_PORT              GPIOC                       /* GPIOC */
#define SD_DETECT_GPIO_CLK               RCC_APB2Periph_GPIOC

#define SDIO_FIFO_ADDRESS                ((uint32_t)0x40018080)
/** 
  * @brief  SDIO Intialization Frequency (400KHz max)
  */
#define SDIO_INIT_CLK_DIV                ((uint8_t)0xB2)
/** 
  * @brief  SDIO Data Transfer Frequency (25MHz max) 
  */
#define SDIO_TRANSFER_CLK_DIV            ((uint8_t)0x02)

#define SD_SDIO_DMA                      DMA2
#define SD_SDIO_DMA_CLK                  RCC_AHBPeriph_DMA2
#define SD_SDIO_DMA_CHANNEL              DMA2_Channel4
#define SD_SDIO_DMA_FLAG_TC              DMA2_FLAG_TC4
#define SD_SDIO_DMA_FLAG_TE              DMA2_FLAG_TE4
#define SD_SDIO_DMA_FLAG_HT              DMA2_FLAG_HT4
#define SD_SDIO_DMA_FLAG_GL              DMA2_FLAG_GL4
#define SD_SDIO_DMA_IRQn                 DMA2_Channel4_5_IRQn
#define SD_SDIO_DMA_IRQHANDLER           DMA2_Channel4_5_IRQHandler

/**
  * @}
  */ 
  
/** @addtogroup STM3210E_EVAL_LOW_LEVEL_FLASH_SPI
  * @{
  */
/**
  * @brief  FLASH SPI Interface pins
  */  
#define sFLASH_SPI                       SPI2
#define sFLASH_SPI_CLK                   RCC_APB1Periph_SPI2

#define sFLASH_SPI_SCK_PIN               GPIO_Pin_13                 /* PB.13 */
#define sFLASH_SPI_SCK_GPIO_PORT         GPIOB                       /* GPIOB */
#define sFLASH_SPI_SCK_GPIO_CLK          RCC_APB2Periph_GPIOB

#define sFLASH_SPI_MISO_PIN              GPIO_Pin_14                 /* PB.14 */
#define sFLASH_SPI_MISO_GPIO_PORT        GPIOB                       /* GPIOB */
#define sFLASH_SPI_MISO_GPIO_CLK         RCC_APB2Periph_GPIOB

#define sFLASH_SPI_MOSI_PIN              GPIO_Pin_15                 /* PB.15 */
#define sFLASH_SPI_MOSI_GPIO_PORT        GPIOB                       /* GPIOB */
#define sFLASH_SPI_MOSI_GPIO_CLK         RCC_APB2Periph_GPIOB

#define sFLASH_CS_PIN                    GPIO_Pin_12                 /* PB.12 */
#define sFLASH_CS_GPIO_PORT              GPIOB                       /* GPIOB */
#define sFLASH_CS_GPIO_CLK               RCC_APB2Periph_GPIOB

/**
  * @}
  */

/** @addtogroup STM3210E_EVAL_LOW_LEVEL_EEPROM_I2C
  * @{
  */
/**
  * @brief  EEPROM I2C Interface pins
  */  
#define OLED_I2C                         I2C1
#define OLED_I2C_CLK                     RCC_APB1Periph_I2C1
#define OLED_I2C_SCL_PIN                 GPIO_Pin_6                  /* PB.06 */
#define OLED_I2C_SCL_GPIO_PORT           GPIOB                       /* GPIOB */
#define OLED_I2C_SCL_GPIO_CLK            RCC_APB2Periph_GPIOB
#define OLED_I2C_SDA_PIN                 GPIO_Pin_7                  /* PB.07 */
#define OLED_I2C_SDA_GPIO_PORT           GPIOB                       /* GPIOB */
#define OLED_I2C_SDA_GPIO_CLK            RCC_APB2Periph_GPIOB

/**
  * @}
  */
  
/**
  * @}
  */
  
/** @defgroup STM3210E_EVAL_LOW_LEVEL_Exported_Macros
  * @{
  */ 
/**
  * @}
  */ 

/** @defgroup STM3210E_EVAL_LOW_LEVEL_Exported_Functions
  * @{
  */ 
void STM_EVAL_LEDInit(Led_TypeDef Led);
void STM_EVAL_LEDOn(Led_TypeDef Led);
void STM_EVAL_LEDOff(Led_TypeDef Led);
void STM_EVAL_LEDToggle(Led_TypeDef Led);
void STM_EVAL_PBInit(Button_TypeDef Button, ButtonMode_TypeDef Button_Mode);
uint32_t STM_EVAL_PBGetState(Button_TypeDef Button);
void STM_EVAL_COMInit(COM_TypeDef COM, USART_InitTypeDef* USART_InitStruct);
void SD_LowLevel_DeInit(void);
void SD_LowLevel_Init(void); 
void SD_LowLevel_DMA_TxConfig(uint32_t *BufferSRC, uint32_t BufferSize);
void SD_LowLevel_DMA_RxConfig(uint32_t *BufferDST, uint32_t BufferSize);
void sFLASH_LowLevel_DeInit(void);
void sFLASH_LowLevel_Init(void); 
void OLED_LowLevel_DeInit(void);
void OLED_LowLevel_Init(void); 
/**
  * @}
  */
#ifdef __cplusplus
}
#endif
  
#endif /* __STM3210E_EVAL_H */
/**
  * @}
  */ 

/**
  * @}
  */ 

/**
  * @}
  */

/**
  * @}
  */
  
/************************ (C) COPYRIGHT STMicroelectronics *****END OF FILE****/
