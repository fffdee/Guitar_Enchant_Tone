/**
  ******************************************************************************
  * @file    stm3210e_eval.c
  * @author  MCD Application Team
  * @version V5.1.0
  * @date    18-January-2013
  * @brief   This file provides
  *            - set of firmware functions to manage Leds, push-button and COM ports
  *            - low level initialization functions for SD card (on SDIO), SPI serial
  *              flash (sFLASH)
  *          available on STM3210E-EVAL evaluation board from STMicroelectronics.
  ******************************************************************************
  * @attention
  *
  * <h2><center>&copy; COPYRIGHT 2013 STMicroelectronics</center></h2>
  *
  * Licensed under MCD-ST Liberty SW License Agreement V2, (the "License");
  * You may not use this file except in compliance with the License.
  * You may obtain a copy of the License at:
  *
  *        http://www.st.com/software_license_agreement_liberty_v2
  *
  * Unless required by applicable law or agreed to in writing, software
  * distributed under the License is distributed on an "AS IS" BASIS,
  * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
  * See the License for the specific language governing permissions and
  * limitations under the License.
  *
  ******************************************************************************
  */

/* Includes ------------------------------------------------------------------*/
#include "stm3210e_eval.h"
#include "stm32f10x_spi.h"
#include "stm32f10x_i2c.h"
#include "stm32f10x_sdio.h"
#include "stm32f10x_dma.h"

/** @addtogroup Utilities
  * @{
  */

/** @addtogroup STM32_EVAL
  * @{
  */

/** @addtogroup STM3210E_EVAL
  * @{
  */

/** @defgroup STM3210E_EVAL_LOW_LEVEL
  * @brief This file provides firmware functions to manage Leds, push-buttons,
  *        COM ports, SD card on SDIO, serial flash (sFLASH), serial OLED (sEE)
  *        available on STM3210E-EVAL evaluation board from STMicroelectronics.
  * @{
  */

/** @defgroup STM3210E_EVAL_LOW_LEVEL_Private_TypesDefinitions
  * @{
  */
/**
  * @}
  */


/** @defgroup STM3210E_EVAL_LOW_LEVEL_Private_Defines
  * @{
  */
/**
  * @}
  */


/** @defgroup STM3210E_EVAL_LOW_LEVEL_Private_Macros
  * @{
  */
/**
  * @}
  */


/** @defgroup STM3210E_EVAL_LOW_LEVEL_Private_Variables
  * @{
  */
GPIO_TypeDef* GPIO_PORT[LEDn] = {LED1_GPIO_PORT, LED2_GPIO_PORT, LED3_GPIO_PORT};
const uint16_t GPIO_PIN[LEDn] = {LED1_PIN, LED2_PIN, LED3_PIN};
const uint32_t GPIO_CLK[LEDn] = {LED1_GPIO_CLK, LED2_GPIO_CLK, LED3_GPIO_CLK};

GPIO_TypeDef* BUTTON_PORT[BUTTONn] = {WAKEUP_BUTTON_GPIO_PORT, TAMPER_BUTTON_GPIO_PORT};

const uint16_t BUTTON_PIN[BUTTONn] = {WAKEUP_BUTTON_PIN, TAMPER_BUTTON_PIN};

const uint32_t BUTTON_CLK[BUTTONn] = {WAKEUP_BUTTON_GPIO_CLK, TAMPER_BUTTON_GPIO_CLK};

const uint16_t BUTTON_EXTI_LINE[BUTTONn] = {WAKEUP_BUTTON_EXTI_LINE,
                                            TAMPER_BUTTON_EXTI_LINE
                                           };

const uint16_t BUTTON_PORT_SOURCE[BUTTONn] = {WAKEUP_BUTTON_EXTI_PORT_SOURCE,
                                              TAMPER_BUTTON_EXTI_PORT_SOURCE
                                             };

const uint16_t BUTTON_PIN_SOURCE[BUTTONn] = {WAKEUP_BUTTON_EXTI_PIN_SOURCE,
                                             TAMPER_BUTTON_EXTI_PIN_SOURCE
                                            };

const uint16_t BUTTON_IRQn[BUTTONn] = {WAKEUP_BUTTON_EXTI_IRQn, TAMPER_BUTTON_EXTI_IRQn};

USART_TypeDef* COM_USART[COMn] = {EVAL_COM1};

GPIO_TypeDef* COM_TX_PORT[COMn] = {EVAL_COM1_TX_GPIO_PORT};

GPIO_TypeDef* COM_RX_PORT[COMn] = {EVAL_COM1_RX_GPIO_PORT};

const uint32_t COM_USART_CLK[COMn] = {EVAL_COM1_CLK};

const uint32_t COM_TX_PORT_CLK[COMn] = {EVAL_COM1_TX_GPIO_CLK};

const uint32_t COM_RX_PORT_CLK[COMn] = {EVAL_COM1_RX_GPIO_CLK};

const uint16_t COM_TX_PIN[COMn] = {EVAL_COM1_TX_PIN};

const uint16_t COM_RX_PIN[COMn] = {EVAL_COM1_RX_PIN};

/**
  * @}
  */


/** @defgroup STM3210E_EVAL_LOW_LEVEL_Private_FunctionPrototypes
  * @{
  */

/**
  * @}
  */


/** @defgroup STM3210E_EVAL_LOW_LEVEL_Private_Functions
  * @{
  */

/**
  * @brief  Configures LED GPIO.
  * @param  Led: Specifies the Led to be configured.
  *   This parameter can be one of following parameters:
  *     @arg LED1
  *     @arg LED2
  *     @arg LED3
  * @retval None
  */
void STM_EVAL_LEDInit(Led_TypeDef Led)
{
    GPIO_InitTypeDef  GPIO_InitStructure;

    /* Enable the GPIO_LED Clock */
    RCC_APB2PeriphClockCmd(GPIO_CLK[Led], ENABLE);

    if (Led == LED3)
    {
        RCC_APB2PeriphClockCmd(RCC_APB2Periph_AFIO, ENABLE);
        GPIO_PinRemapConfig(GPIO_Remap_SWJ_JTAGDisable, ENABLE);        
    }

    /* Configure the GPIO_LED pin */
    GPIO_InitStructure.GPIO_Pin = GPIO_PIN[Led];
    GPIO_InitStructure.GPIO_Mode = GPIO_Mode_Out_PP;
    GPIO_InitStructure.GPIO_Speed = GPIO_Speed_50MHz;

    GPIO_Init(GPIO_PORT[Led], &GPIO_InitStructure);
}

/**
  * @brief  Turns selected LED On.
  * @param  Led: Specifies the Led to be set on.
  *   This parameter can be one of following parameters:
  *     @arg LED1
  *     @arg LED2
  *     @arg LED3
  * @retval None
  */
void STM_EVAL_LEDOn(Led_TypeDef Led)
{
    GPIO_PORT[Led]->BRR = GPIO_PIN[Led];

}

/**
  * @brief  Turns selected LED Off.
  * @param  Led: Specifies the Led to be set off.
  *   This parameter can be one of following parameters:
  *     @arg LED1
  *     @arg LED2
  *     @arg LED3
  * @retval None
  */
void STM_EVAL_LEDOff(Led_TypeDef Led)
{
    GPIO_PORT[Led]->BSRR = GPIO_PIN[Led];
}

/**
  * @brief  Toggles the selected LED.
  * @param  Led: Specifies the Led to be toggled.
  *   This parameter can be one of following parameters:
  *     @arg LED1
  *     @arg LED2
  *     @arg LED3
  * @retval None
  */
void STM_EVAL_LEDToggle(Led_TypeDef Led)
{
    GPIO_PORT[Led]->ODR ^= GPIO_PIN[Led];
}

/**
  * @brief  Configures Button GPIO and EXTI Line.
  * @param  Button: Specifies the Button to be configured.
  *   This parameter can be one of following parameters:
  *     @arg BUTTON_WAKEUP: Wakeup Push Button
  *     @arg BUTTON_TAMPER: Tamper Push Button
  * @param  Button_Mode: Specifies Button mode.
  *   This parameter can be one of following parameters:
  *     @arg BUTTON_MODE_GPIO: Button will be used as simple IO
  *     @arg BUTTON_MODE_EXTI: Button will be connected to EXTI line with interrupt
  *                     generation capability
  * @retval None
  */
void STM_EVAL_PBInit(Button_TypeDef Button, ButtonMode_TypeDef Button_Mode)
{
    GPIO_InitTypeDef GPIO_InitStructure;
    EXTI_InitTypeDef EXTI_InitStructure;
    NVIC_InitTypeDef NVIC_InitStructure;

    /* Enable the BUTTON Clock */
    RCC_APB2PeriphClockCmd(BUTTON_CLK[Button] | RCC_APB2Periph_AFIO, ENABLE);

    /* Configure Button pin as input floating */
    GPIO_InitStructure.GPIO_Mode = GPIO_Mode_IN_FLOATING;
    GPIO_InitStructure.GPIO_Pin = BUTTON_PIN[Button];
    GPIO_Init(BUTTON_PORT[Button], &GPIO_InitStructure);


    if (Button_Mode == BUTTON_MODE_EXTI)
    {
        /* Connect Button EXTI Line to Button GPIO Pin */
        GPIO_EXTILineConfig(BUTTON_PORT_SOURCE[Button], BUTTON_PIN_SOURCE[Button]);

        /* Configure Button EXTI line */
        EXTI_InitStructure.EXTI_Line = BUTTON_EXTI_LINE[Button];
        EXTI_InitStructure.EXTI_Mode = EXTI_Mode_Interrupt;

        if (Button != BUTTON_WAKEUP)
        {
            EXTI_InitStructure.EXTI_Trigger = EXTI_Trigger_Falling;
        }
        else
        {
            EXTI_InitStructure.EXTI_Trigger = EXTI_Trigger_Rising;
        }

        EXTI_InitStructure.EXTI_LineCmd = ENABLE;
        EXTI_Init(&EXTI_InitStructure);

        /* Enable and set Button EXTI Interrupt to the lowest priority */
        NVIC_InitStructure.NVIC_IRQChannel = BUTTON_IRQn[Button];
        NVIC_InitStructure.NVIC_IRQChannelPreemptionPriority = 0x0F;
        NVIC_InitStructure.NVIC_IRQChannelSubPriority = 0x0F;
        NVIC_InitStructure.NVIC_IRQChannelCmd = ENABLE;

        NVIC_Init(&NVIC_InitStructure);
    }
}

/**
  * @brief  Returns the selected Button state.
  * @param  Button: Specifies the Button to be checked.
  *   This parameter can be one of following parameters:
  *     @arg BUTTON_WAKEUP: Wakeup Push Button
  *     @arg BUTTON_TAMPER: Tamper Push Button
  * @retval The Button GPIO pin value.
  */
uint32_t STM_EVAL_PBGetState(Button_TypeDef Button)
{
    return GPIO_ReadInputDataBit(BUTTON_PORT[Button], BUTTON_PIN[Button]);
}

/**
  * @brief  Configures COM port.
  * @param  COM: Specifies the COM port to be configured.
  *   This parameter can be one of following parameters:
  *     @arg COM1
  * @param  USART_InitStruct: pointer to a USART_InitTypeDef structure that
  *   contains the configuration information for the specified USART peripheral.
  * @retval None
  */
void STM_EVAL_COMInit(COM_TypeDef COM, USART_InitTypeDef* USART_InitStruct)
{
    GPIO_InitTypeDef GPIO_InitStructure;

    /* Enable GPIO clock */
    RCC_APB2PeriphClockCmd(COM_TX_PORT_CLK[COM] | COM_RX_PORT_CLK[COM] | RCC_APB2Periph_AFIO, ENABLE);

    /* Enable UART clock */
    if (COM == COM1)
    {
        RCC_APB2PeriphClockCmd(COM_USART_CLK[COM], ENABLE);
    }
    else
    {
        RCC_APB1PeriphClockCmd(COM_USART_CLK[COM], ENABLE);
    }

    /* Configure USART Tx as alternate function push-pull */
    GPIO_InitStructure.GPIO_Mode = GPIO_Mode_AF_PP;
    GPIO_InitStructure.GPIO_Pin = COM_TX_PIN[COM];
    GPIO_InitStructure.GPIO_Speed = GPIO_Speed_50MHz;
    GPIO_Init(COM_TX_PORT[COM], &GPIO_InitStructure);

    /* Configure USART Rx as input floating */
    GPIO_InitStructure.GPIO_Mode = GPIO_Mode_IN_FLOATING;
    GPIO_InitStructure.GPIO_Pin = COM_RX_PIN[COM];
    GPIO_Init(COM_RX_PORT[COM], &GPIO_InitStructure);

    /* USART configuration */
    USART_Init(COM_USART[COM], USART_InitStruct);

    /* Enable USART */
    USART_Cmd(COM_USART[COM], ENABLE);
}

/**
  * @brief  DeInitializes the SDIO interface.
  * @param  None
  * @retval None
  */
void SD_LowLevel_DeInit(void)
{
    GPIO_InitTypeDef  GPIO_InitStructure;

    /*!< Disable SDIO Clock */
    SDIO_ClockCmd(DISABLE);

    /*!< Set Power State to OFF */
    SDIO_SetPowerState(SDIO_PowerState_OFF);

    /*!< DeInitializes the SDIO peripheral */
    SDIO_DeInit();

    /*!< Disable the SDIO AHB Clock */
    RCC_AHBPeriphClockCmd(RCC_AHBPeriph_SDIO, DISABLE);

    /*!< Configure PC.08, PC.09, PC.10, PC.11, PC.12 pin: D0, D1, D2, D3, CLK pin */
    GPIO_InitStructure.GPIO_Pin = GPIO_Pin_8 | GPIO_Pin_9 | GPIO_Pin_10 | GPIO_Pin_11 | GPIO_Pin_12;
    GPIO_InitStructure.GPIO_Mode = GPIO_Mode_IN_FLOATING;
    GPIO_Init(GPIOC, &GPIO_InitStructure);

    /*!< Configure PD.02 CMD line */
    GPIO_InitStructure.GPIO_Pin = GPIO_Pin_2;
    GPIO_Init(GPIOD, &GPIO_InitStructure);
}

/**
  * @brief  Initializes the SD Card and put it into StandBy State (Ready for
  *         data transfer).
  * @param  None
  * @retval None
  */
void SD_LowLevel_Init(void)
{
    GPIO_InitTypeDef  GPIO_InitStructure;

    /*!< GPIOC and GPIOD Periph clock enable */
    RCC_APB2PeriphClockCmd(RCC_APB2Periph_GPIOC | RCC_APB2Periph_GPIOD | SD_DETECT_GPIO_CLK, ENABLE);

    /*!< Configure PC.08, PC.09, PC.10, PC.11, PC.12 pin: D0, D1, D2, D3, CLK pin */
    GPIO_InitStructure.GPIO_Pin = GPIO_Pin_8 | GPIO_Pin_9 | GPIO_Pin_10 | GPIO_Pin_11 | GPIO_Pin_12;
    GPIO_InitStructure.GPIO_Speed = GPIO_Speed_50MHz;
    GPIO_InitStructure.GPIO_Mode = GPIO_Mode_AF_PP;
    GPIO_Init(GPIOC, &GPIO_InitStructure);

    /*!< Configure PD.02 CMD line */
    GPIO_InitStructure.GPIO_Pin = GPIO_Pin_2;
    GPIO_Init(GPIOD, &GPIO_InitStructure);

    /*!< Configure SD_SPI_DETECT_PIN pin: SD Card detect pin */
    GPIO_InitStructure.GPIO_Pin = SD_DETECT_PIN;
    GPIO_InitStructure.GPIO_Mode = GPIO_Mode_IPU;
    GPIO_Init(SD_DETECT_GPIO_PORT, &GPIO_InitStructure);

    /*!< Enable the SDIO AHB Clock */
    RCC_AHBPeriphClockCmd(RCC_AHBPeriph_SDIO, ENABLE);

    /*!< Enable the DMA2 Clock */
    RCC_AHBPeriphClockCmd(SD_SDIO_DMA_CLK, ENABLE);
}

/**
  * @brief  Configures the DMA2 Channel4 for SDIO Tx request.
  * @param  BufferSRC: pointer to the source buffer
  * @param  BufferSize: buffer size
  * @retval None
  */
void SD_LowLevel_DMA_TxConfig(uint32_t *BufferSRC, uint32_t BufferSize)
{

    DMA_InitTypeDef DMA_InitStructure;

    DMA_ClearFlag(SD_SDIO_DMA_FLAG_TC | SD_SDIO_DMA_FLAG_TE | SD_SDIO_DMA_FLAG_HT | SD_SDIO_DMA_FLAG_GL);

    /*!< DMA2 Channel4 disable */
    DMA_Cmd(SD_SDIO_DMA_CHANNEL, DISABLE);

    /*!< SDIO DMA CHANNEL Config */
    DMA_InitStructure.DMA_PeripheralBaseAddr = (uint32_t)SDIO_FIFO_ADDRESS;
    DMA_InitStructure.DMA_MemoryBaseAddr = (uint32_t)BufferSRC;
    DMA_InitStructure.DMA_DIR = DMA_DIR_PeripheralDST;
    DMA_InitStructure.DMA_BufferSize = BufferSize / 4;
    DMA_InitStructure.DMA_PeripheralInc = DMA_PeripheralInc_Disable;
    DMA_InitStructure.DMA_MemoryInc = DMA_MemoryInc_Enable;
    DMA_InitStructure.DMA_PeripheralDataSize = DMA_PeripheralDataSize_Word;
    DMA_InitStructure.DMA_MemoryDataSize = DMA_MemoryDataSize_Word;
    DMA_InitStructure.DMA_Mode = DMA_Mode_Normal;
    DMA_InitStructure.DMA_Priority = DMA_Priority_High;
    DMA_InitStructure.DMA_M2M = DMA_M2M_Disable;
    DMA_Init(SD_SDIO_DMA_CHANNEL, &DMA_InitStructure);

    DMA_ITConfig(SD_SDIO_DMA_CHANNEL, DMA_IT_TC, ENABLE);

    /*!< SDIO DMA CHANNEL enable */
    DMA_Cmd(SD_SDIO_DMA_CHANNEL, ENABLE);
}

/**
  * @brief  Configures the DMA2 Channel4 for SDIO Rx request.
  * @param  BufferDST: pointer to the destination buffer
  * @param  BufferSize: buffer size
  * @retval None
  */
void SD_LowLevel_DMA_RxConfig(uint32_t *BufferDST, uint32_t BufferSize)
{
    DMA_InitTypeDef DMA_InitStructure;

    DMA_ClearFlag(SD_SDIO_DMA_FLAG_TC | SD_SDIO_DMA_FLAG_TE | SD_SDIO_DMA_FLAG_HT | SD_SDIO_DMA_FLAG_GL);

    /*!< SDIO DMA CHANNEL disable */
    DMA_Cmd(SD_SDIO_DMA_CHANNEL, DISABLE);

    /*!< SDIO DMA CHANNEL Config */
    DMA_InitStructure.DMA_PeripheralBaseAddr = (uint32_t)SDIO_FIFO_ADDRESS;
    DMA_InitStructure.DMA_MemoryBaseAddr = (uint32_t)BufferDST;
    DMA_InitStructure.DMA_DIR = DMA_DIR_PeripheralSRC;
    DMA_InitStructure.DMA_BufferSize = BufferSize / 4;
    DMA_InitStructure.DMA_PeripheralInc = DMA_PeripheralInc_Disable;
    DMA_InitStructure.DMA_MemoryInc = DMA_MemoryInc_Enable;
    DMA_InitStructure.DMA_PeripheralDataSize = DMA_PeripheralDataSize_Word;
    DMA_InitStructure.DMA_MemoryDataSize = DMA_MemoryDataSize_Word;
    DMA_InitStructure.DMA_Mode = DMA_Mode_Normal;
    DMA_InitStructure.DMA_Priority = DMA_Priority_High;
    DMA_InitStructure.DMA_M2M = DMA_M2M_Disable;
    DMA_Init(SD_SDIO_DMA_CHANNEL, &DMA_InitStructure);

    DMA_ITConfig(SD_SDIO_DMA_CHANNEL, DMA_IT_TC, ENABLE);

    /*!< SDIO DMA CHANNEL enable */
    DMA_Cmd(SD_SDIO_DMA_CHANNEL, ENABLE);
}

/**
  * @brief  Returns the DMA End Of Transfer Status.
  * @param  None
  * @retval DMA SDIO Channel Status.
  */
uint32_t SD_DMAEndOfTransferStatus(void)
{
    return (uint32_t)DMA_GetFlagStatus(DMA2_FLAG_TC4);
}

/**
  * @brief  DeInitializes the peripherals used by the SPI FLASH driver.
  * @param  None
  * @retval None
  */
void sFLASH_LowLevel_DeInit(void)
{
    GPIO_InitTypeDef GPIO_InitStructure;

    /*!< Disable the sFLASH_SPI  */
    SPI_Cmd(sFLASH_SPI, DISABLE);

    /*!< DeInitializes the sFLASH_SPI */
    SPI_I2S_DeInit(sFLASH_SPI);

    /*!< sFLASH_SPI Periph clock disable */
    RCC_APB2PeriphClockCmd(sFLASH_SPI_CLK, DISABLE);

    /*!< Configure sFLASH_SPI pins: SCK */
    GPIO_InitStructure.GPIO_Pin = sFLASH_SPI_SCK_PIN;
    GPIO_InitStructure.GPIO_Mode = GPIO_Mode_IN_FLOATING;
    GPIO_Init(sFLASH_SPI_SCK_GPIO_PORT, &GPIO_InitStructure);

    /*!< Configure sFLASH_SPI pins: MISO */
    GPIO_InitStructure.GPIO_Pin = sFLASH_SPI_MISO_PIN;
    GPIO_Init(sFLASH_SPI_MISO_GPIO_PORT, &GPIO_InitStructure);

    /*!< Configure sFLASH_SPI pins: MOSI */
    GPIO_InitStructure.GPIO_Pin = sFLASH_SPI_MOSI_PIN;
    GPIO_Init(sFLASH_SPI_MOSI_GPIO_PORT, &GPIO_InitStructure);

    /*!< Configure sFLASH_CS_PIN pin: sFLASH Card CS pin */
    GPIO_InitStructure.GPIO_Pin = sFLASH_CS_PIN;
    GPIO_Init(sFLASH_CS_GPIO_PORT, &GPIO_InitStructure);
}

/**
  * @brief  Initializes the peripherals used by the SPI FLASH driver.
  * @param  None
  * @retval None
  */
void sFLASH_LowLevel_Init(void)
{
    GPIO_InitTypeDef GPIO_InitStructure;

    /*!< sFLASH_SPI_CS_GPIO, sFLASH_SPI_MOSI_GPIO, sFLASH_SPI_MISO_GPIO
         and sFLASH_SPI_SCK_GPIO Periph clock enable */
    RCC_APB2PeriphClockCmd(sFLASH_CS_GPIO_CLK | sFLASH_SPI_MOSI_GPIO_CLK | sFLASH_SPI_MISO_GPIO_CLK |
                           sFLASH_SPI_SCK_GPIO_CLK, ENABLE);

    /*!< sFLASH_SPI Periph clock enable */
    RCC_APB1PeriphClockCmd(sFLASH_SPI_CLK, ENABLE);

    /*!< Configure sFLASH_SPI pins: SCK */
    GPIO_InitStructure.GPIO_Pin = sFLASH_SPI_SCK_PIN;
    GPIO_InitStructure.GPIO_Speed = GPIO_Speed_50MHz;
    GPIO_InitStructure.GPIO_Mode = GPIO_Mode_AF_PP;
    GPIO_Init(sFLASH_SPI_SCK_GPIO_PORT, &GPIO_InitStructure);

    /*!< Configure sFLASH_SPI pins: MOSI */
    GPIO_InitStructure.GPIO_Pin = sFLASH_SPI_MOSI_PIN;
    GPIO_Init(sFLASH_SPI_MOSI_GPIO_PORT, &GPIO_InitStructure);

    /*!< Configure sFLASH_SPI pins: MISO */
    GPIO_InitStructure.GPIO_Pin = sFLASH_SPI_MISO_PIN;
    GPIO_InitStructure.GPIO_Mode = GPIO_Mode_IN_FLOATING;
    GPIO_Init(sFLASH_SPI_MISO_GPIO_PORT, &GPIO_InitStructure);

    /*!< Configure sFLASH_CS_PIN pin: sFLASH Card CS pin */
    GPIO_InitStructure.GPIO_Pin = sFLASH_CS_PIN;
    GPIO_InitStructure.GPIO_Mode = GPIO_Mode_Out_PP;
    GPIO_Init(sFLASH_CS_GPIO_PORT, &GPIO_InitStructure);
}

/**
  * @brief  DeInitializes the I2C OLED.
  * @param  None
  * @retval None
  */
void I2C_LowLevel_DeInit(void)
{
    GPIO_InitTypeDef  GPIO_InitStructure;

    /*!< Disable OLED_I2C */
    I2C_Cmd(OLED_I2C, DISABLE);
    /*!< DeInitializes the OLED_I2C */
    I2C_DeInit(OLED_I2C);

    /*!< OLED_I2C Periph clock disable */
    RCC_APB1PeriphClockCmd(OLED_I2C_CLK, DISABLE);

    /*!< Configure OLED_I2C pins: SCL */
    GPIO_InitStructure.GPIO_Pin = OLED_I2C_SCL_PIN;
    GPIO_InitStructure.GPIO_Mode = GPIO_Mode_IN_FLOATING;
    GPIO_Init(OLED_I2C_SCL_GPIO_PORT, &GPIO_InitStructure);

    /*!< Configure OLED_I2C pins: SDA */
    GPIO_InitStructure.GPIO_Pin = OLED_I2C_SDA_PIN;
    GPIO_Init(OLED_I2C_SDA_GPIO_PORT, &GPIO_InitStructure);

}

/**
  * @brief  Initializes the OLED_I2C..
  * @param  None
  * @retval None
  */
void OLED_LowLevel_Init(void)
{
    GPIO_InitTypeDef  GPIO_InitStructure;

    /*!< OLED_I2C Periph clock enable */
    RCC_APB1PeriphClockCmd(OLED_I2C_CLK, ENABLE);

    /*!< OLED_I2C_SCL_GPIO_CLK, OLED_I2C_SDA_GPIO_CLK Periph clock enable */
    RCC_APB2PeriphClockCmd(OLED_I2C_SCL_GPIO_CLK | OLED_I2C_SDA_GPIO_CLK, ENABLE);

    /*!< Configure OLED_I2C pins: SCL */
    GPIO_InitStructure.GPIO_Pin = OLED_I2C_SCL_PIN;
    GPIO_InitStructure.GPIO_Speed = GPIO_Speed_50MHz;
    GPIO_InitStructure.GPIO_Mode = GPIO_Mode_AF_OD;
    GPIO_Init(OLED_I2C_SCL_GPIO_PORT, &GPIO_InitStructure);

    /*!< Configure OLED_I2C pins: SDA */
    GPIO_InitStructure.GPIO_Pin = OLED_I2C_SDA_PIN;
    GPIO_Init(OLED_I2C_SDA_GPIO_PORT, &GPIO_InitStructure);


}

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

/**
  * @}
  */

/************************ (C) COPYRIGHT STMicroelectronics *****END OF FILE****/
