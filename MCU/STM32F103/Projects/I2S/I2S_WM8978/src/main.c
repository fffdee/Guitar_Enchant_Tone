/**
  ******************************************************************************
  * @file    main.c
  * @author  yidianusb
  * @version V1.0.0
  * @date    2026-02-xx
  * @brief   This file provides all the Application firmware functions.
  ******************************************************************************
  * @attention
  *
  * 实验平台: 亿点-起点STM32F103_USB开发板
  * 淘    宝: https://yidianusb.taobao.com
  *
  ******************************************************************************
  */


/* Includes ------------------------------------------------------------------*/
#include "stm3210e_eval.h"
#include "bsp_wm8978.h"
#include <stdio.h>

/* Private typedef -----------------------------------------------------------*/
/* Private define ------------------------------------------------------------*/
/* Private macro -------------------------------------------------------------*/
/* Private variables ---------------------------------------------------------*/
/* Extern variables ----------------------------------------------------------*/
/* Private function prototypes -----------------------------------------------*/
/* Private functions ---------------------------------------------------------*/

/*******************************************************************************
* Function Name  : main.
* Description    : main routine.
* Input          : None.
* Output         : None.
* Return         : None.
*******************************************************************************/
int main(void)
{
    /* Initialize Leds mounted on STM3210X-EVAL board */
    STM_EVAL_LEDInit(LED1);
    STM_EVAL_LEDInit(LED2);
    STM_EVAL_LEDInit(LED3);
    STM_EVAL_LEDOff(LED1);
    STM_EVAL_LEDOff(LED2);
    STM_EVAL_LEDOff(LED3);
    
    /* WM8978 Config */
    /* Call low layer function */
    if (wm8978_Init() == 0)
    {
        STM_EVAL_LEDOn(LED1);
        STM_EVAL_LEDOn(LED2);
        STM_EVAL_LEDOn(LED3);  
        
        while (1)
        {
        }
    }
    
    /* 配置WM8978芯片，输出为耳机和喇叭 */
    wm8978_CfgAudioPath(MIC_LEFT_ON | MIC_RIGHT_ON | ADC_ON | DAC_ON, EAR_LEFT_ON | EAR_RIGHT_ON | SPK_ON);

    /* 调节音量，左右相同音量 */
    wm8978_SetOUT1Volume(VOLUME_MAX*0.5);
    wm8978_SetOUT2Volume(VOLUME_MAX*0.5);

    /* 配置WM8978音频接口为飞利浦标准I2S接口，16bit */
    wm8978_CfgAudioIF(I2S_Standard_Phillips, 16);

    /*  初始化并配置I2S  */
    I2S_GPIO_Config();
    I2Sx_Mode_Config(I2S_Standard_Phillips, I2S_DataFormat_16b, I2S_AudioFreq_8k);
    I2Sx_TX_DMA_Init();

    DMA_Cmd(I2Sx_TX_DMA_CHANNEL, ENABLE); //开启DMA TX传输,开始播放
    I2S_Cmd(WM8978_I2Sx_SPI, ENABLE);

    while (1)
    {

    }
}


#ifdef  USE_FULL_ASSERT
/*******************************************************************************
* Function Name  : assert_failed
* Description    : Reports the name of the source file and the source line number
*                  where the assert_param error has occurred.
* Input          : - file: pointer to the source file name
*                  - line: assert_param error line source number
* Output         : None
* Return         : None
*******************************************************************************/
void assert_failed(uint8_t* file, uint32_t line)
{
    /* User can add his own implementation to report the file name and line number,
       ex: printf("Wrong parameters value: file %s on line %d\r\n", file, line) */

    /* Infinite loop */
    while (1)
    {
    }
}
#endif

/************************ (C) COPYRIGHT STMicroelectronics *****END OF FILE****/
