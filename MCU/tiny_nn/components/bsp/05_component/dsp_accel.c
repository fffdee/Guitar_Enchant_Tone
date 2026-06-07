#include "dsp_accel.h"
#include <string.h>
#include "esp_log.h"
#include "esp_heap_caps.h"
#include "dsps_fft2r.h"

static const char *TAG = "dsp_accel";

static float *s_tmp     = NULL;   /* 交错复数 scratch, 长度 2*max_n (内部 RAM) */
static int    s_max_n   = 0;
static int    s_ready   = 0;

static int is_pow2(int n) { return n > 0 && (n & (n - 1)) == 0; }

esp_err_t dsp_accel_init(int max_n)
{
    if (s_ready && max_n <= s_max_n) return ESP_OK;
    if (!is_pow2(max_n)) {
        ESP_LOGE(TAG, "max_n=%d 不是 2 的幂", max_n);
        return ESP_ERR_INVALID_ARG;
    }
    /* esp-dsp twiddle 表 (内部分配, 大小 max_n 个 float) */
    esp_err_t e = dsps_fft2r_init_fc32(NULL, max_n);
    if (e != ESP_OK) {
        ESP_LOGE(TAG, "dsps_fft2r_init_fc32(%d) failed: 0x%x", max_n, e);
        return e;
    }
    /* scratch 放内部 RAM (FFT 频繁随机访问, 不能放 PSRAM) */
    if (s_tmp) { heap_caps_free(s_tmp); s_tmp = NULL; }
    s_tmp = (float *)heap_caps_malloc(sizeof(float) * 2 * max_n, MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
    if (!s_tmp) {
        ESP_LOGE(TAG, "scratch OOM (%d floats)", 2 * max_n);
        return ESP_ERR_NO_MEM;
    }
    s_max_n = max_n;
    s_ready = 1;
    ESP_LOGI(TAG, "ready: esp-dsp FFT max_n=%d (P4 单精度硬件 FPU)", max_n);
    return ESP_OK;
}

/* 正变换 (dir<0, e^{-j}) 或未归一逆变换 (dir>0, e^{+j}), 原地. data[2*n] 交错. */
static void accel_cfft(float *data, int n, int dir)
{
    if (!s_ready || n > s_max_n || !is_pow2(n)) return;
    if (dir > 0)
        for (int i = 0; i < n; ++i) data[2 * i + 1] = -data[2 * i + 1];  /* 共轭 */
    dsps_fft2r_fc32(data, n);
    dsps_bit_rev_fc32(data, n);
    if (dir > 0)
        for (int i = 0; i < n; ++i) data[2 * i + 1] = -data[2 * i + 1];  /* 共轭回 (未归一) */
}

static void accel_rfft(const float *in, float *out_complex, int n)
{
    if (!s_ready || n > s_max_n || !is_pow2(n)) return;
    for (int i = 0; i < n; ++i) { s_tmp[2 * i] = in[i]; s_tmp[2 * i + 1] = 0.f; }
    dsps_fft2r_fc32(s_tmp, n);
    dsps_bit_rev_fc32(s_tmp, n);
    memcpy(out_complex, s_tmp, sizeof(float) * 2 * (size_t)(n / 2 + 1));
}

static void accel_irfft(const float *in_complex, float *out, int n)
{
    if (!s_ready || n > s_max_n || !is_pow2(n)) return;
    int half = n / 2 + 1;
    for (int k = 0; k < half; ++k) {
        s_tmp[2 * k]     = in_complex[2 * k];
        s_tmp[2 * k + 1] = in_complex[2 * k + 1];
    }
    for (int k = half; k < n; ++k) {        /* 厄米镜像重建完整谱 */
        int src = n - k;
        s_tmp[2 * k]     =  in_complex[2 * src];
        s_tmp[2 * k + 1] = -in_complex[2 * src + 1];
    }
    /* 逆变换 = (1/N) conj(FFT(conj(x))); 厄米输入 -> 实数输出, 取实部 */
    for (int i = 0; i < n; ++i) s_tmp[2 * i + 1] = -s_tmp[2 * i + 1];
    dsps_fft2r_fc32(s_tmp, n);
    dsps_bit_rev_fc32(s_tmp, n);
    float inv = 1.0f / (float)n;
    for (int i = 0; i < n; ++i) out[i] = s_tmp[2 * i] * inv;
}

static const bnn_dsp_backend_t s_backend = {
    .name  = "esp_dsp_p4",
    .rfft  = accel_rfft,
    .irfft = accel_irfft,
    .cfft  = accel_cfft,
};

const bnn_dsp_backend_t *dsp_accel_backend(void) { return &s_backend; }
