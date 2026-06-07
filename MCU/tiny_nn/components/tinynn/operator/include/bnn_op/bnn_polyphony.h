#ifndef BNN_POLYPHONY_H
#define BNN_POLYPHONY_H

#ifdef __cplusplus
extern "C" {
#endif

/*
 * 复音/单音判别 (对应 §8) —— 由单帧幅度谱估计:
 *   - 显著谱峰计数 (局部极大且高于相对阈值);
 *   - 谱平坦度 (几何均值/算术均值, 噪声样高、谐波样低).
 * 用于驱动"谱掩码(复音) vs DDSP(单音深变换)"双模切换. 纯统计, 无状态.
 */
typedef struct bnn_poly_cfg {
    float peak_rel_thresh;   /* 峰值需 > 该比例 * 最大谱值 (默认 0.1) */
    int   peak_count_thresh; /* 峰数 >= 该值判为复音 (默认 6) */
} bnn_poly_cfg_t;

typedef struct bnn_poly_result {
    int   n_peaks;
    float flatness;          /* [0,1], 越大越噪声样 */
    int   is_poly;           /* 1=复音, 0=单音 */
} bnn_poly_result_t;

static inline void bnn_poly_cfg_default(bnn_poly_cfg_t *c) {
    c->peak_rel_thresh = 0.1f; c->peak_count_thresh = 6;
}

/* 分析单帧幅度谱 mag[n_bins]. cfg 可为 NULL(用默认). */
void bnn_polyphony_analyze(const float *mag, int n_bins,
                           const bnn_poly_cfg_t *cfg, bnn_poly_result_t *out);

#ifdef __cplusplus
}
#endif
#endif
