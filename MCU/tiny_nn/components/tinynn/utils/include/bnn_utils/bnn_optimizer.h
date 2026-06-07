#ifndef BNN_OPTIMIZER_H
#define BNN_OPTIMIZER_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    BNN_OPT_SGD      = 0,   /* 可带 momentum / nesterov */
    BNN_OPT_ADAM     = 1,
    BNN_OPT_ADAMW    = 2,   /* 解耦权重衰减 */
    BNN_OPT_RMSPROP  = 3,
} bnn_optimizer_type_t;

typedef struct bnn_param {
    float  *data;
    float  *grad;
    size_t  numel;
    float  *m;   /* 一阶动量 / RMSProp 中的 mean square */
    float  *v;   /* Adam 二阶动量 */
    struct bnn_param *next;
} bnn_param_t;

typedef struct bnn_optimizer {
    bnn_optimizer_type_t type;
    float lr;
    float beta1, beta2, eps;
    float momentum;
    float weight_decay;
    int   nesterov;
    int   t;
    bnn_param_t *params;
} bnn_optimizer_t;

/* 构造 */
bnn_optimizer_t *bnn_optimizer_sgd(float lr, float weight_decay);
bnn_optimizer_t *bnn_optimizer_sgd_momentum(float lr, float momentum, float weight_decay, int nesterov);
bnn_optimizer_t *bnn_optimizer_adam(float lr, float beta1, float beta2, float eps);
bnn_optimizer_t *bnn_optimizer_adamw(float lr, float beta1, float beta2, float eps, float weight_decay);
bnn_optimizer_t *bnn_optimizer_rmsprop(float lr, float alpha, float eps);

void bnn_optimizer_free(bnn_optimizer_t *opt);
void bnn_optimizer_add_param(bnn_optimizer_t *opt, float *data, float *grad, size_t numel);
void bnn_optimizer_zero_grad(bnn_optimizer_t *opt);
void bnn_optimizer_step(bnn_optimizer_t *opt);

/* 动态设置学习率 (供 scheduler 使用) */
void  bnn_optimizer_set_lr(bnn_optimizer_t *opt, float lr);
float bnn_optimizer_get_lr(const bnn_optimizer_t *opt);

/* ============= LR Scheduler ============= */
typedef enum {
    BNN_LR_STEP   = 0,   /* lr = base * gamma^(epoch / step_size) */
    BNN_LR_COSINE = 1,   /* cosine annealing: lr = lr_min + 0.5*(base-lr_min)*(1+cos(pi*epoch/total)) */
    BNN_LR_EXP    = 2,   /* lr = base * gamma^epoch */
} bnn_lr_kind_t;

typedef struct {
    bnn_lr_kind_t kind;
    float base_lr;
    float gamma;       /* step / exp */
    int   step_size;   /* step */
    int   total;       /* cosine */
    float lr_min;      /* cosine */
} bnn_lr_scheduler_t;

void bnn_lr_init_step  (bnn_lr_scheduler_t *s, float base, int step_size, float gamma);
void bnn_lr_init_cosine(bnn_lr_scheduler_t *s, float base, int total, float lr_min);
void bnn_lr_init_exp   (bnn_lr_scheduler_t *s, float base, float gamma);
void bnn_lr_step       (bnn_lr_scheduler_t *s, bnn_optimizer_t *opt, int epoch);

#ifdef __cplusplus
}
#endif
#endif
