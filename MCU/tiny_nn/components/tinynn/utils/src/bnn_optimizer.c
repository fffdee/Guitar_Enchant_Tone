#include "bnn_utils/bnn_optimizer.h"
#include "bnn_utils/bnn_mem.h"
#include <math.h>
#include <string.h>

#ifndef BNN_PI
#define BNN_PI 3.14159265358979323846f
#endif

static bnn_optimizer_t *new_opt(bnn_optimizer_type_t type, float lr) {
    bnn_optimizer_t *o = (bnn_optimizer_t *)bnn_calloc(1, sizeof(*o));
    if (!o) return NULL;
    o->type   = type;
    o->lr     = lr;
    o->beta1  = 0.9f;
    o->beta2  = 0.999f;
    o->eps    = 1e-8f;
    return o;
}

bnn_optimizer_t *bnn_optimizer_sgd(float lr, float weight_decay) {
    bnn_optimizer_t *o = new_opt(BNN_OPT_SGD, lr);
    if (o) o->weight_decay = weight_decay;
    return o;
}
bnn_optimizer_t *bnn_optimizer_sgd_momentum(float lr, float momentum, float wd, int nesterov) {
    bnn_optimizer_t *o = new_opt(BNN_OPT_SGD, lr);
    if (o) { o->momentum = momentum; o->weight_decay = wd; o->nesterov = nesterov; }
    return o;
}
bnn_optimizer_t *bnn_optimizer_adam(float lr, float b1, float b2, float eps) {
    bnn_optimizer_t *o = new_opt(BNN_OPT_ADAM, lr);
    if (o) { o->beta1 = b1; o->beta2 = b2; o->eps = eps; }
    return o;
}
bnn_optimizer_t *bnn_optimizer_adamw(float lr, float b1, float b2, float eps, float wd) {
    bnn_optimizer_t *o = new_opt(BNN_OPT_ADAMW, lr);
    if (o) { o->beta1 = b1; o->beta2 = b2; o->eps = eps; o->weight_decay = wd; }
    return o;
}
bnn_optimizer_t *bnn_optimizer_rmsprop(float lr, float alpha, float eps) {
    bnn_optimizer_t *o = new_opt(BNN_OPT_RMSPROP, lr);
    if (o) { o->beta2 = alpha; o->eps = eps; }
    return o;
}

void bnn_optimizer_free(bnn_optimizer_t *opt) {
    if (!opt) return;
    bnn_param_t *p = opt->params;
    while (p) {
        bnn_param_t *n = p->next;
        if (p->m) bnn_free(p->m);
        if (p->v) bnn_free(p->v);
        bnn_free(p);
        p = n;
    }
    bnn_free(opt);
}

void bnn_optimizer_add_param(bnn_optimizer_t *opt, float *data, float *grad, size_t numel) {
    if (!opt || !data || !grad) return;
    bnn_param_t *p = (bnn_param_t *)bnn_calloc(1, sizeof(*p));
    if (!p) return;
    p->data = data; p->grad = grad; p->numel = numel;
    int need_m = 0, need_v = 0;
    switch (opt->type) {
        case BNN_OPT_SGD:    need_m = opt->momentum != 0.f; break;
        case BNN_OPT_RMSPROP:need_m = 1; break;
        case BNN_OPT_ADAM:
        case BNN_OPT_ADAMW:  need_m = 1; need_v = 1; break;
    }
    if (need_m) p->m = (float *)bnn_calloc(numel, sizeof(float));
    if (need_v) p->v = (float *)bnn_calloc(numel, sizeof(float));
    p->next = opt->params;
    opt->params = p;
}

void bnn_optimizer_zero_grad(bnn_optimizer_t *opt) {
    if (!opt) return;
    for (bnn_param_t *p = opt->params; p; p = p->next)
        memset(p->grad, 0, p->numel * sizeof(float));
}

void bnn_optimizer_step(bnn_optimizer_t *opt) {
    if (!opt) return;
    opt->t++;
    switch (opt->type) {
    case BNN_OPT_SGD: {
        float lr = opt->lr, wd = opt->weight_decay, mom = opt->momentum;
        for (bnn_param_t *p = opt->params; p; p = p->next) {
            for (size_t i = 0; i < p->numel; ++i) {
                float g = p->grad[i] + wd * p->data[i];
                if (mom != 0.f) {
                    p->m[i] = mom * p->m[i] + g;
                    float upd = opt->nesterov ? (g + mom * p->m[i]) : p->m[i];
                    p->data[i] -= lr * upd;
                } else {
                    p->data[i] -= lr * g;
                }
            }
        }
    } break;
    case BNN_OPT_ADAM: {
        float lr = opt->lr, b1 = opt->beta1, b2 = opt->beta2, eps = opt->eps;
        float bc1 = 1.f - powf(b1, (float)opt->t);
        float bc2 = 1.f - powf(b2, (float)opt->t);
        for (bnn_param_t *p = opt->params; p; p = p->next) {
            for (size_t i = 0; i < p->numel; ++i) {
                float g = p->grad[i];
                p->m[i] = b1 * p->m[i] + (1.f - b1) * g;
                p->v[i] = b2 * p->v[i] + (1.f - b2) * g * g;
                float mh = p->m[i] / bc1;
                float vh = p->v[i] / bc2;
                p->data[i] -= lr * mh / (sqrtf(vh) + eps);
            }
        }
    } break;
    case BNN_OPT_ADAMW: {
        float lr = opt->lr, b1 = opt->beta1, b2 = opt->beta2, eps = opt->eps, wd = opt->weight_decay;
        float bc1 = 1.f - powf(b1, (float)opt->t);
        float bc2 = 1.f - powf(b2, (float)opt->t);
        for (bnn_param_t *p = opt->params; p; p = p->next) {
            for (size_t i = 0; i < p->numel; ++i) {
                float g = p->grad[i];
                p->m[i] = b1 * p->m[i] + (1.f - b1) * g;
                p->v[i] = b2 * p->v[i] + (1.f - b2) * g * g;
                float mh = p->m[i] / bc1;
                float vh = p->v[i] / bc2;
                p->data[i] -= lr * (mh / (sqrtf(vh) + eps) + wd * p->data[i]);
            }
        }
    } break;
    case BNN_OPT_RMSPROP: {
        float lr = opt->lr, a = opt->beta2, eps = opt->eps;
        for (bnn_param_t *p = opt->params; p; p = p->next) {
            for (size_t i = 0; i < p->numel; ++i) {
                float g = p->grad[i];
                p->m[i] = a * p->m[i] + (1.f - a) * g * g;
                p->data[i] -= lr * g / (sqrtf(p->m[i]) + eps);
            }
        }
    } break;
    }
}

void  bnn_optimizer_set_lr(bnn_optimizer_t *opt, float lr) { if (opt) opt->lr = lr; }
float bnn_optimizer_get_lr(const bnn_optimizer_t *opt) { return opt ? opt->lr : 0.f; }

/* ============ Scheduler ============ */
void bnn_lr_init_step(bnn_lr_scheduler_t *s, float base, int step_size, float gamma) {
    s->kind = BNN_LR_STEP; s->base_lr = base; s->step_size = step_size; s->gamma = gamma;
}
void bnn_lr_init_cosine(bnn_lr_scheduler_t *s, float base, int total, float lr_min) {
    s->kind = BNN_LR_COSINE; s->base_lr = base; s->total = total; s->lr_min = lr_min;
}
void bnn_lr_init_exp(bnn_lr_scheduler_t *s, float base, float gamma) {
    s->kind = BNN_LR_EXP; s->base_lr = base; s->gamma = gamma;
}
void bnn_lr_step(bnn_lr_scheduler_t *s, bnn_optimizer_t *opt, int epoch) {
    if (!s || !opt) return;
    float lr = s->base_lr;
    switch (s->kind) {
    case BNN_LR_STEP:
        lr = s->base_lr * powf(s->gamma, (float)(epoch / (s->step_size > 0 ? s->step_size : 1)));
        break;
    case BNN_LR_COSINE: {
        int t = s->total > 0 ? s->total : 1;
        int e = epoch > t ? t : epoch;
        lr = s->lr_min + 0.5f * (s->base_lr - s->lr_min) * (1.f + cosf(BNN_PI * (float)e / (float)t));
    } break;
    case BNN_LR_EXP:
        lr = s->base_lr * powf(s->gamma, (float)epoch);
        break;
    }
    opt->lr = lr;
}
