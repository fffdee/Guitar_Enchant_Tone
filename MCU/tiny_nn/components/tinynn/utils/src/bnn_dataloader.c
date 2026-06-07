#include "bnn_utils/bnn_dataloader.h"
#include "bnn_utils/bnn_mem.h"
#include <string.h>

static unsigned int xrand(unsigned int *s) { *s = (*s) * 1664525u + 1013904223u; return *s; }

void bnn_dataloader_init(bnn_dataloader_t *dl, const bnn_dataset_t *ds, int batch, int shuffle, unsigned int seed) {
    dl->ds = ds; dl->batch = batch; dl->shuffle = shuffle;
    dl->cursor = 0; dl->rng = seed ? seed : 0xDEADBEEFu;
    dl->index = (int *)bnn_malloc(sizeof(int) * ds->num);
    for (int i = 0; i < ds->num; ++i) dl->index[i] = i;
    bnn_dataloader_reset(dl);
}
void bnn_dataloader_free(bnn_dataloader_t *dl) {
    if (dl && dl->index) { bnn_free(dl->index); dl->index = NULL; }
}
void bnn_dataloader_reset(bnn_dataloader_t *dl) {
    dl->cursor = 0;
    if (dl->shuffle) {
        for (int i = dl->ds->num - 1; i > 0; --i) {
            int j = (int)(xrand(&dl->rng) % (unsigned)(i + 1));
            int t = dl->index[i]; dl->index[i] = dl->index[j]; dl->index[j] = t;
        }
    }
}
int bnn_dataloader_next(bnn_dataloader_t *dl, float *x_out, float *y_out) {
    if (!dl || dl->cursor >= dl->ds->num) return 0;
    int n = dl->batch;
    if (dl->cursor + n > dl->ds->num) n = dl->ds->num - dl->cursor;
    int fd = dl->ds->feat_dim, ld = dl->ds->label_dim;
    for (int i = 0; i < n; ++i) {
        int idx = dl->index[dl->cursor + i];
        if (x_out) memcpy(x_out + i * fd, dl->ds->x + (size_t)idx * fd, fd * sizeof(float));
        if (y_out) memcpy(y_out + i * ld, dl->ds->y + (size_t)idx * ld, ld * sizeof(float));
    }
    dl->cursor += n;
    return n;
}
