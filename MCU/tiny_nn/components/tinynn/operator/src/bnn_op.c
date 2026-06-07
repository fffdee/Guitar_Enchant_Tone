#include "bnn_op/bnn_op.h"

static const bnn_op_backend_t *g_backend = NULL;

const bnn_op_backend_t *bnn_op_get_backend(void) {
    if (!g_backend) g_backend = bnn_op_cpu_backend();
    return g_backend;
}

void bnn_op_set_backend(const bnn_op_backend_t *be) {
    g_backend = be ? be : bnn_op_cpu_backend();
}
