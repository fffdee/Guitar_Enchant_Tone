#include "bnn_op/bnn_nn.h"
#include <stddef.h>

static const bnn_nn_backend_t *g_nn_backend = NULL;

void bnn_nn_set_backend(const bnn_nn_backend_t *be)
{
    g_nn_backend = be;
}

const bnn_nn_backend_t *bnn_nn_get_backend(void)
{
    return g_nn_backend;
}
