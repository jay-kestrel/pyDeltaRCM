#include "model.h"

#include <stdio.h>

/*
 * Foundation stub: the update pipeline is filled in by the water-routing,
 * sediment-routing, and diffusion commits. Fail fast rather than run a
 * silent no-op and emit meaningless "parity" outputs.
 */
int pyd_update(grid *g, int n)
{
    (void)g;
    fprintf(stderr,
            "pydeltarcm_c: model updates not implemented yet (%d requested)\n",
            n);
    return -1;
}
