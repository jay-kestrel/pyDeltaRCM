#ifndef PYD_MODEL_H
#define PYD_MODEL_H

#include "grid.h"

/* Run n full model updates (water routing, sediment routing, topo
 * diffusion, subsidence, finalization) in reference order.
 * Returns 0 on success. The per-implementation compute clock is read
 * around this call by the driver (I/O excluded). */
int pyd_update(grid *g, int n);

#endif
