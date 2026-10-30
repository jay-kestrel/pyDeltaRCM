#ifndef PYD_GRID_H
#define PYD_GRID_H

#include "rng.h"

#include <stddef.h>
#include <stdint.h>

/*
 * pyDeltaRCM model state loaded from a normalized benchmark fixture.
 *
 * Dtype policy mirrors the Python reference exactly (spec: mixed precision
 * is reproduced, never "improved"):
 *   - state fields are float32 (eta, depth, stage, fluxes, velocity, qs,
 *     deposition volumes, sand_frac, active_layer, sfc accumulators, the
 *     init_derived modifier fields)
 *   - scalars are float64
 *   - the water weight array is float64
 *   - walk tables are int64, ivec/jvec/distances are float32
 *
 * qx/qy/qw/qwn are float32 *persisting* across updates: the Python probe
 * (task todo_CP1v4hBZ session log) confirmed the float64 temporaries inside
 * update_flow_field are stored back as float32, so the port stores float32.
 */
typedef struct {
    int L, W;

    /* float64 scalars (config.json names in comments) */
    uint32_t seed;             /* _seed */
    double dx;                 /* _dx */
    double dt;                 /* _dt */
    double gamma;              /* _gamma */
    double theta_water;        /* _theta_water */
    double theta_sand;         /* theta_sand (derived) */
    double theta_mud;          /* theta_mud (derived) */
    double u_max;              /* u_max */
    double U_dep_mud;          /* U_dep_mud */
    double U_ero_mud;          /* U_ero_mud */
    double U_ero_sand;         /* U_ero_sand */
    double f_bedload;          /* _f_bedload */
    double C0_percent;         /* _C0_percent */
    double dry_depth;          /* dry_depth */
    double S0;                 /* _S0 */
    double h0;                 /* _h0 */
    double hb;                 /* _hb */
    double H_SL;               /* _H_SL */
    double SLR;                /* _SLR */
    double omega_flow;         /* _omega_flow */
    double omega_sfc;          /* _omega_sfc */
    double omega_flow_iter;    /* omega_flow_iter */
    double alpha;              /* _alpha */
    double Csmooth;            /* _Csmooth */
    double lambda;             /* _lambda */
    double beta;               /* _beta */
    double stepmax;            /* _stepmax */
    double Vp_sed;             /* Vp_sed */
    double dVs;                /* dVs */
    double qw0;                /* qw0 */
    double qs0;                /* qs0 */
    double u0;                 /* _u0 */
    double Qw0;                /* Qw0 */
    double Qs0;                /* Qs0 */
    double V0;                 /* V0 */
    double Qp_water;           /* Qp_water */
    double g;                  /* g */
    double diffusion_multiplier; /* diffusion_multiplier */
    double CTR;                /* CTR */
    double active_layer_thickness; /* _active_layer_thickness */
    double sand_frac_bc;       /* _sand_frac_bc */
    double subsidence_rate;    /* _subsidence_rate */
    double start_subsidence;   /* _start_subsidence */
    double time;               /* _time, advances by dt per update */

    /* integer scalars */
    int itermax;               /* _itermax */
    int Np_water;              /* _Np_water */
    int Np_sed;                /* _Np_sed */
    int N0;                    /* N0 (inlet width in cells) */
    int L0;                    /* L0 */
    int N_crossdiff;           /* N_crossdiff */
    int Nsmooth;               /* _Nsmooth */
    int toggle_subsidence;     /* _toggle_subsidence */
    int force_deposit;         /* _force_deposit */
    long time_iter;            /* _time_iter */

    /* float32 state fields, L*W each */
    float *eta, *eta0, *depth, *stage, *Hnew;
    float *qw, *qx, *qy, *qxn, *qyn, *qwn;
    float *uw, *ux, *uy, *qs;
    float *Vp_dep_sand, *Vp_dep_mud, *sand_frac, *active_layer;
    float *sfc_sum, *sfc_visit;
    float *mod_water_weight, *mod_sed_weight, *mod_stable_weight, *mod_erosion;

    /* integer fields */
    int64_t *cell_type;        /* L*W */
    int64_t *inlet;            /* n_inlet flat column indices */
    int     n_inlet;

    /* padded (L+2)*(W+2) buffers, refilled per routing pass */
    float   *pad_eta, *pad_depth;
    int64_t *pad_cell_type;

    /* init_derived walk tables (flat 9 entries) */
    float   ivec[9], jvec[9], distances[9];
    int64_t iwalk[9], jwalk[9], ravel_walk_flat[9];
    int64_t kernel1[9], kernel2[9];

    /* water weight array, L*W*9 float64 */
    double *water_weights;

    mt_t rng;
} grid;

/* Load config.json + all state arrays from a fixture directory.
 * Seeds the RNG with the fixture seed. Returns 0 on success. */
int  grid_init(grid *g, const char *fixture_dir);

/* Free all allocations owned by the grid. */
void grid_free(grid *g);

/* Pad a float32 (L,W) field with edge replication into dst ((L+2)*(W+2)). */
void grid_pad_f32(const float *src, float *dst, int L, int W);
/* Pad an int64 (L,W) field with edge replication. */
void grid_pad_i64(const int64_t *src, int64_t *dst, int L, int W);

#endif
