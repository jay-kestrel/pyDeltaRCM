using Test
using DeltaRCMJulia
const DJ = DeltaRCMJulia

@testset "water weights: bit-exact against the numba kernel" begin
    st = load_water_state()
    ref = load_reference(1)
    WW = DJ._get_water_weight_array(st.depth, st.stage, st.mod_water_weight,
                                    st.cell_type, st.qx, st.qy,
                                    st.ivec_flat, st.jvec_flat,
                                    st.distances_flat, st.dry_depth,
                                    st.gamma, st.theta_water)
    bit_identical(WW, ref["water_weights_iter0"], "water_weights_iter0")
end

@testset "start indices: RNG-consumption parity" begin
    st = load_water_state()
    ref = load_reference(1)
    rng = DJ.MT19937()
    DJ.set_rng_state!(rng, ref["rng_before"]...)
    inlet_weights = DJ.get_inlet_weights(st.inlet)
    starts = DJ.get_start_indices(rng, st.inlet, inlet_weights, st.Np_water)
    bit_identical(starts, ref["start_indices_iter0"], "start_indices_iter0")

    # and through the phase method, which must consume exactly one draw set
    st2 = load_water_state()
    rng2 = DJ.MT19937()
    DJ.set_rng_state!(rng2, ref["rng_before"]...)
    DJ.init_water_iteration!(st2, rng2)
    bit_identical(st2.start_indices, ref["start_indices_iter0"],
                  "start_indices via init_water_iteration!")
end

@testset "free-surface accumulation and smoothing kernels" begin
    # drive the kernels with the reference update-1 walk outputs, compare to
    # the reference iter0 accumulators and the smoothed surface
    st = load_water_state()
    ref = load_reference(1)
    walk = ref["free_surf_walk_inds"]        # julia layout (size_indices, Np)
    flag = vec(ref["free_surf_flag"])        # python (Np,) int64 vector
    sfc_visit, sfc_sum = DJ._accumulate_free_surface_walks(
        walk, flag, st.cell_type, st.uw, st.ux, st.uy, st.depth, st.dx,
        st.u0, st.h0, st.H_SL, st.S0, st.L, st.W)
    bit_identical(sfc_visit, ref["sfc_visit"], "sfc_visit")
    bit_identical(sfc_sum, ref["sfc_sum"], "sfc_sum")

    Hsmth = DJ._smooth_free_surface(ref["Hnew"], st.cell_type, st.Nsmooth,
                                    st.Csmooth, st.L, st.W)
    bit_identical(Hsmth, ref["Hsmth"], "Hsmth")
end
