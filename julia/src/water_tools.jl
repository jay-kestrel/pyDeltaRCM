# Port of pyDeltaRCM/water_tools.py: the water-routing vertical slice.
#
# Every function mirrors its Python counterpart (phase methods drop the `!`
# in Python and gain it here for mutation). Index conventions are documented
# in the module docstring of DeltaRCMJulia.

"""
    WaterState

Mutable model state for the water phase, hydrated from a fixture dump
(post-init arrays + derived scalars). Field arrays are `(W, L)` in the Julia
convention (`M[j, i] == A[i0, j0]`); parcel index arrays keep Python's
0-based flat integers with the `0` inactive sentinel.
"""
Base.@kwdef mutable struct WaterState
    # domain shape
    L::Int
    W::Int
    # domain geometry / BC scalars
    L0::Int
    CTR::Int
    itermax::Int
    Np_water::Int
    dx::Float64
    Qp_water::Float64
    stepmax::Int
    H_SL::Float64
    omega_sfc::Float64
    omega_flow::Float64
    omega_flow_iter::Float64
    Nsmooth::Int
    Csmooth::Float64
    u0::Float64
    h0::Float64
    S0::Float64
    gamma::Float64
    theta_water::Float64
    u_max::Float64
    dry_depth::Float64
    qw0::Float64
    time_iter::Int = 0
    # state fields (float32 in the reference; cell_type int64)
    eta::Matrix{Float32}
    stage::Matrix{Float32}
    depth::Matrix{Float32}
    qw::Matrix{Float32}
    qx::Matrix{Float32}
    qy::Matrix{Float32}
    qwn::Matrix{Float32}
    qxn::Matrix{Float32}
    qyn::Matrix{Float32}
    uw::Matrix{Float32}
    ux::Matrix{Float32}
    uy::Matrix{Float32}
    cell_type::Matrix{Int64}
    mod_water_weight::Matrix{Float32}
    inlet::Vector{Int64}
    # flattened D8 walk arrays (9-vectors, Python C-order (3, 3) flatten)
    iwalk_flat::Vector{Int64}
    jwalk_flat::Vector{Int64}
    ravel_walk_flat::Vector{Int64}
    ivec_flat::Vector{Float32}
    jvec_flat::Vector{Float32}
    distances_flat::Vector{Float32}
    # scratch, refilled per water iteration (zeros at init per the fixture
    # manifest; sfc_visit/sfc_sum are float64 because the numba accumulator
    # allocates np.zeros -> float64); free_surf_walk_inds starts at the
    # fixture's size_indices columns
    free_surf_flag::Vector{Int64} = zeros(Int64, Np_water)
    free_surf_walk_inds::Matrix{Int64}
    sfc_visit::Matrix{Float64} = zeros(Float64, W, L)
    sfc_sum::Matrix{Float64} = zeros(Float64, W, L)
    Hnew::Matrix{Float32} = zeros(Float32, W, L)
    water_weights::Array{Float64,3} = zeros(Float64, 9, W, L)
    start_indices::Vector{Int64} = zeros(Int64, Np_water)
    pad_stage::Matrix{Float32} = zeros(Float32, W + 2, L + 2)
    pad_depth::Matrix{Float32} = zeros(Float32, W + 2, L + 2)
    pad_cell_type::Matrix{Int64} = zeros(Int64, W + 2, L + 2)
end

# ---------------------------------------------------------------------------
# kernels (port of the @njit functions; promotion semantics pinned by probe)
# ---------------------------------------------------------------------------

"""`_get_weight_at_cell_water`: one cell's 9-neighbor water weights."""
function _get_weight_at_cell_water(i0::Int, weight_sfc::Vector{Float32},
                                   weight_int::Vector{Float32},
                                   depth_nbrs::Vector{Float32},
                                   mod_water_weight::Vector{Float32},
                                   ct_nbrs::Vector{Int64},
                                   dry_depth::Float64, gamma::Float64,
                                   theta::Float64)
    dry = depth_nbrs .< dry_depth      # numba promotes to float64 compare
    wall = ct_nbrs .== -2
    ctr = [d0 == 4 for d0 in 0:8]      # 0-based flat position of the center
    drywall = dry .| wall
    invalid = wall .| ctr
    if i0 == 0                          # top edge row: no N-row neighbors
        invalid[1:3] .= true
    end

    weight_sfc[drywall] .= 0.0f0
    weight_int[drywall] .= 0.0f0
    weight_sfc[ctr] .= 0.0f0
    weight_int[ctr] .= 0.0f0

    (any(isnan, weight_sfc) || any(isnan, weight_int)) &&
        error("NaN encountered in input to water weighting. Please report error.")

    # initial rebalance: float32 sums accumulate in float32 and the division
    # stays float32 (numba: f32 array / f32 scalar -> f32 array); the blend
    # below then widens to float64
    s_sfc = pysum(weight_sfc)
    ws = s_sfc > 0 ? weight_sfc ./ Float32(s_sfc) : Float64.(weight_sfc)
    s_int = pysum(weight_int)
    wi = s_int > 0 ? weight_int ./ Float32(s_int) : Float64.(weight_int)

    weight = gamma .* ws .+ (1 - gamma) .* wi
    scale = Float64.(depth_nbrs .* mod_water_weight) .^ theta
    weight = scale .* weight
    weight[ctr] .= 0.0

    s = pysum(weight)
    if s > 0
        weight = weight ./ s
    elseif s == 0
        if wall[5]                     # center cell (d0 == 4) is a wall
            weight[:] .= 0.0
        else
            wetvalid = (!).(dry) .& (!).(invalid)
            nwetvalid = count(wetvalid)
            if nwetvalid > 0
                weight[:] .= 0.0
                weight[wetvalid] .= 1 / nwetvalid
            else
                notwallvalid = (!).(wall) .& (!).(invalid)
                nnotwall = count(notwallvalid)
                weight[:] .= 0.0
                if nnotwall > 0
                    weight[notwallvalid] .= 1 / nnotwall
                else
                    error("No non-wall cells surrounding cell." *
                          "Please report error including logfile.")
                end
            end
            weight = weight ./ pysum(weight)
        end
    else
        error("Water sum(weight) less than 0. This is common when the value " *
              "of gamma is too large. See documentation for complete explanation.")
    end

    any(isnan, weight) &&
        error("NaN encountered in return from water weighting. Please report error.")
    return weight
end

"""`_get_water_weight_array`: full-domain water weights, `(9, W, L)` float64."""
function _get_water_weight_array(depth::Matrix{Float32}, stage::Matrix{Float32},
                                 mod_water_weight::Matrix{Float32},
                                 cell_type::Matrix{Int64}, qx::Matrix{Float32},
                                 qy::Matrix{Float32}, ivec_flat::Vector{Float32},
                                 jvec_flat::Vector{Float32},
                                 distances_flat::Vector{Float32},
                                 dry_depth::Float64, gamma::Float64,
                                 theta_water::Float64)
    L, W = size(depth, 2), size(depth, 1)
    pad_stage = custom_pad(stage)
    pad_depth = custom_pad(depth)
    pad_mod = custom_pad(mod_water_weight)
    pad_ct = custom_pad(cell_type)
    water_weights = zeros(Float64, 9, W, L)
    for i0 in 0:(L - 1), j0 in 0:(W - 1)
        stage_nbrs = Vector{Float32}(undef, 9)
        depth_nbrs = Vector{Float32}(undef, 9)
        mod_nbrs = Vector{Float32}(undef, 9)
        ct_nbrs = Vector{Int64}(undef, 9)
        for d0 in 0:8
            di = d0 ÷ 3 - 1
            dj = d0 % 3 - 1
            jj = j0 + dj + 2   # padded-array coordinates (1-based)
            ii = i0 + di + 2
            stage_nbrs[d0 + 1] = pad_stage[jj, ii]
            depth_nbrs[d0 + 1] = pad_depth[jj, ii]
            mod_nbrs[d0 + 1] = pad_mod[jj, ii]
            ct_nbrs[d0 + 1] = pad_ct[jj, ii]
        end
        weight_sfc, weight_int = get_weight_sfc_int(
            stage[j0 + 1, i0 + 1], stage_nbrs, qx[j0 + 1, i0 + 1],
            qy[j0 + 1, i0 + 1], ivec_flat, jvec_flat, distances_flat)
        water_weights[:, j0 + 1, i0 + 1] .= _get_weight_at_cell_water(
            i0, weight_sfc, weight_int, depth_nbrs, mod_nbrs, ct_nbrs,
            dry_depth, gamma, theta_water)
    end
    return water_weights
end

"""`_choose_next_directions`: D8 pick per active parcel (0-based codes)."""
function _choose_next_directions(rng, inds::Vector{Int64},
                                 water_weights::Array{Float64,3}, L::Int, W::Int)
    next_direction = zeros(Int64, length(inds))
    for p in eachindex(inds)
        f0 = inds[p]
        if f0 != 0
            weight = vec(water_weights[:, (f0 % W) + 1, (f0 ÷ W) + 1])
            next_direction[p] = random_pick(rng, weight)
        else
            next_direction[p] = 4  # inactive parcels "stay"
        end
    end
    return next_direction
end

"""`_calculate_new_inds`: flat-index step; `newd == 4` pins the sentinel 0."""
function _calculate_new_inds(current_inds::Vector{Int64},
                             new_direction::Vector{Int64},
                             ravel_walk::Vector{Int64})
    new_inds = zeros(Int64, length(current_inds))
    for p in eachindex(current_inds)
        newd = new_direction[p]
        if newd != 4
            new_inds[p] = current_inds[p] + ravel_walk[newd + 1]
        else
            new_inds[p] = 0
        end
    end
    return new_inds
end

"""
    _inject_inlet_discharge!(qxn, qyn, qwn, start_indices, Qp_water, dx, W)

`_inject_inlet_discharge`: duplicate start indices accumulate once per
parcel. Augmentation through the flat field is a float32 store of a float64
sum (double rounding), matching numpy's `f32 += f64` behavior.
"""
function _inject_inlet_discharge!(qxn::Matrix{Float32}, qyn::Matrix{Float32},
                                  qwn::Matrix{Float32},
                                  start_indices::Vector{Int64},
                                  Qp_water::Float64, dx::Float64, W::Int)
    # np.unique(..., return_counts=True): sorted unique values + counts
    sorted = sort(start_indices)
    uniq = Int64[]
    counts = Int64[]
    for v in sorted
        if !isempty(uniq) && uniq[end] == v
            counts[end] += 1
        else
            push!(uniq, v)
            push!(counts, 1)
        end
    end
    for (u0, c) in zip(uniq, counts)
        k = (u0 % W) + 1 + W * (u0 ÷ W)  # == vec-index of flat u0
        qxn[k] = Float32(Float64(qxn[k]) + Float64(c))
        qwn[k] = Float32(Float64(qwn[k]) + Float64(c) * Qp_water / dx / 2)
    end
    for f0 in start_indices
        k = (f0 % W) + 1 + W * (f0 ÷ W)
        qyn[k] = Float32(Float64(qyn[k]) + 0.0)  # no y-directed inlet discharge
    end
    return nothing
end

"""`_check_for_loops`: relocate looping parcels along the mean-transport
vector; `looped` disqualifies a parcel from the free surface."""
function _check_for_loops(free_surf_walk_inds::Matrix{Int64},
                          new_inds::Vector{Int64}, _step::Int, L0::Int, CTR::Int,
                          stage_above_SL::Matrix{Float32}, L::Int, W::Int)
    nparcels = size(free_surf_walk_inds, 2)
    domain_min_x = L - 2
    domain_min_y = W - 2
    L0_ind_cut = L0 * W - 1

    looped = zeros(Int64, nparcels)
    # numba compares float32 values promoted to float64 against 1e-1
    stage_v_SL = Float64.(abs.(stage_above_SL)) .< 1e-1

    if _step > L0
        for p in 1:nparcels
            new_ind = new_inds[p]
            full_walk = free_surf_walk_inds[:, p]
            nonz_walk = full_walk[full_walk .> 0]
            relv_walk = nonz_walk[nonz_walk .> L0_ind_cut]

            if new_ind > 0
                has_repeat_ind = any(==(new_ind), relv_walk)
                if has_repeat_ind
                    px0, py0 = custom_unravel(new_ind, L, W)
                    Fx = px0 - 1
                    Fy = py0 - CTR
                    Fw = sqrt(Float64(Fx^2 + Fy^2))
                    if Fw != 0
                        px = px0 + round(Int, Fx / Fw * 5.0)
                        py = py0 + round(Int, Fy / Fw * 5.0)
                    else
                        # degenerate transport vector: leave in place
                        px, py = px0, py0
                    end
                    px = min(domain_min_x, max(px, L0))
                    py = min(domain_min_y, max(py, 1))
                    new_inds[p] = custom_ravel(px, py, L, W)
                    if !stage_v_SL[py0 + 1, px0 + 1]
                        looped[p] = 1
                    end
                end
            end
        end
    end
    return new_inds, looped
end

"""`_update_dirQfield`: flat float32 field, float64 delta, double-rounded
store — numba's `qfield[ii] += dirstep[i] / dist[i]`."""
function _update_dirQfield!(qfield::Vector{Float32}, dist::Vector{Float64},
                            inds::Vector{Int64}, astep::BitVector,
                            dirstep::Vector{Int64})
    for i in eachindex(inds)
        if astep[i]
            k = inds[i] + 1
            qfield[k] = Float32(Float64(qfield[k]) + Float64(dirstep[i]) / dist[i])
        end
    end
    return qfield
end

"""`_update_absQfield`: same store semantics with the parcel discharge."""
function _update_absQfield!(qfield::Vector{Float32}, dist::Vector{Float64},
                            inds::Vector{Int64}, astep::BitVector,
                            Qp_water::Float64, dx::Float64)
    for i in eachindex(inds)
        if astep[i]
            k = inds[i] + 1
            qfield[k] = Float32(Float64(qfield[k]) + Qp_water / dx / 2)
        end
    end
    return qfield
end

"""`_accumulate_free_surface_walks`: backward walks -> `sfc_visit`, `sfc_sum`
(float64 throughout, as numba's `np.zeros` allocates float64)."""
function _accumulate_free_surface_walks(free_surf_walk_inds::Matrix{Int64},
                                        free_surf_flag::Vector{Int64},
                                        cell_type::Matrix{Int64},
                                        uw::Matrix{Float32}, ux::Matrix{Float32},
                                        uy::Matrix{Float32},
                                        depth::Matrix{Float32}, dx::Float64,
                                        u0::Float64, h0::Float64, H_SL::Float64,
                                        S0::Float64, L::Int, W::Int)
    Hnew = zeros(Float64, W, L)
    sfc_visit = zeros(Float64, W, L)
    sfc_sum = zeros(Float64, W, L)

    for p in 1:size(free_surf_walk_inds, 2)
        inds = free_surf_walk_inds[:, p]
        inds_whr = inds[inds .> 0]
        n = length(inds_whr)
        xs = Vector{Int64}(undef, n)
        ys = Vector{Int64}(undef, n)
        for k in 1:n
            xs[k], ys[k] = custom_unravel(inds_whr[k], L, W)
        end

        fill!(Hnew, 0.0)
        if (cell_type[ys[end] + 1, xs[end] + 1] == -1) &&
           (free_surf_flag[p] == 1)
            Hnew[ys[end] + 1, xs[end] + 1] = H_SL

            in_ocean = true
            dH = 0.0
            for it in (n - 1):-1:1
                i = xs[it]; ip = xs[it + 1]
                j = ys[it]; jp = ys[it + 1]
                if (i != ip) || (j != jp)
                    if in_ocean
                        if (Float64(uw[j + 1, i + 1]) > u0 * 0.5) ||
                           (Float64(depth[j + 1, i + 1]) < 0.1 * h0)
                            in_ocean = false
                        end
                    else
                        if Float64(uw[j + 1, i + 1]) == 0
                            dH = 0.0
                        else
                            dH = S0 * (Float64(ux[j + 1, i + 1]) * (ip - i) * dx +
                                       Float64(uy[j + 1, i + 1]) * (jp - j) * dx) /
                                 Float64(uw[j + 1, i + 1])
                        end
                    end
                end
                Hnew[j + 1, i + 1] = Hnew[jp + 1, ip + 1] + dH
                sfc_visit[j + 1, i + 1] = sfc_visit[j + 1, i + 1] + 1
                sfc_sum[j + 1, i + 1] = sfc_sum[j + 1, i + 1] + Hnew[j + 1, i + 1]
            end
        end
    end
    return sfc_visit, sfc_sum
end

"""`_smooth_free_surface`: `Nsmooth` underrelaxed neighbor-average passes;
the per-cell update stores a float64 result into the float32 array."""
function _smooth_free_surface(Hin::Matrix{Float32}, cell_type::Matrix{Int64},
                              Nsmooth::Int, Csmooth::Float64, L::Int, W::Int)
    cell_type_pad = custom_pad(cell_type)
    Htemp = copy(Hin)
    for _ in 1:Nsmooth
        Hsmth = copy(Htemp)
        Hsmth_pad = custom_pad(Hsmth)
        for i0 in 0:(L - 1), j0 in 0:(W - 1)
            if cell_type[j0 + 1, i0 + 1] != -1
                Hsmth_nbrs = Vector{Float32}(undef, 9)
                ct_nbrs = Vector{Int64}(undef, 9)
                for d0 in 0:8
                    di = d0 ÷ 3 - 1
                    dj = d0 % 3 - 1
                    jj = j0 + dj + 2
                    ii = i0 + di + 2
                    Hsmth_nbrs[d0 + 1] = Hsmth_pad[jj, ii]
                    ct_nbrs[d0 + 1] = cell_type_pad[jj, ii]
                end
                invalid_nbrs = ct_nbrs .== -2
                invalid_nbrs[5] = true
                Hsmth_nbrs[invalid_nbrs] .= 0.0f0
                sumH = pysum(Hsmth_nbrs)
                nbcount = count(.!invalid_nbrs)
                if nbcount > 0
                    Htemp[j0 + 1, i0 + 1] =
                        Float32(Csmooth * Float64(Hsmth[j0 + 1, i0 + 1]) +
                                (1 - Csmooth) * (sumH / nbcount))
                end
            end
        end
    end
    return Htemp
end

# ---------------------------------------------------------------------------
# phase methods (port of the mixin methods on WaterState)
# ---------------------------------------------------------------------------

"""`get_water_weight_array`: compute and store `water_weights`."""
function get_water_weight_array!(st::WaterState)
    st.water_weights = _get_water_weight_array(
        st.depth, st.stage, st.mod_water_weight, st.cell_type, st.qx, st.qy,
        st.ivec_flat, st.jvec_flat, st.distances_flat, st.dry_depth,
        st.gamma, st.theta_water)
    return st
end

"""`check_size_of_indices_matrix!`: grow the walk matrix by `stepmax/4`."""
function check_size_of_indices_matrix!(st::WaterState, it::Int)
    if it >= size(st.free_surf_walk_inds, 1)
        blanks = zeros(Int64, st.stepmax ÷ 4, st.Np_water)
        st.free_surf_walk_inds = vcat(st.free_surf_walk_inds, blanks)
    end
    return st
end

"""`check_for_boundary`: parcels sitting on an edge cell (cell_type == -1);
the sentinel `f0 == 0` reads cell `(0, 0)` exactly like the Python fancy
index `cell_type.flat[inds]`."""
function check_for_boundary(st::WaterState, inds::Vector{Int64})
    W = st.W
    return BitVector(st.cell_type[(f0 % W) + 1, (f0 ÷ W) + 1] == -1 for f0 in inds)
end

"""`update_Q`: discharge updates for the current and/or next parcel cells.
Python rebinds `qxn/qyn/qwn` from the flat kernel result; so does this."""
function update_Q!(st::WaterState, dist::Vector{Float64}, current_inds::Vector{Int64},
                   next_inds::Vector{Int64}, astep::BitVector, jstep::Vector{Int64},
                   istep::Vector{Int64}; update_current::Bool=false,
                   update_next::Bool=false)
    if update_current
        st.qxn = _update_dirQfield!(vec(st.qxn), dist, current_inds, astep, jstep) |>
                 v -> reshape(v, st.W, st.L)
        st.qyn = _update_dirQfield!(vec(st.qyn), dist, current_inds, astep, istep) |>
                 v -> reshape(v, st.W, st.L)
        st.qwn = _update_absQfield!(vec(st.qwn), dist, current_inds, astep,
                                    st.Qp_water, st.dx) |>
                 v -> reshape(v, st.W, st.L)
    end
    if update_next
        st.qxn = _update_dirQfield!(vec(st.qxn), dist, next_inds, astep, jstep) |>
                 v -> reshape(v, st.W, st.L)
        st.qyn = _update_dirQfield!(vec(st.qyn), dist, next_inds, astep, istep) |>
                 v -> reshape(v, st.W, st.L)
        st.qwn = _update_absQfield!(vec(st.qwn), dist, next_inds, astep,
                                    st.Qp_water, st.dx) |>
                 v -> reshape(v, st.W, st.L)
    end
    return st
end

"""`init_water_iteration!`: zero fluxes, reset trackers, seed parcel starts."""
function init_water_iteration!(st::WaterState, rng)
    fill!(st.qxn, 0.0f0)
    fill!(st.qyn, 0.0f0)
    fill!(st.qwn, 0.0f0)

    fill!(st.free_surf_flag, 1)
    fill!(st.free_surf_walk_inds, 0)
    fill!(st.sfc_visit, 0.0)
    fill!(st.sfc_sum, 0.0)

    st.pad_stage = custom_pad(st.stage)
    st.pad_depth = custom_pad(st.depth)
    st.pad_cell_type = custom_pad(st.cell_type)

    inlet_weights = get_inlet_weights(st.inlet)
    st.start_indices = get_start_indices(rng, st.inlet, inlet_weights, st.Np_water)
    return st
end

"""`run_water_iteration!`: lockstep parcel stepping with flux deposition."""
function run_water_iteration!(st::WaterState, rng)
    start_indices = st.start_indices
    _step = 0

    _inject_inlet_discharge!(st.qxn, st.qyn, st.qwn, start_indices,
                             st.Qp_water, st.dx, st.W)

    st.free_surf_walk_inds[1, :] = start_indices
    current_inds = copy(start_indices)

    get_water_weight_array!(st)

    while sum(current_inds) > 0 && _step < st.stepmax
        _step += 1
        check_size_of_indices_matrix!(st, _step)

        new_direction = _choose_next_directions(rng, current_inds,
                                                st.water_weights, st.L, st.W)
        new_inds = _calculate_new_inds(current_inds, new_direction,
                                       st.ravel_walk_flat)
        dist, istep, jstep, astep = get_steps(new_direction, st.iwalk_flat,
                                              st.jwalk_flat)

        update_Q!(st, dist, current_inds, new_inds, astep, jstep, istep;
                  update_current=true, update_next=true)

        # Python: self.stage - self.H_SL with H_SL a float property -> the
        # numpy weak-scalar rule keeps the result float32
        stage_above_SL = st.stage .- Float32(st.H_SL)
        new_inds, looped = _check_for_loops(st.free_surf_walk_inds, new_inds,
                                            _step, st.L0, st.CTR,
                                            stage_above_SL, st.L, st.W)
        looped_b = looped .!= 0

        current_inds .= new_inds
        st.free_surf_flag[looped_b] .= 0

        boundary = check_for_boundary(st, current_inds)
        boundary_looped = looped_b .& boundary
        current_inds[boundary_looped] .= 0

        st.free_surf_walk_inds[_step + 1, :] = current_inds
        current_inds[boundary] .= 0

        if any(boundary)
            update_Q!(st, dist[boundary], current_inds[boundary],
                      new_inds[boundary], astep[boundary], jstep[boundary],
                      istep[boundary]; update_current=false, update_next=true)
        end
    end
    return st
end

"""`compute_free_surface!`: accumulate walks into a new stage surface."""
function compute_free_surface!(st::WaterState)
    st.sfc_visit, st.sfc_sum = _accumulate_free_surface_walks(
        st.free_surf_walk_inds, st.free_surf_flag, st.cell_type, st.uw, st.ux,
        st.uy, st.depth, st.dx, st.u0, st.h0, st.H_SL, st.S0, st.L, st.W)

    Hnew = st.eta .+ st.depth  # float32, as in the reference
    m = st.sfc_visit .> 0
    Hnew[m] .= Float32.(st.sfc_sum[m] ./ st.sfc_visit[m])
    Hnew[Hnew .< Float32(st.H_SL)] .= Float32(st.H_SL)
    m2 = Hnew .< st.eta
    Hnew[m2] .= st.eta[m2]
    st.Hnew = Hnew

    finalize_free_surface!(st)
    return st
end

"""`finalize_free_surface!`: smoothing, underrelaxation, flooding fix."""
function finalize_free_surface!(st::WaterState)
    Hsmth = _smooth_free_surface(st.Hnew, st.cell_type, st.Nsmooth, st.Csmooth,
                                 st.L, st.W)
    if st.time_iter > 0
        # Python: (1 - omega_sfc) * stage + omega_sfc * Hsmth with float32
        # weak-scalar arithmetic on both terms
        st.stage = Float32(1 - st.omega_sfc) .* st.stage .+
                   Float32(st.omega_sfc) .* Hsmth
    end
    flooding_correction!(st)
    return st
end

"""`finalize_water_iteration!`: clamp stage/depth, update flow and velocity."""
function finalize_water_iteration!(st::WaterState, iteration::Int)
    st.stage .= max.(st.stage, Float32(st.H_SL))
    st.depth .= max.(st.stage .- st.eta, 0.0f0)
    update_flow_field!(st, iteration)
    update_velocity_field!(st)
    return st
end

"""`update_flow_field!`: normalize the directional fluxes and blend."""
function update_flow_field!(st::WaterState, iteration::Int)
    # Python computes ** 0.5 in float32 (weak scalar); sqrt is the exactly
    # rounded form of the same power
    dloc = sqrt.((st.qxn .^ 2.0f0) .+ (st.qyn .^ 2.0f0))

    qwn_div = ones(Float64, st.W, st.L)  # reference quirk: float64 divider
    m = dloc .> 0.0f0
    qwn_div[m] .= st.qwn[m] ./ dloc[m]

    st.qxn .= Float32.(qwn_div .* st.qxn)  # f64 math, f32 store (double round)
    st.qyn .= Float32.(qwn_div .* st.qyn)

    if st.time_iter > 0
        omega = st.omega_flow_iter
        if iteration == 0
            omega = st.omega_flow
        end
        st.qx = Float32(omega) .* st.qxn .+ Float32(1 - omega) .* st.qx
        st.qy = Float32(omega) .* st.qyn .+ Float32(1 - omega) .* st.qy
    else
        st.qx = copy(st.qxn)
        st.qy = copy(st.qyn)
    end

    st.qw = sqrt.((st.qx .^ 2.0f0) .+ (st.qy .^ 2.0f0))

    inlet_rows = st.inlet .+ 1  # python `self.qx[0, self.inlet]`: row i=0,
    # columns j=inlet -> (W, L) layout is M[inlet+1, 1]
    st.qx[inlet_rows, 1] .= Float32(st.qw0)
    st.qy[inlet_rows, 1] .= 0.0f0
    st.qw[inlet_rows, 1] .= Float32(st.qw0)
    return st
end

"""`update_velocity_field!`: velocity from discharge; dry cells zeroed."""
function update_velocity_field!(st::WaterState)
    mask = (st.depth .> Float32(st.dry_depth)) .& (st.qw .> 0.0f0)
    st.uw[mask] .= min.(Float32(st.u_max), st.qw[mask] ./ st.depth[mask])
    st.ux[mask] .= st.uw[mask] .* st.qx[mask] ./ st.qw[mask]
    st.uy[mask] .= st.uw[mask] .* st.qy[mask] ./ st.qw[mask]
    st.uw[.!mask] .= 0.0f0
    st.ux[.!mask] .= 0.0f0
    st.uy[.!mask] .= 0.0f0
    return st
end

"""`build_weight_array`: `(8, W, L)` float64 neighbor stack (E, NE, N, NW,
W, SW, S, SE) with the reference's edge replication."""
function build_weight_array(array::Matrix{T}, fix_edges::Bool=false) where {T}
    W, L = size(array)
    wgt = zeros(Float64, 8, W, L)
    for i0 in 0:(L - 1), j0 in 0:(W - 1)
        # neighbor source cells in 0-based (x, y); zero when out of bounds
        for (n, (si, sj)) in enumerate(((0, 1), (-1, 1), (-1, 0), (-1, -1),
                                        (0, -1), (1, -1), (1, 0), (1, 1)))
            ii, jj = i0 + si, j0 + sj
            if 0 <= ii < L && 0 <= jj < W
                wgt[n, j0 + 1, i0 + 1] = Float64(array[jj + 1, ii + 1])
            end
        end
    end
    if fix_edges
        # replicate the Python assignment sequence exactly: later copies read
        # values already written by earlier ones (e.g. NE row 0 at j = W-1)
        for i in 1:L
            wgt[1, W, i] = wgt[1, W - 1, i]   # E   col -1 <- col -2
            wgt[2, W, i] = wgt[2, W - 1, i]   # NE  col -1
            wgt[8, W, i] = wgt[8, W - 1, i]   # SE  col -1
        end
        for j in 1:W
            wgt[2, j, 1] = wgt[2, j, 2]       # NE  row 0 <- row 1
            wgt[3, j, 1] = wgt[3, j, 2]       # N   row 0
            wgt[4, j, 1] = wgt[4, j, 2]       # NW  row 0
        end
        for i in 1:L
            wgt[4, 1, i] = wgt[4, 2, i]       # NW  col 0 <- col 1
            wgt[5, 1, i] = wgt[5, 2, i]       # W   col 0
            wgt[6, 1, i] = wgt[6, 2, i]       # SW  col 0
        end
        for j in 1:W
            wgt[6, j, L] = wgt[6, j, L - 1]   # SW  row -1 <- row -2
            wgt[7, j, L] = wgt[7, j, L - 1]   # S   row -1
            wgt[8, j, L] = wgt[8, j, L - 1]   # SE  row -1
        end
    end
    return wgt
end

"""`get_wet_mask_nh`: (8, W, L) float64 wet mask of the neighbors."""
function get_wet_mask_nh(st::WaterState)
    wet_mask = st.depth .> Float32(st.dry_depth)
    return build_weight_array(Float64.(wet_mask), true)
end

"""`flooding_correction!`: flood dry shore cells whose wet-neighbor stage
exceeds their bed elevation."""
function flooding_correction!(st::WaterState)
    wet_mask = st.depth .> Float32(st.dry_depth)
    wet_mask_nh = get_wet_mask_nh(st)
    wet_mask_nh_sum = dropdims(sum(wet_mask_nh; dims=1); dims=1)  # (W, L) f64
    wet_mask_nh_sum[wet_mask] .= 0.0

    # stage_nhs is snapshotted once, before any write (as in Python)
    stage_nhs = build_weight_array(st.stage)

    # np.where(sum > 0): all shore cells; writes are independent per cell
    for i0 in 0:(st.L - 1), j0 in 0:(st.W - 1)
        if wet_mask_nh_sum[j0 + 1, i0 + 1] > 0
            wet8 = wet_mask_nh[:, j0 + 1, i0 + 1]
            stage_nh = wet8 .* stage_nhs[:, j0 + 1, i0 + 1]  # f64 * f32 -> f64
            eta_shore = Float64(st.eta[j0 + 1, i0 + 1])
            if any(>(eta_shore), stage_nh)
                st.stage[j0 + 1, i0 + 1] = Float32(maximum(stage_nh))
            end
        end
    end
    return st
end

"""`route_water!`: the full water phase for one model update."""
function route_water!(st::WaterState, rng)
    for iteration in 0:(st.itermax - 1)
        init_water_iteration!(st, rng)
        run_water_iteration!(st, rng)
        compute_free_surface!(st)
        finalize_water_iteration!(st, iteration)
    end
    return st
end
