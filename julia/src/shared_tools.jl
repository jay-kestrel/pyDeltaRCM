# Port of pyDeltaRCM/shared_tools.py helpers used by the water phase.
# All stochastic functions take an explicit RNG object (MT19937, or any type
# implementing `get_random_uniform(rng, limit)` for test doubles).

"""
    get_random_uniform(rng, limit) -> Float64

`shared_tools.get_random_uniform`: uniform draw over `[0, limit]`
(numba: `np.random.uniform(0, limit)` = `limit * rk_double`).
"""
get_random_uniform(rng::MT19937, limit::Real) = limit * mt_next_double(rng)

"""
    pysum(v) -> Float64

Python `np.sum` inside numba: sequential left-to-right accumulation in the
array's own dtype (float32 arrays accumulate in float32 — probe-verified),
the scalar then widened to Float64.
"""
function pysum(v::AbstractVector{Float32})
    acc = 0.0f0
    for x in v
        acc += x
    end
    return Float64(acc)
end

function pysum(v::AbstractVector{Float64})
    acc = 0.0
    for x in v
        acc += x
    end
    return acc
end

"""Python `np.cumsum` (sequential, dtype of input)."""
function pycumsum(v::AbstractVector{Float64})
    out = similar(v)
    acc = 0.0
    for i in eachindex(v)
        acc += v[i]
        out[i] = acc
    end
    return out
end

"""
    custom_pad(arr) -> Matrix

`shared_tools.custom_pad`: one-cell edge pad (`np.pad(arr, 1, "edge")`).
A NumPy `(L, W)` array pads to `(L+2, W+2)`; in the Julia convention the
result is `(W+2, L+2)`.
"""
function custom_pad(arr::Matrix{T}) where {T}
    W, L = size(arr)
    pad = zeros(T, W + 2, L + 2)
    # center
    pad[2:(W + 1), 2:(L + 1)] .= arr
    # edges (left/right replicate columns j0=0 and j0=W-1; top/bottom rows)
    for i in 1:L
        pad[1, i + 1] = arr[1, i]          # top    (i0 = -1 row)
        pad[W + 2, i + 1] = arr[W, i]      # bottom
    end
    for j in 1:W
        pad[j + 1, 1] = arr[j, 1]          # left   (j0 = -1 col)
        pad[j + 1, L + 2] = arr[j, L]      # right
    end
    # corners
    pad[1, 1] = arr[1, 1]                  # ul
    pad[1, L + 2] = arr[1, L]              # ur
    pad[W + 2, 1] = arr[W, 1]              # ll
    pad[W + 2, L + 2] = arr[W, L]          # lr
    return pad
end

"""
    custom_unravel(f0, L, W) -> (x, y)

`shared_tools.custom_unravel`: 0-based flat index -> 0-based `(x, y)` with
`y` fastest (C order). Raises like Python when out of bounds.
"""
function custom_unravel(f0::Integer, L::Integer, W::Integer)
    f0 >= L * W && throw(ErrorException("Index is out of array bounds"))
    return f0 ÷ W, f0 % W
end

"""`shared_tools.custom_ravel`: 0-based `(x, y)` -> flat, with bounds check."""
function custom_ravel(px::Integer, py::Integer, L::Integer, W::Integer)
    (px >= L || py >= W) && throw(ErrorException("Index is out of array bounds"))
    return px * W + py
end

"""
    random_pick(rng, prob) -> Int (0-based)

`shared_tools.random_pick`: one draw `u = uniform(0, cumprob[end])`, then
`np.searchsorted(cumprob, u)` (side='left' -> `searchsortedfirst`). The
all-zero/negative/NaN-total case raises (the Python guard at
shared_tools.py, which replaced the silent direction-0 fallback).
"""
function random_pick(rng, prob::Vector{Float64})::Int64
    cumprob = pycumsum(prob)
    total = cumprob[end]
    if !(total > 0)
        throw(DomainError(prob, "prob must contain a positive total probability"))
    end
    u = get_random_uniform(rng, total)
    return searchsortedfirst(cumprob, u) - 1
end

"""
    get_inlet_weights(inlet) -> Vector{Int64}

`shared_tools.get_inlet_weights`: uniform weights over inlet cells.
"""
get_inlet_weights(inlet::Vector{Int64}) = ones(Int64, length(inlet))

"""
    get_start_indices(rng, inlet, inlet_weights, num_starts) -> Vector{Int64}

`shared_tools.get_start_indices`: `num_starts` weighted picks over the inlet
cells (0-based flat domain indices, exactly as the Python dump records them).
"""
function get_start_indices(rng, inlet::Vector{Int64}, inlet_weights::Vector{Int64},
                           num_starts::Integer)
    norm_weights = Float64.(inlet_weights) ./ Float64(sum(inlet_weights))
    idxs = Vector{Int64}(undef, num_starts)
    for p in 1:num_starts
        idxs[p] = random_pick(rng, norm_weights)
    end
    return [inlet[idx + 1] for idx in idxs]
end

"""
    get_steps(new_direction, iwalk_flat, jwalk_flat)

`shared_tools.get_steps`, vectorized over parcels: `(dist, istep, jstep,
astep)`; `dist` is Float64 (`np.sqrt` of integer steps), `astep = dist .!= 0`.
Direction codes are 0-based (index into the flattened (3, 3) walk arrays).
"""
function get_steps(new_direction::Vector{Int64}, iwalk_flat::Vector{Int64},
                   jwalk_flat::Vector{Int64})
    n = length(new_direction)
    dist = Vector{Float64}(undef, n)
    istep = Vector{Int64}(undef, n)
    jstep = Vector{Int64}(undef, n)
    astep = BitVector(undef, n)
    for p in 1:n
        nd = new_direction[p] + 1  # 0-based direction -> Julia index
        istep[p] = iwalk_flat[nd]
        jstep[p] = jwalk_flat[nd]
        dist[p] = sqrt(Float64(istep[p]^2 + jstep[p]^2))
        astep[p] = dist[p] != 0
    end
    return dist, istep, jstep, astep
end

"""
    get_weight_sfc_int(stage, stage_nbrs, qx, qy, ivec, jvec, distances)

`shared_tools.get_weight_sfc_int`, in float32 (numba keeps these arrays
float32; the caller promotes on rebalancing).
"""
function get_weight_sfc_int(stage::Float32, stage_nbrs::Vector{Float32},
                            qx::Float32, qy::Float32, ivec::Vector{Float32},
                            jvec::Vector{Float32}, distances::Vector{Float32})
    weight_sfc = max.(0.0f0, (stage .- stage_nbrs) ./ distances)
    weight_int = max.(0.0f0, (qx .* jvec .+ qy .* ivec) ./ distances)
    return weight_sfc, weight_int
end
