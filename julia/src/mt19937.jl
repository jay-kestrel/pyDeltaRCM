"""
    MT19937

Numpy-legacy MT19937 generator, exactly as numba's `@njit` functions see it
(numba embeds numpy's `mt19937_genrand`). Bit-parity contract, verified in
`test/test_mt19937.jl` against pinned vectors:

- `mt_seed!(rng, 0); get_random_uniform(rng, 1.0)` == 0.5488135039273248
- next draw with limit 0.1 == 0.07151893663724194
- after the two draws, `(pos, key[1:4])` == `(4, [2443250962, 1093594115,
  1878467924, 2709361018])` (pos is numba's 0-based word count; a double
  draw consumes two words)

`uniform(0, limit)` is `limit * next_double()` where `next_double` takes two
words: `((a >> 5) * 2^27 + (b >> 6)) / 2^53`.
"""

mutable struct MT19937
    key::Vector{UInt32}   # 624 words
    pos::Int              # numba/numpy convention: 0-based consumed count
end

MT19937() = MT19937(zeros(UInt32, 624), 624)

const _MT_N = 624
const _MT_M = 397
const _MATRIX_A = 0x9908b0df
const _UPPER = 0x80000000
const _LOWER = 0x7fffffff

"""`set_random_seed` equivalent (numpy `mt19937_seed`)."""
function mt_seed!(rng::MT19937, seed::Integer)
    seed < 0 && throw(ArgumentError("seed must be non-negative"))
    rng.key[1] = UInt32(seed)
    for i in 2:_MT_N
        prev = rng.key[i - 1]
        # numpy's C loop adds the 0-based index i (Julia index i -> C i-1)
        rng.key[i] = 0x6c078965 * (prev ⊻ (prev >> 30)) + UInt32(i - 1)
    end
    rng.pos = _MT_N
    return rng
end

"""In-place state twist, replicating numpy's two-loop `mt19937_genrand`."""
function _mt_twist!(rng::MT19937)
    key = rng.key
    for i in 1:(_MT_N - _MT_M)
        y = (key[i] & _UPPER) | (key[i + 1] & _LOWER)
        key[i] = key[i + _MT_M] ⊻ (y >> 1) ⊻ (iszero(y & 0x1) ? 0x00000000 : _MATRIX_A)
    end
    for i in (_MT_N - _MT_M + 1):(_MT_N - 1)
        y = (key[i] & _UPPER) | (key[i + 1] & _LOWER)
        key[i] = key[i + (_MT_M - _MT_N)] ⊻ (y >> 1) ⊻ (iszero(y & 0x1) ? 0x00000000 : _MATRIX_A)
    end
    y = (key[_MT_N] & _UPPER) | (key[1] & _LOWER)
    key[_MT_N] = key[_MT_M] ⊻ (y >> 1) ⊻ (iszero(y & 0x1) ? 0x00000000 : _MATRIX_A)
    rng.pos = 0
    return rng
end

"""One tempered UInt32 word (`rk_random`)."""
function mt_next_word!(rng::MT19937)::UInt32
    rng.pos >= _MT_N && _mt_twist!(rng)
    y = rng.key[rng.pos + 1]
    rng.pos += 1
    y ⊻= y >> 11
    y ⊻= (y << 7) & 0x9d2c5680
    y ⊻= (y << 15) & 0xefc60000
    y ⊻= y >> 18
    return y
end

"""`rk_double`: two words per draw, `(a >> 5) * 2^27 + (b >> 6)` over `2^53`."""
function mt_next_double(rng::MT19937)::Float64
    a = UInt64(mt_next_word!(rng) >> 5)
    b = UInt64(mt_next_word!(rng) >> 6)
    return (a * 67108864.0 + b) / 9007199254740992.0
end

"""
    pack_rng_state(rng) -> (index, key)

Matches `pyDeltaRCM.shared_tools.pack_rng_state`: the 0-based word count and
the full 624-word key array (checkpoint/RNG-state dump format).
"""
pack_rng_state(rng::MT19937) = (rng.pos, copy(rng.key))

"""
    set_rng_state!(rng, index, key)

Restore a packed state (dumped `rng_state_index` / `rng_state_key`).
"""
function set_rng_state!(rng::MT19937, index::Integer, key::AbstractVector{UInt32})
    length(key) == _MT_N || throw(ArgumentError("MT19937 key must have 624 words"))
    0 <= index <= _MT_N || throw(ArgumentError("MT19937 index out of range"))
    rng.key .= key
    rng.pos = Int(index)
    return rng
end
