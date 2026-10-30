"""
Minimal `.npy` reader for the reference fixtures (v1.0/v2.0 headers).

Supports the dtypes the fixture set actually uses and C-order arrays only
(`fortran_order` is rejected; every dumper writes C-order). Returned arrays
follow the module's transpose convention: a NumPy `(L, W)` C-order array is
returned as a Julia `(W, L)` `Matrix` with byte-identical memory order, so
`reshape(reinterpret(T, bytes), reverse(dims)...)` is exact for every rank.
Zero-dimensional files (e.g. the packed RNG state index) return the scalar.
"""

const _NPY_DTYPES = Dict(
    "<f4" => Float32,
    "<f8" => Float64,
    "<i8" => Int64,
    "<i4" => Int32,
    "<u4" => UInt32,
    "|b1" => Bool,
    "|u1" => UInt8,
)

function _npy_header(path::AbstractString)
    raw = read(path)
    length(raw) >= 10 || error("truncated npy file: $path")
    raw[1:6] == UInt8[0x93, UInt8('N'), UInt8('U'), UInt8('M'), UInt8('P'), UInt8('Y')] ||
        error("not an npy file: $path")
    major = Int(raw[7])
    hlen, off = if major == 1
        Int(UInt16(raw[9]) | (UInt16(raw[10]) << 8)), 10
    else
        Int(UInt32(raw[9]) | (UInt32(raw[10]) << 8) | (UInt32(raw[11]) << 16) |
            (UInt32(raw[12]) << 24)), 12
    end
    header = String(raw[(off + 1):(off + hlen)])
    return header, off + hlen
end

_parse_descr(header) = match(r"'descr':\s*'([^']+)'", header).captures[1]

function _parse_dims(header)
    m = match(r"'shape':\s*\(([^)]*)\)", header).captures[1]
    return [parse(Int, strip(s)) for s in split(m, ",") if !isempty(strip(s))]
end

"""
    npyread(path) -> Array or scalar

Read a `.npy` file with the module's transpose convention (NumPy `(L, W)` ->
Julia `(W, L)`; `(L, W, K)` -> `(K, W, L)`; `(Np, C)` -> `(C, Np)`).
"""
function npyread(path::AbstractString)
    header, dataoff = _npy_header(path)
    descr = _parse_descr(header)
    haskey(_NPY_DTYPES, descr) || error("unsupported npy dtype in $path: $descr")
    T = _NPY_DTYPES[descr]
    occursin("'fortran_order': True", header) &&
        error("fortran-order npy not supported: $path")
    dims = _parse_dims(header)
    raw = read(path)
    nbytes = length(raw) - dataoff
    if isempty(dims)  # 0-d scalar
        return reinterpret(T, raw[(dataoff + 1):(dataoff + nbytes)])[1]
    end
    v = reinterpret(T, view(raw, (dataoff + 1):length(raw)))
    return copy(reshape(v, reverse(dims)...))  # plain Array, standard strides
end
