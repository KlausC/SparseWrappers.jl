
import SparseArrays: nzrange, rowvals, nonzeros

nonzeros(S::XSubArray{<:Any,<:Integer,2}) = nonzeros(S.sub.parent)

function nzrange(S::XSubArray{<:Any,<:Integer,2,<:Any,<:Tuple{<:Base.Slice{<:Base.OneTo},<:AbstractVector{<:Integer}},false}, i::Integer)
    nzrange(S.sub, i)
end

function nzrange(S::XSubArray{<:Any,<:Integer,2,<:Any,<:Tuple{<:AbstractVector{<:Integer},<:AbstractVector{<:Integer}},false}, i::Integer)
    A = S.sub.parent
    r = nzrange(A, S.sub.indices[2][i])
    rangestrip(rowvals(A), r, axes(A,1), S)
end

function rangestrip(rvA, r::AbstractVector{Int}, m::AbstractUnitRange, S::XSubArray{<:Any,<:Integer,2,<:Any,<:Tuple{<:AbstractUnitRange,<:Any}})

    r1 = first(r)
    r2 = last(r)
    ri = S.sub.indices[1]
    i1 = first(ri)
    i2 = last(ri)
    if i1 > first(m)
        r1 = searchsortedfirst(rvA, i1, r1, r2, Forward)
    end
    if i2 < last(m)
        r2 = searchsortedlast(rvA, i2, r1, r2, Forward)
    end
    r1:r2
end

function rangestrip(rvA, r::AbstractVector{Int}, m::AbstractUnitRange, S::XSubArray{<:Any,<:Integer,2,<:Any,<:Tuple{<:AbstractVector,<:Any}})
    r1 = first(r)
    r2 = last(r)
    ri = S.sub.indices[1]
    i1, i2 = if ri isa StepRange
        i1 = first(ri)
        i2 = last(ri)
        minmax(i1, i2)
    else
        extrema(ri)
    end
    if i1 > first(m)
        r1 = searchsortedfirst(rvA, i1, r1, r2, Forward)
    end
    if i2 < last(m)
        r2 = searchsortedlast(rvA, i2, r1, r2, Forward)
    end
    if ri isa StepRange
        st = step(ri)
        st == 1 && return r1:r2
        st == -1 && return r2:-1:r1
    end
    vind = rowvals(S)[r1:r2]
    p = sortperm(vind)
    k = searchsortedlast(vind[p], 0)
    (r1:r2)[p][k+1:end]
end

rowvals(xs::XSubArray) = xs.rowval

struct RowIndexVector{U,T,D} <: AbstractVector{Int}
    rvp::T
    ind::U
    dict::D
end

function rowvals(S::SubArray{<:Any,2,<:Any,<:Tuple{<:Base.Slice,<:AbstractVector},false})
    rowvals(S.parent)
end
function rowvals(S::SubArray{<:Any,2,<:Any,<:Tuple{<:UnitRange,<:AbstractVector},false})
    RowIndexVector(rowvals(S.parent), S.indices[1], nothing)
end
function rowvals(S::SubArray{<:Any,2,<:Any,<:Tuple{<:OrdinalRange,<:AbstractVector},false})
    RowIndexVector(rowvals(S.parent), S.indices[1], nothing)
end
function rowvals(S::SubArray{<:Any,2,<:Any,<:Tuple{<:AbstractVector,<:AbstractVector},false})
    ind = S.indices[1]
    RowIndexVector(rowvals(S.parent), ind, Dict(reverse.(enumerate(ind))))
end

import Base: getindex, size
size(rv::RowIndexVector) = size(rv.rvp)
size(s::XSubArray) = size(s.sub)
getindex(s::XSubArray, ind...) = getindex(s.sub, ind...)

function getindex(rv::RowIndexVector{<:UnitRange}, k::Integer)
    i = rv.rvp[k]
    ind = rv.ind
    a = first(ind)
    b = last(ind)
    ( a <= i <= b ) ? ( i - a + 1 ) : 0
end
function getindex(rv::RowIndexVector{<:StepRange}, k::Integer)
    i = rv.rvp[k]
    ind = rv.ind
    a = first(ind)
    b = last(ind)
    a <= i <= b || return 0
    s = step(ind)
    q, r = fldmod(i - a, s)
    ifelse(r == 0, q + 1, 0)
end
function getindex(rv::RowIndexVector{<:AbstractVector}, k::Integer)
    i = rv.rvp[k]
    ind = rv.ind
    a, b = extrema(ind)
    get(rv.dict, i, 0)
end

