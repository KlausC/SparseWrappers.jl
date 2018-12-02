
import SparseArrays: nzrange, rowvals, nonzeros, SparseMatrixCSCView

sparseaccess(A::SparseMatrixCSC) = A
sparseaccess(A::SubArray{<:Any,2,<:SparseMatrixCSC{<:Any,<:Any},<:Tuple{I,<:Any},false}) where
    I<:Union{Base.Slice{<:Base.OneTo},AbstractUnitRange} = A

sparseaccess(A::SubArray) = XSubArray(A)

nonzeros(S::SubArray) = nonzeros(S.parent)
nonzeros(S::XSubArray{<:Any,<:Integer,2}) = nonzeros(S.sub.parent)

function nzrange(S::SubArray{<:Any,2,<:SparseMatrixCSC,<:Tuple{I,J},false}, i::Integer) where {I<:AbstractUnitRange,J<:AbstractVector{<:Integer}}
    A = S.parent
    r = nzrange(A, S.indices[2][i])
    rvA = rowvals(A)
    m = axes(A, 1)
    r1 = first(r)
    r2 = last(r)
    ri = S.indices[1]
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

function nzrange(xS::XSubArray{<:Any,<:Integer,2,<:SparseMatrixCSC,<:Tuple{I,J},false}, i::Integer) where {I<:AbstractVector{<:Integer},J<:AbstractVector{<:Integer}}
    S = xS.sub
    A = S.parent
    r = nzrange(A, S.indices[2][i])
    rvA = rowvals(A)
    m = axes(A, 1)
    rangestrip(rvA, r, m, xS)
end

function rangestrip(rvA, r::AbstractVector{Int}, m::AbstractUnitRange, S::XSubArray{<:Any,<:Integer,2,<:Any,<:Tuple{I,<:Any}}) where I<:StepRange
    r1 = first(r)
    r2 = last(r)
    ri = S.sub.indices[1]
    i1 = first(ri)
    i2 = last(ri)
    minmax(i1, i2)
    if i1 > first(m)
        r1 = searchsortedfirst(rvA, i1, r1, r2, Forward)
    end
    if i2 < last(m)
        r2 = searchsortedlast(rvA, i2, r1, r2, Forward)
    end
    nzr = Vector{eltype(ri)}(undef, r2 - r1 + 1)
    rv = rowvals(S)
    k = 0
    for i in r1:r2
        j = rv[i]
        if j != 0
            nzr[k+=1] = i
        end
    end
    resize!(nzr, k)
    step(ri) < 0 ? reverse(nzr) : nzr
end

function rangestrip(rvA, r::AbstractVector{Int}, m::AbstractUnitRange, S::XSubArray{<:Any,<:Integer,2,<:Any,<:Tuple{I,<:Any}}) where I<:AbstractVector{<:Integer}
    r1 = first(r)
    r2 = last(r)
    ri = S.sub.indices[1]
    i1, i2 = extrema(ri)
    if i1 > first(m)
        r1 = searchsortedfirst(rvA, i1, r1, r2, Forward)
    end
    if i2 < last(m)
        r2 = searchsortedlast(rvA, i2, r1, r2, Forward)
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
    i -= a
    rem(i, s) == 0 ? 0 : div(i, s)
end
function getindex(rv::RowIndexVector{<:AbstractVector}, k::Integer)
    i = rv.rvp[k]
    ind = rv.ind
    a, b = extrema(ind)
    get(rv.dict, i, 0)
end

SparseArrays.nzrange(X::SparseVector, i) = 1:length(X.nzind)
SparseArrays.rowvals(X::SparseVector) = X.nzind

