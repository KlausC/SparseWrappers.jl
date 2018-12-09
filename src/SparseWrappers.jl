module SparseWrappers

using LinearAlgebra
using SparseArrays

import LinearAlgebra:   mul!
import SparseArrays:    possible_adjoint, SparseMatrixCSCUnion
import Base.Order: Forward

"""
    Conjugate(A::AbstractMatrix)

Represent the elementwise conjugate of a given matrix.
"""
struct Conjugate{T,S} <:AbstractMatrix{T}
    parent::S
    Conjugate(A::AbstractMatrix{T}) where T = new{T,typeof(A)}(A)
end
Base.size(A::Conjugate) = size(A.parent)
Base.getindex(A::Conjugate,ind...) = conj.(getindex(A.parent, ind...))
Base.parent(A::Conjugate) = A.parent

"""
    HermiteTridiagonal(diagonal::Vector,upper::Vector)

Represent a Hermitian tridiagonal matrix.
"""
struct HermiteTridiagonal{T,V<:AbstractVector{T}} <: AbstractMatrix{T}
    dv::V
    ev::V
end
Base.size(A::HermiteTridiagonal) = (length(dv), length(dv))


struct XSubArray{T,Ti,N,S<:SparseMatrixCSC{T,Ti},I,B,R<:AbstractVector{Ti}} <: AbstractSparseArray{T,Ti,N}
    sub::SubArray{T,N,S,I,B}
    rowval::R
end
function XSubArray(s::SubArray{T,N,SparseMatrixCSC{T,Ti},I,B}) where {T,N,Ti,I,B}
    rv = rowvals(s)
    XSubArray(s, rv)
end

const SparseUnion{Tv,Ti} = Union{SparseMatrixCSC{Tv,Ti},
                                 SubArray{Tv,2,SparseMatrixCSC{Tv,Ti},<:Tuple,false},
                                 XSubArray{Tv,Ti,2,SparseMatrixCSC{Tv,Ti},<:Tuple,false}}


struct ArrayViewWrapper{T,S<:AbstractMatrix,W} <: AbstractMatrix{T}
    data::W
end
ArrayViewWrapper(X::ArrayViewWrapper) = X
function ArrayViewWrapper(X::S) where {T,S<:AbstractArray{T}}
    ArrayViewWrapper{T,array_storage(S),S}(X)
end
Base.parent(X::ArrayViewWrapper) = X.data
Base.size(X::ArrayViewWrapper) = size(X.data)
Base.getindex(X::ArrayViewWrapper, I...) = getindex(X.data, I...)

"""
    SparseMatrixCSCSymmHerm

`Symmetric` or `Hermitian` of a `SparseMatrixCSC` or `SparseMatrixCSCView`.
"""
const SparseMatrixCSCSymmHerm{Tv,Ti} = Union{Symmetric{Tv,<:SparseUnion{Tv,Ti}},
                                            Hermitian{Tv,<:SparseUnion{Tv,Ti}}}


export iswrsparse, indextype, isupper, inflate, depth, sparsecsc, sparseaccess, sparsecopy
export array_storage
export Conjugate, HermiteTridigonal

up(B::Union{Symmetric,Hermitian,Bidiagonal}) = B.uplo == 'U' ? :U : :L
toggle(s::Symbol) = s == :U ? :L : :U

#export nziterator
#export Conjugate, HermitianTriangular

#include("nziterators.jl")
#include("operations.jl")
include("sparsify.jl")
include("combine.jl")
#include("semigroup.jl")
include("linalg.jl")
include("fallback.jl")
include("views.jl")

end # module
