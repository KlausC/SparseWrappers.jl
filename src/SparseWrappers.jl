module SparseWrappers

using LinearAlgebra
using SparseArrays

import LinearAlgebra:   mul!
import SparseArrays:    SparseMatrixCSCUnion
import Base.Order: Forward
import LinearAlgebra: AbstractTriangular, UnitLowerTriangular, UnitUpperTriangular

import SparseArrays: nzrange, rowvals, nonzeros, nnz, nzvalview, SparseMatrixCSCView

if VERSION >= v"1.1.0-DEV"
import SparseArrays: LowerTriangularPlain, UpperTriangularPlain
else
const UnitDiagonalTriangular = Union{UnitUpperTriangular,UnitLowerTriangular}

const LowerTriangularPlain{T} = Union{
            LowerTriangular{T,<:SparseMatrixCSCUnion{T}},
            UnitLowerTriangular{T,<:SparseMatrixCSCUnion{T}}}

const LowerTriangularWrapped{T} = Union{
            Adjoint{T,<:UpperTriangular{T,<:SparseMatrixCSCUnion{T}}},
            Adjoint{T,<:UnitUpperTriangular{T,<:SparseMatrixCSCUnion{T}}},
            Transpose{T,<:UpperTriangular{T,<:SparseMatrixCSCUnion{T}}},
            Transpose{T,<:UnitUpperTriangular{T,<:SparseMatrixCSCUnion{T}}}} where T

const UpperTriangularPlain{T} = Union{
            UpperTriangular{T,<:SparseMatrixCSCUnion{T}},
            UnitUpperTriangular{T,<:SparseMatrixCSCUnion{T}}}

const UpperTriangularWrapped{T} = Union{
            Adjoint{T,<:LowerTriangular{T,<:SparseMatrixCSCUnion{T}}},
            Adjoint{T,<:UnitLowerTriangular{T,<:SparseMatrixCSCUnion{T}}},
            Transpose{T,<:LowerTriangular{T,<:SparseMatrixCSCUnion{T}}},
            Transpose{T,<:UnitLowerTriangular{T,<:SparseMatrixCSCUnion{T}}}} where T

const UpperTriangularSparse{T} = Union{
            UpperTriangularWrapped{T}, UpperTriangularPlain{T}} where T

const LowerTriangularSparse{T} = Union{
            LowerTriangularWrapped{T}, LowerTriangularPlain{T}} where T

const TriangularSparse{T} = Union{
            LowerTriangularSparse{T}, UpperTriangularSparse{T}} where T

end

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
export unwrap
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
include("timings.jl")
include("mmatrix.jl")
include("universal.jl")

end # module
