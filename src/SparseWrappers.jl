module SparseWrappers

using LinearAlgebra
using SparseArrays

import LinearAlgebra:   mul!
import SparseArrays:    possible_adjoint, SparseMatrixCSCUnion
import Base.Order: Forward


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

"""
    SparseMatrixCSCSymmHerm

`Symmetric` or `Hermitian` of a `SparseMatrixCSC` or `SparseMatrixCSCView`.
"""
const SparseMatrixCSCSymmHerm{Tv,Ti} = Union{Symmetric{Tv,<:SparseUnion{Tv,Ti}},
                                            Hermitian{Tv,<:SparseUnion{Tv,Ti}}}


export issparse, indextype, isupper

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
