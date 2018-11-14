module SparseWrappers

using LinearAlgebra
using SparseArrays

import LinearAlgebra:   mul!
import SparseArrays:    possible_adjoint, SparseMatrixCSCUnion
import Base.Order: Forward

"""
    SparseMatrixCSCSymmHerm

`Symmetric` or `Hermitian` of a `SparseMatrixCSC` or `SparseMatrixCSCView`.
"""
const SparseMatrixCSCSymmHerm{Tv,Ti} = Union{Symmetric{Tv,<:SparseMatrixCSCUnion{Tv,Ti}},
                                            Hermitian{Tv,<:SparseMatrixCSCUnion{Tv,Ti}}}

#export nziterator
#export Conjugate, HermitianTriangular

#include("nziterators.jl")
#include("operations.jl")
include("sparsify.jl")
#include("combine.jl")
#include("semigroup.jl")
include("linalg.jl")

end # module
