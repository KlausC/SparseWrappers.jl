module SparseWrappers

using LinearAlgebra
using SparseArrays

export nziterator

include("nziterators.jl")
include("operations.jl")

end # module
