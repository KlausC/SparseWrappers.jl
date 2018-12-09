using SparseWrappers
using Test
using Random
using LinearAlgebra
using SparseArrays

import SparseWrappers:  Conjugate

@testset "linalg"     begin include("linalg.jl") end
@testset "sparsify"   begin include("sparsify.jl") end
@testset "views"      begin include("views.jl") end
@testset "fallback"   begin include("fallback.jl") end
@testset "combine"    begin include("combine.jl") end
nothing
