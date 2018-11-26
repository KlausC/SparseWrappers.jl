
export linop, linop_oriig, fallback_demo
using BenchmarkTools
using Random

"""
    fallback_demo()

```julia-repl
julia> n = 10000
10000
julia> Random.seed!(0); Asym = Symmetric(sprandn(n, n, 0.01)); b = randn(n); nnz(Asym.data)
1000598

julia> @benchmark linop_orig(Asym) # abstractarray fallback 
BenchmarkTools.Trial:
  memory estimate:  16 bytes
  allocs estimate:  1
  --------------
  minimum time:     2.406 s (0.00% GC)
  median time:      2.409 s (0.00% GC)
  mean time:        2.439 s (0.00% GC)
  maximum time:     2.503 s (0.00% GC)
  --------------
  samples:          3
  evals/sample:     1

julia> @benchmark linop(Asym) # sparsearray fallback in action
BenchmarkTools.Trial:
  memory estimate:  15.32 MiB
  allocs estimate:  8
  --------------
  minimum time:     10.980 ms (0.00% GC)
  median time:      11.669 ms (3.40% GC)
  mean time:        11.807 ms (3.59% GC)
  maximum time:     64.878 ms (82.49% GC)
  --------------
  samples:          423
  evals/sample:     1

julia> @benchmark linop(\$(sparse(Asym))) # only sparse operation
BenchmarkTools.Trial:
  memory estimate:  0 bytes
  allocs estimate:  0
  --------------
  minimum time:     957.077 μs (0.00% GC)
  median time:      1.001 ms (0.00% GC)
  mean time:        1.008 ms (0.00% GC)
  maximum time:     1.091 ms (0.00% GC)
  --------------
  samples:          4923
  evals/sample:     1
```
"""
function fallback_demo(n::Integer=10000)
    Random.seed!(0)
    Asym = Symmetric(sprandn(n, n, 0.01))
    b1 = @benchmark linop_orig($Asym) # abstractarray fallback 
    b2 = @benchmark linop($Asym) # sparsearray fallback in action
    b3 = @benchmark linop($(sparse(Asym))) # only sparse operation
    b1, b2, b3
end

# This file demonstrates how to change the abstractarray fallback to a
# sparse fallback.
#
#
# Sample function with abstarctarray fallback
function linop_orig(A::AbstractMatrix)
    n, m = size(A)
    sum = zero(eltype(A))
    for j = 1:n
        for i = 1:m
            sum += A[i,j]
        end
    end
    sum
end

# An implementation for sparse matrices exists
function linop(A::SparseMatrixCSC)
    m, n = size(A)
    sum = zero(eltype(A))
    nz = nonzeros(A)
    for k = 1:nnz(A)
        sum += nz[k]
    end
    sum
end

# convert argument to SparseMatrixCSC to call sparse implementation
linop_sparse(A::AbstractMatrix) = linop(sparsecsc(A))

# The original function is modified
function linop(A::AbstractMatrix)
    # issparse(A) && return linop(sparse(A)) # That could easily lead to infinite loops
    issparse(A) && return linop_sparse(A)
    n, m = size(A)
    sum = zero(eltype(A))
    for j = 1:n
        for i = 1:m
            sum += A[i,j]
        end
    end
    sum
end

