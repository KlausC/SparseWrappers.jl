
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

