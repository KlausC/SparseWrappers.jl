# for demo only.

export equal2, multiply2

# equality of 2 matrices
function equal2(A::AbstractMatrix, B::AbstractMatrix)
    A === B && return true
    size(A) != size(B) && return false
    for (i, j, v) in nziterator(A)
        v == B[i,j] || return false
    end
    for (i, j, v) in nziterator(B)
        v == A[i,j] || return false
    end
    true
end

# multiplication
function multiply2(A::AbstractMatrix{T}, B::Union{AbstractVector{T},AbstractMatrix{T}}) where T<:Number
    m, n = size(A)
    size(B, 1) == n || throw(DimensionMismatch("matrix A has dimensions $(size(A)), vector B has length $(length(B))"))

    v = zeros(T, m, size(B)[2:end]...)
    for (i, j, aij) in nziterator(A)
        v[i,:] += B[j,:] * aij
    end
    v
end

