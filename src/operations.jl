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

# for comparison: multiplication of a Symmetirc(A, :U)
function multiply3(A::Symmetric{T,<:SparseMatrixCSC}, B::Union{AbstractVector{T},AbstractMatrix{T}}) where T<:Number
    
    m, n = size(A)
    size(B, 1) == n || throw(DimensionMismatch("matrix A has dimensions $(size(A)), vector B has length $(length(B))"))
    colptr = A.data.colptr
    rowval = A.data.rowval
    nzval = A.data.nzval
    v = zeros(T, m, size(B)[2:end]...)

    for j = 1:n, k = colptr[j]:colptr[j+1]-1
        i = rowval[k]
        if i <= j
            aij = nzval[k]
            v[i,:] += B[j,:] * aij
            if i < j
                v[j,:] += B[i,:] * aij
            end
        end
    end
    v
end

# converting to a SparseMatrixCSC
function sparse2(A::AbstractMatrix{T}) where T<:Number
    
    Ti = Int
    m, n = size(A)
    I = Ti[]
    J = Ti[]
    Z = T[]
    colptr = zeros(Ti, n+1)
    colptr[1] = 1
    for (i, j, z) in nziterator(A)
        push!(I, i)
        push!(J, j)
        push!(Z, z)
        colptr[j+1] += 1
    end
    cumsum!(colptr, colptr)
    p = collect(1:length(I))
    p = sortperm2!(p, J, I)
    SparseMatrixCSC{T,Ti}(m, n, colptr, I[p], Z[p])
end

# sorting two integer vectors
function sortperm2!(p::Vector{Int}, v1::Vector{T}, v2::Vector{T}) where T<:Integer
    n = length(v1)
    n == length(v2) || throw(DimensionMismatch("vectors need equal lengths"))
    function lt(i::Int, j::Int)
        @inbounds begin v1[i] < v1[j] || v1[i] === v1[j] &&  v2[i] < v2[j] end
    end
    sort!(p, lt=lt)
    p
end

