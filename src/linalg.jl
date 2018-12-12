

function testmul(A,x)
    tmul!(similar(x), A, x)
end

# y .= A * x
tmul!(y::AbstractVector, A::SparseMatrixCSCSymmHerm, x::AbstractVector) = tmul!(y, A, x, 1, 0)

# C .= α * A * B + β * C
function tmul!(C::StridedVecOrMat{T}, sA::SparseMatrixCSCSymmHerm, B::StridedVecOrMat,
              α::Number, β::Number) where T

    fuplo = sA.uplo == 'U' ? nzrangeup : nzrangelo
    _tmul!(fuplo, C, sA, B, T(α), T(β))
end


_mul!(::Val{'U'}, fadj, C, sA, B, α, β) = _mul!(nzrangeup, fadj, C, sA, B, α, β)
_mul!(::Val{'L'}, fadj, C, sA, B, α, β) = _mul!(nzrangelo, fadj, C, sA, B, α, β)

function _tmul!(nzrang::Function, C::StridedVecOrMat{T}, sA, B, α, β) where T
    A = sA.data
    n = size(A, 2)
    m = size(B, 2)
    n == size(B, 1) == size(C, 1) && m == size(C, 2) || throw(DimensionMismatch())
    rv = rowvals(A)
    nzv = nonzeros(A)
    let z = T(0), sumcol=z, αxj=z, aarc=z, α = α #, row
    # row = 0 # zero(eltype(rv))
    if β != 1 
        β != 0 ? rmul!(C, β) : fill!(C, z)
    end
    @inbounds for k = 1:m
        for col1 = 1:n
            col = col1
            αxj = B[col,k] * α
            sumcol = z
            for j = nzrang(A, col)
                row = rv[j]
                aarc = nzv[j]
                if row == col 
                    sumcol += (sA isa Hermitian ? real : identity)(aarc) * B[row,k]
                else #if (row < col) == ul 
                    C[row,k] += aarc * αxj
                    sumcol += (sA isa Hermitian ? adjoint : transpose)(aarc) * B[row,k]
                end
            end
            C[col,k] += α * sumcol
        end
    end
    end
    C
end

#nzrangeup(A, i) = nzrangeup(A, rowvals(A), i)
#nzrangelo(A, i) = nzrangelo(A, rowvals(A), i)

nzrangeup(A, i) = nzrangeup(A, nzrange(A, i), i)
nzrangelo(A, i) = nzrangelo(A, nzrange(A, i), i)
function nzrangeup(A, r::AbstractUnitRange, i)
    r1 = r.start; r2 = r.stop
    rv = rowvals(A)
    r1:searchsortedlast(rowvals(A), i, r1, r2, Forward)
    # @inbounds r2 < r1 || rv[r2] <= i ? r : r1:searchsortedlast(rv, i, r1, r2, Forward)
end
function nzrangeup(A, r::AbstractVector{<:Integer}, i)
    view(r, 1:searchsortedlast(view(rowvals(A), r), i))
end
function nzrangelo(A, r::AbstractUnitRange, i)
    r1 = r.start; r2 = r.stop
    rv = rowvals(A)
    # searchsortedfirst(rv, i, r1, r2, Forward):r2
    @inbounds r2 < r1 || rv[r1] >= i ? r : searchsortedfirst(rv, i, r1, r2, Forward):r2
end
function nzrangelo(A, r::AbstractVector{<:Integer}, i)
    view(r, searchsortedfirst(view(rowvals(A), r), i):length(r))
end

# Gustavsen's matrix multiplication algorithm revisited
function spmatmul(A::SparseMatrixCSC{Tv,Ti}, B::Union{<:SparseMatrixCSC{Tv,Ti},<:SparseVector{Tv,Ti}}) where {Tv,Ti,N}
    mA, nA = size(A)
    nB = size(B, 2)
    nA == size(B, 1) || throw(DimensionMismatch())

    rowvalA = rowvals(A); nzvalA = nonzeros(A)
    rowvalB = rowvals(B); nzvalB = nonzeros(B)
    nnzC = estimate_mulsize(mA, nnz(A), nA, nnz(B), nB)
    if B isa SparseMatrixCSC; colptrC = Vector{Ti}(undef, nB+1) end
    rowvalC = Vector{Ti}(undef, nnzC)
    nzvalC = Vector{Tv}(undef, nnzC)

    @inbounds begin
        ip = 1
        x  = Vector{Tv}(undef, mA)
        xb = BitArray(undef, mA)
        for i in 1:nB
            fill!(xb, false)
            if ip + mA - 1 > nnzC
                nnzC += max(mA, nnzC>>2)
                resize!(rowvalC, nnzC)
                resize!(nzvalC, nnzC)
            end
            if B isa SparseMatrixCSC; colptrC[i] = ip end
            for jp in nzrange(B, i)
                nzB = nzvalB[jp]
                j = rowvalB[jp]
                for kp in nzrange(A, j)
                    nzC = nzvalA[kp] * nzB
                    k = rowvalA[kp]
                    if xb[k]
                        x[k] += nzC
                    else
                        x[k] = nzC
                        xb[k] = true
                    end
                end
            end
            for k in findall(xb)
                nzvalC[ip] = x[k]
                rowvalC[ip] = k
                ip += 1
            end
        end
        if B isa SparseMatrixCSC; colptrC[nB+1] = ip end
    end

    ip -= 1
    resize!(rowvalC, ip)
    resize!(nzvalC, ip)

    # This modification of Gustavson algorithm has sorted row indices.
    if B isa SparseMatrixCSC
        SparseMatrixCSC(mA, nB, colptrC, rowvalC, nzvalC)
    else
        SparseVector(mA, rowvalC, nzvalC)
    end
end

function estimate_mulsize(m::Integer, nnzA::Integer, n::Integer, nnzB::Integer, k::Integer)
    p = (nnzA / (m * n)) * (nnzB / (n * k))
    Int(ceil(-expm1(log1p(-p) * n) * m * k)) # is (1 - (1 - p)^n) * m * k
end

