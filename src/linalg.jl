

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

function spmatmul_orig(A::SparseMatrixCSC{Tv,Ti}, B::SparseMatrixCSC{Tv,Ti};
                  sortindices::Symbol = :sortcols) where {Tv,Ti}
    mA, nA = size(A)
    mB, nB = size(B)
    nA==mB || throw(DimensionMismatch())

    colptrA = A.colptr; rowvalA = A.rowval; nzvalA = A.nzval
    colptrB = B.colptr; rowvalB = B.rowval; nzvalB = B.nzval
    # TODO: Need better estimation of result space
    nnzC = min(mA*nB, length(nzvalA) + length(nzvalB))
    colptrC = Vector{Ti}(undef, nB+1)
    rowvalC = Vector{Ti}(undef, nnzC)
    nzvalC = Vector{Tv}(undef, nnzC)

    @inbounds begin
        ip = 1
        xb = zeros(Ti, mA)
        x  = zeros(Tv, mA)
        for i in 1:nB 
            if ip + mA - 1 > nnzC 
                resize!(rowvalC, nnzC + max(nnzC,mA))
                resize!(nzvalC, nnzC + max(nnzC,mA))
                nnzC = length(nzvalC)
            end
            colptrC[i] = ip 
            for jp in colptrB[i]:(colptrB[i+1] - 1) 
                nzB = nzvalB[jp]
                j = rowvalB[jp]
                for kp in colptrA[j]:(colptrA[j+1] - 1) 
                    nzC = nzvalA[kp] * nzB
                    k = rowvalA[kp]
                    if xb[k] != i 
                        rowvalC[ip] = k
                        ip += 1 
                        xb[k] = i
                        x[k] = nzC
                    else
                        x[k] += nzC
                    end
                end
            end
                        for vp in colptrC[i]:(ip - 1)
                nzvalC[vp] = x[rowvalC[vp]]
            end
        end
        colptrC[nB+1] = ip
    end

    deleteat!(rowvalC, colptrC[end]:length(rowvalC))
    deleteat!(nzvalC, colptrC[end]:length(nzvalC))

    # The Gustavson algorithm does not guarantee the product to have sorted row indices.
    Cunsorted = SparseMatrixCSC(mA, nB, colptrC, rowvalC, nzvalC)
    C = SparseArrays.sortSparseMatrixCSC!(Cunsorted, sortindices=sortindices)
    return C
end

function spmatmul_mod(A::SparseMatrixCSC{Tv,Ti}, B::SparseMatrixCSC{Tv,Ti};
                  sortindices::Symbol = :sortcols) where {Tv,Ti}
    mA, nA = size(A)
    mB, nB = size(B)
    nA==mB || throw(DimensionMismatch())

    colptrA = A.colptr; rowvalA = A.rowval; nzvalA = A.nzval
    colptrB = B.colptr; rowvalB = B.rowval; nzvalB = B.nzval
    # TODO: Need better estimation of result space
    # nnzC = min(mA*nB, length(nzvalA) + length(nzvalB))
    nnzC = estimate_mulsize(mA, nnz(A), nA, nnz(B), nB) * 11 ÷ 10
    colptrC = Vector{Ti}(undef, nB+1)
    rowvalC = Vector{Ti}(undef, nnzC)
    nzvalC = Vector{Tv}(undef, nnzC)

    @inbounds begin
        ip = 1
        xb = fill(false, mA)
        x  = Vector{Tv}(undef, mA)
        for i in 1:nB
            if ip + mA - 1 > nnzC
                resize!(rowvalC, nnzC + max(nnzC,mA))
                resize!(nzvalC, nnzC + max(nnzC,mA))
                nnzC = length(nzvalC)
            end
            colptrC[i] = ip
            for jp in colptrB[i]:(colptrB[i+1] - 1)
                nzB = nzvalB[jp]
                j = rowvalB[jp]
                for kp in colptrA[j]:(colptrA[j+1] - 1)
                    nzC = nzvalA[kp] * nzB
                    k = rowvalA[kp]
                    if !xb[k]
                        rowvalC[ip] = k
                        ip += 1
                        xb[k] = true
                        x[k] = nzC
                    else
                        x[k] += nzC
                    end
                end
            end
            for vp in colptrC[i]:(ip - 1)
                nzvalC[vp] = x[rowvalC[vp]]
                xb[rowvalC[vp]] = false
            end
        end
        colptrC[nB+1] = ip
    end

    deleteat!(rowvalC, colptrC[end]:length(rowvalC))
    deleteat!(nzvalC, colptrC[end]:length(nzvalC))

    # The Gustavson algorithm does not guarantee the product to have sorted row indices.
    Cunsorted = SparseMatrixCSC(mA, nB, colptrC, rowvalC, nzvalC)
    C = SparseArrays.sortSparseMatrixCSC!(Cunsorted, sortindices=sortindices)
    return C
end

# Gustavsen's matrix multiplication algorithm revisited
function spmatmul_alt(A::SparseMatrixCSC{Tv,Ti}, B::SparseMatrixCSC{Tv,Ti};
                  sortindices::Symbol = :sortcols) where {Tv,Ti}

    mA, nA = size(A)
    nB = size(B, 2)
    nA == size(B, 1) || throw(DimensionMismatch())

    rowvalA = rowvals(A); nzvalA = nonzeros(A)
    rowvalB = rowvals(B); nzvalB = nonzeros(B)
    nnzC = estimate_mulsize(mA, nnz(A), nA, nnz(B), nB) * 11 ÷ 10
    colptrC = Vector{Ti}(undef, nB+1)
    rowvalC = Vector{Ti}(undef, nnzC)
    nzvalC = Vector{Tv}(undef, nnzC)
    nzpercol = nnzC ÷ max(nB, 1)

    @inbounds begin
        ip = 1
        #x  = Vector{Tv}(undef, mA)
        xb = fill(false, mA)
        for i in 1:nB
            if ip + mA - 1 > nnzC
                nnzC += max(mA, nnzC>>2)
                resize!(rowvalC, nnzC)
                resize!(nzvalC, nnzC)
            end
            ip0 = ip
            colptrC[i] = ip
            for jp in nzrange(B, i)
                nzB = nzvalB[jp]
                j = rowvalB[jp]
                for kp in nzrange(A, j)
                    nzC = nzvalA[kp] * nzB
                    k = rowvalA[kp]
                    if xb[k]
                        nzvalC[k+ip0-1] += nzC
                    else
                        nzvalC[k+ip0-1] = nzC
                        xb[k] = true
                        ip += 1
                    end
                end
            end
            if ip > ip0
                vp = ip0
                for k = 1:mA
                    if xb[k]
                        rowvalC[vp] = k
                        nzvalC[vp] = nzvalC[k+ip0-1]
                        xb[k] = false 
                        vp += 1
                    end
                end
            end
        end
        colptrC[nB+1] = ip
    end

    resize!(rowvalC, ip - 1)
    resize!(nzvalC, ip - 1)

    # This modification of Gustavson algorithm has sorted row indices
    SparseMatrixCSC(mA, nB, colptrC, rowvalC, nzvalC)
end
# Gustavsen's matrix multiplication algorithm revisited
function spmatmul(A::SparseMatrixCSC{Tv,Ti}, B::SparseMatrixCSC{Tv,Ti};
                  sortindices::Symbol = :sortcols) where {Tv,Ti}
    mA, nA = size(A)
    nB = size(B, 2)
    nA == size(B, 1) || throw(DimensionMismatch())

    rowvalA = rowvals(A); nzvalA = nonzeros(A)
    rowvalB = rowvals(B); nzvalB = nonzeros(B)
    nnzC = estimate_mulsize(mA, nnz(A), nA, nnz(B), nB) * 11 ÷ 10
    colptrC = Vector{Ti}(undef, nB+1)
    rowvalC = Vector{Ti}(undef, nnzC)
    nzvalC = Vector{Tv}(undef, nnzC)
    nzpercol = nnzC ÷ max(nB, 1)

    @inbounds begin
        ip = 1
        xb = fill(false, mA)
        for i in 1:nB
            if ip + mA - 1 > nnzC
                nnzC += max(mA, nnzC>>2)
                resize!(rowvalC, nnzC)
                resize!(nzvalC, nnzC)
            end
            colptrC[i] = ip0 = ip
            k0 = ip - 1
            for jp in nzrange(B, i)
                nzB = nzvalB[jp]
                j = rowvalB[jp]
                for kp in nzrange(A, j)
                    nzC = nzvalA[kp] * nzB
                    k = rowvalA[kp]
                    if xb[k]
                        nzvalC[k+k0] += nzC
                    else
                        nzvalC[k+k0] = nzC
                        xb[k] = true
                        rowvalC[ip] = k
                        ip += 1
                    end
                end
            end
            if ip > ip0
                if prefer_sort(ip-k0, mA)
                    sort!(rowvalC, ip0, ip-1, QuickSort, Base.Order.Forward)
                    for vp = ip0:ip-1
                        k = rowvalC[vp]
                        xb[k] = false
                        nzvalC[vp] = nzvalC[k+k0]
                    end
                else
                    for k = 1:mA
                        if xb[k]
                            xb[k] = false
                            rowvalC[ip0] = k
                            nzvalC[ip0] = nzvalC[k+k0]
                            ip0 += 1
                        end
                    end
                end
            end
        end
        colptrC[nB+1] = ip
    end

    resize!(rowvalC, ip - 1)
    resize!(nzvalC, ip - 1)

    # This modification of Gustavson algorithm has sorted row indices
    C = SparseMatrixCSC(mA, nB, colptrC, rowvalC, nzvalC)
    return C
end

# estimated number of non-zeros in matrix product
# it is assumed, that the non-zero indices are distributed independently and uniformly
# in both matrices. Over-estimation is possible if that is not the case.
function estimate_mulsize(m::Integer, nnzA::Integer, n::Integer, nnzB::Integer, k::Integer)
    p = (nnzA / (m * n)) * (nnzB / (n * k))
    p >= 1 ? m*k : p > 0 ? Int(ceil(-expm1(log1p(-p) * n)*m*k)) : 0 # (1-(1-p)^n)*m*k
end

# determine if sort! shall be used or the whole column be scanned
# based on empirical data on i7-3610QM CPU
# which measured runtimes of the scanning and the sorting loops of the algorithm.
# The parameters 6 and 3 might be modified.
prefer_sort(nz::Integer, m::Integer) = m > 6 && 3 * ilog2(nz) * nz < m

# minimal number of bits required to represent integer; ilog2(n) >= log2(n)
ilog2(n::Integer) = sizeof(n)<<3 - leading_zeros(n)

