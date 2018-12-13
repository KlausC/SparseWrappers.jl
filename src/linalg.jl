

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
    # nnzC = min(mA*nB, length(nzvalA) + length(nzvalB))
    nnzC = estimate_mulsize(mA, nnz(A), nA, nnz(B), nB) * 11 ÷ 10
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
    nzpercol = nnzC ÷ nB
    sorting_is_better = (ilog2(nzpercol) - 1) * nzpercol < mA

    # used in sortcol!
    # index = zeros(Ti, mA)
    # row = zeros(Ti, mA)
    # perm = Base.Perm(Base.ord(isless, identity, false, Base.Order.Forward), row)

    @inbounds begin
        ip = 1
        x  = Vector{Tv}(undef, mA)
        xb = zeros(Ti, mA)
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
                    if xb[k] == i
                        x[k] += nzC
                    else
                        x[k] = nzC
                        xb[k] = i
                        rowvalC[ip] = k
                        ip += 1
                    end
                end
            end
            if ip > ip0
                if sorting_is_better
                    # sortcol!(ip0, ip-1, rowvalC, index, row, perm)
                    @simd for k = ip0:ip-1
                        nzvalC[k] = x[rowvalC[k]]
                    end
                else
                    for k = 1:mA
                        if xb[k] == i
                            rowvalC[ip0] = k
                            nzvalC[ip0] = x[k]
                            ip0 += 1
                        end
                    end
                end
            end
        end
        colptrC[nB+1] = ip
    end

    ip -= 1
    resize!(rowvalC, ip)
    resize!(nzvalC, ip)

    # This modification of Gustavson algorithm has sorted row indices if !sorting_is_better.
    Cunsorted = SparseMatrixCSC(mA, nB, colptrC, rowvalC, nzvalC)
    if sorting_is_better
        SparseArrays.sortSparseMatrixCSC!(Cunsorted, sortindices=sortindices)
    else
        Cunsorted
    end
end

# estimated number of non-zeros in matrix product
function estimate_mulsize(m::Integer, nnzA::Integer, n::Integer, nnzB::Integer, k::Integer)
    p = (nnzA / (m * n)) * (nnzB / (n * k))
    isnan(p) ? 0 : Int(ceil(-expm1(log1p(-p) * n) * m * k)) # is (1 - (1 - p)^n) * m * k
end

ilog2(n::Integer) = sizeof(n)<<3 - leading_zeros(n)

function sortcol!(lo::Integer, hi::Integer, rowval::Vector, index, row, perm)

    @inbounds begin
        numrows = hi - lo + 1
        if numrows <= 1
            return
        elseif numrows == 2
            f, s = lo, hi
            if rowval[f] > rowval[s]
                rowval[f], rowval[s] = rowval[s], rowval[f]
            end
            return
        end
        resize!(row, numrows)
        resize!(index, numrows)

        jj = 1
        @simd for j = lo:hi
            row[jj] = rowval[j]
            jj += 1
        end

        if numrows <= 16
            alg = Base.Sort.InsertionSort
        else
            alg = Base.Sort.QuickSort
        end

        # Reset permutation
        index .= 1:numrows

        sort!(index, alg, perm)

        jj = 1
        @simd for j = lo:hi
            rowval[j] = row[index[jj]]
            jj += 1
        end
    end
end

