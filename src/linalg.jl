
import LinearAlgebra:   mul!
import SparseArrays:    possible_adjoint, SparseMatrixCSCUnion, getcolptr, getrowval, getnzval

"""
    SparseMatrixCSCSymAdj

`Symmetric` or `Hermitian` of a `SparseMatrixCSC` or `SparseMatrixCSCView`.
"""
const SparseMatrixCSCSymAdj{T,Ti} = Union{Symmetric{T,<:SparseMatrixCSCUnion{T,Ti}},
                                          Hermitian{T,<:SparseMatrixCSCUnion{T,Ti}}}

# y .= A * x
mul!(y::AbstractVector, A::SparseMatrixCSCSymAdj, x::AbstractVector) = mul!(y, A, x, 1, 0)

# C .= α * C + β * A * B
function mul!(C::StridedVecOrMat, sA::SparseMatrixCSCSymAdj, B::StridedVecOrMat, α::Number, β::Number)
    _mul!(C, sA, B, Val(sA.uplo=='U'), α, β)
end

function _mul!(C, sA, B, uplo::Val{UPLO}, α, β) where UPLO
    A = sA.data
    n = A.n
    m = size(B, 2)
    n == size(B, 1) == size(C, 1) && m == size(C, 2) || throw(DimensionMismatch())
    colp = getcolptr(A)
    rv = getrowval(A)
    nzv = getnzval(A)
    adj = sA isa Hermitian
    z = zero(eltype(C))
    if β != 1 
        β != 0 ? rmul!(C, β) : fill!(C, z)
    end
    α == 0 && return C
    for k = 1:size(C, 2)
        @inbounds for col = 1:A.n
            αxj = α * B[col,k]
            sumcol = z
            for j = nzrange(colp, col, uplo)
                row = rv[j]
                aarc = nzv[j]
                if row == col 
                    sumcol += real(aarc) * αxj
                elseif UPLO == (row < col)
                    C[row,k] += aarc * αxj
                    sumcol += possible_adjoint(adj, aarc) * B[row,k]
                else
                    break
                end
            end
            C[col,k] += α * sumcol
        end
    end
    C
end

nzrange(colp, col, ::Val{true}) = @inbounds colp[col]:colp[col+1]-1
nzrange(colp, col, ::Val{false}) = @inbounds colp[col+1]-1:-1:colp[col]

"""
    rowindrange(rowval, colptr, col, uplo::Bool)

For a sparse matrix with `rowval` and `colptr`, return the range of indices in `rowvals`, which
are dedicated to contain the matrix row indices `1:col` (if `uplo`)
respectively `col:end` (if `!uplo`) in colum `col`.
Assumes that `rowval`is sorted increasingly.
-- unused --
"""
function rowindrange(rv::Vector{Ti}, colp::AbstractVector{Ti}, col::Ti, uplo::Bool) where Ti<:Integer
    r1 = colp[col]
    r2 = colp[col+1] - 1
    r3 = searchsortedfirst(rv, col, r1, r2, Base.Order.Forward)
    if r3 <= r2 && rv[r3] != col
        r3 -= 1uplo
    end
    uplo ? (r1:r3) : (r3:r2)
end

