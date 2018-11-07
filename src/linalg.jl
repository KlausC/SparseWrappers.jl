
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
    
    A = sA.data
    n = A.n
    n == size(B, 1) == size(C, 1) || throw(DimensionMismatch())
    size(B, 2) == size(C, 2) || throw(DimensionMismatch())
    colp = getcolptr(A)
    rv = getrowval(A)
    nzv = getnzval(A)
    uplo = sA.uplo == 'U'
    adjup = !uplo && sA isa Hermitian
    adjlow = uplo && sA isa Hermitian
    z = zero(eltype(C))
    if β != 1 
        β != 0 ? rmul!(C, β) : fill!(C, z)
    end
    jk = 0
    for k = 1:size(C, 2)
        @inbounds for col = 1:A.n
            αxj = α * B[col+jk]
            sumcol = z
            for j = rowindrange(rv, colp, col, uplo)
                row = rv[j]
                aarc = nzv[j]
                if row == col 
                    C[row+jk] += real(aarc) * αxj
                elseif uplo == (row < col)
                    C[row+jk] += possible_adjoint(adjup, aarc) * αxj
                    sumcol += possible_adjoint(adjlow, aarc) * B[row+jk]
                end
            end
            C[col+jk] += α * sumcol
        end
        jk += n
    end
    C
end

"""
    rowindrange(rowval, colptr, col, uplo::Bool)

For a sparse matrix with `rowval` and `colptr`, return the range of indices in `rowvals`, which
are dedicated to contain the matrix row indices `1:col` (if `uplo`)
respectively `col:end` (if `!uplo`) in colum `col`.
Assumes that `rowval`is sorted increasingly.
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

