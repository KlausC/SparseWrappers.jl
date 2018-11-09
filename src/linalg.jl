
import LinearAlgebra:   mul!
import SparseArrays:    possible_adjoint, SparseMatrixCSCUnion
import Base.Order: Forward

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
    
    T = eltype(sA)
    uplo = sA.uplo == 'U'
    fuplo = uplo ? nzrangeup : nzrangelo
    fadj = sA isa Hermitian ? adjoint : transpose
    fal = α * one(T) == one(T) ? identity : x -> α * x
    _mul!(fuplo, fadj, C, sA, B, uplo, fal, β)
end

function _mul!(fuplo::Function, fadj::Function, C, sA, B, uplo::Bool, fal::Function, β)
    A = sA.data
    n = A.n
    m = size(B, 2)
    n == size(B, 1) == size(C, 1) && m == size(C, 2) || throw(DimensionMismatch())
    rv = rowvals(A)
    nzv = nonzeros(A)
    z = zero(eltype(C))
    if β != 1 
        β != 0 ? (@inbounds rmul!(C, β)) : (@inbounds fill!(C, z))
    end
    for k = 1:size(C, 2)
        @inbounds for col = 1:A.n
            αxj = fal(B[col,k])
            sumcol = z
            for j = fuplo(A, col)
                row = rv[j]
                aarc = nzv[j]
                if row == col 
                    sumcol += real(aarc) * αxj
                else
                    C[row,k] += aarc * αxj
                    sumcol += fadj(aarc) * B[row,k]
                end
            end
            C[col,k] += fal(sumcol)
        end
    end
    C
end

#nzrangeup(A, i) = nzrangeup(A, rowvals(A), i)
#nzrangelo(A, i) = nzrangelo(A, rowvals(A), i)

#nzrangeup(A, i) = nzrangeuplo(A, i, 1, i)
function nzrangeup(A, i)
    r1, r2 = extrema(nzrange(A, i))
    r1:searchsortedlast(rowvals(A), i, r1, r2, Forward)
end
function nzrangelo(A, i)
    r1, r2 = extrema(nzrange(A, i))
    searchsortedfirst(rowvals(A), i, r1, r2, Forward):r2
end

function nzrangeuplo(A, i, y1, y2)
    r1, r2 = extrema(nzrange(A, i))
    if y1 > 1
        r1 = searchsortedfirst(rowvals(A), i, r1, r2, Forward)
    end
    if y2 < size(A,2)
        r2 = searchsortedlast(rowvals(A), i, r1, r2, Forward)
    end
    r1:r2
end

#nzrange(colp, col, ::Val{true}) = @inbounds colp[col]:colp[col+1]-1
#nzrange(colp, col, ::Val{false}) = @inbounds colp[col+1]-1:-1:colp[col]

"""
    nzrangeup(A, rowval, colptr, col, uplo::Bool)

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

