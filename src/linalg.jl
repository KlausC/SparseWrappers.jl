
# y .= A * x
mul!(y::AbstractVector, A::SparseMatrixCSCSymmHerm, x::AbstractVector) = mul!(y, A, x, 1, 0)

# C .= α * A * B + β * C
function mul!(C::StridedVecOrMat{T}, sA::SparseMatrixCSCSymmHerm, B::StridedVecOrMat,
              α::Number, β::Number) where T

    fadj = sA isa Hermitian ? adjoint : transpose
    fuplo = sA.uplo == 'U' ? nzrangeup : nzrangelo
    _mul!(fuplo, fadj, C, sA, B, T(α), T(β))
end


_mul!(::Val{'U'}, fadj, C, sA, B, α, β) = _mul!(nzrangeup, fadj, C, sA, B, α, β)
_mul!(::Val{'L'}, fadj, C, sA, B, α, β) = _mul!(nzrangelo, fadj, C, sA, B, α, β)

function _mul!(nzrang::Function, fadj::Function, C, sA, B, α, β)
    A = sA.data
    n = size(A, 2)
    m = size(B, 2)
    n == size(B, 1) == size(C, 1) && m == size(C, 2) || throw(DimensionMismatch())
    rv = rowvals(A)
    nzv = nonzeros(A)
    z = zero(eltype(C))
    diagmap = fadj == transpose ? identity : real
    if β != 1 
        β != 0 ? rmul!(C, β) : fill!(C, z)
    end
    for k = 1:m
        @inbounds for col = 1:n
            αxj = α * B[col,k]
            sumcol = z
            for j = nzrang(A, col)
                row = rv[j]
                aarc = nzv[j]
                if row == col 
                    sumcol += diagmap(aarc) * αxj
                else
                    C[row,k] += aarc * αxj
                    sumcol += fadj(aarc) * B[row,k]
                end
            end
            C[col,k] += α * sumcol
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
    r1:searchsortedlast(rowvals(A), i, r1, r2, Forward)
end
function nzrangeup(A, r::AbstractVector{<:Integer}, i)
    view(r, 1:searchsortedlast(view(rowvals(A), r), i))
end
function nzrangelo(A, r::AbstractUnitRange, i)
    r1 = r.start; r2 = r.stop
    searchsortedfirst(rowvals(A), i, r1, r2, Forward):r2
end
function nzrangelo(A, r::AbstractVector{<:Integer}, i)
    view(r, searchsortedfirst(view(rowvals(A), r), i):length(r))
end

