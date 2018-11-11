
import LinearAlgebra: AbstractTriangular

"""
    isbasedsparse(A::AbstractiArray)

Returns true if A is based on a sparse array, and false otherwise.
"""
isbasedsparse(A::AbstractArray) = issparse(A)
isbasedsparse(A::Union{Transpose,Adjoint}) = isbasedsparse(A.parent)
isbasedsparse(A::AbstractTriangular) = isbasedsparse(A.data)
isbasedsparse(A::Union{Symmetric,Hermitian}) = isbasedsparse(A.data)

"""
    sparsecsc(A::AbstractArray)

Return `A` if it is a `SparseMatrixCSC` or `SparseVector`, otherwise convert to that type
in an efficient manner.
"""
sparsecsc(A::AbstractArray) = sparse(A)
sparsecsc(A::SparseMatrixCSC) = A
sparsecsc(A::SparseVector) = A
sparsecsc(A::Transpose) = sparsecsc(Transpose(sparsecsc(A.parent)))
sparsecsc(A::Adjoint) = sparsecsc(Adjoint(sparsecsc(A.parent)))
sparsecsc(A::UpperTriangular) = sparsecsc(UpperTriangular(sparsecsc(A.data)))
sparsecsc(A::UnitUpperTriangular) = sparsecsc(UnitUpperTriangular(sparsecsc(A.data)))
sparsecsc(A::LowerTriangular) = sparsecsc(LowerTriangular(sparsecsc(A.data)))
sparsecsc(A::UnitLowerTriangular) = sparsecsc(UnitLowerTriangular(sparsecsc(A.data)))
sparsecsc(A::Symmetric) = sparsecsc(Symmetric(sparsecsc(A.data), Symbol(A.uplo)))
sparsecsc(A::Hermitian) = sparsecsc(Hermitian(sparsecsc(A.data), Symbol(A.uplo)))

sparsecsc(A::Adjoint{Tv,<:SparseMatrixCSC{Tv,Ti}}) where {Tv,Ti} = sparse(A)
sparsecsc(A::Transpose{Tv,<:SparseMatrixCSC{Tv,Ti}}) where {Tv,Ti} = sparse(A)

sparsecsc(A::UpperTriangular{Tv,<:SparseMatrixCSC{Tv}}) where Tv = _sparse(nzrangeup, A, false)
sparsecsc(A::UnitUpperTriangular{Tv,<:SparseMatrixCSC{Tv}}) where Tv = _sparse(nzrangeup, A, true)
sparsecsc(A::LowerTriangular{Tv,<:SparseMatrixCSC{Tv}}) where Tv = _sparse(nzrangelo, A, false)
sparsecsc(A::UnitLowerTriangular{Tv,<:SparseMatrixCSC{Tv}}) where Tv = _sparse(nzrangelo, A, true)
function sparsecsc(A::Symmetric{Tv,<:SparseMatrixCSC{Tv}}) where Tv
    A.uplo == 'U' ? _sparse(nzrangeup, transpose, A) : _sparse(nzrangelo, transpose, A)
end
function sparsecsc(A::Hermitian{Tv,<:SparseMatrixCSC{Tv}}) where Tv
    A.uplo == 'U' ? _sparse(nzrangeup, adjoint, A) : _sparse(nzrangelo, adjoint, A)
end
    
function _sparseup(A::AbstractTriangular{Tv}, isunit::Bool) where {Tv}
    S = A.data
    rowval = rowvals(S)
    nzval = nonzeros(S)
    m, n = size(S)
    Ti = eltype(rowval)
    newcolptr = Vector{Ti}(undef, n+1)
    newrowval = Vector{Ti}(undef, nnz(S))
    newnzval = Vector{Tv}(undef, nnz(S))
    newcolptr[1] = 1
    newk = 1
    for j = 1:n
        for k = nzrange(S, j)
            i = rowval[k]
            if i < j || i == j && !isunit
                newrowval[newk] = i
                newnzval[newk] = nzval[k]
                newk += 1
            end
            i >= j && break
        end
        if isunit
            newrowval[newk] = j
            newnzval[newk] = one(Tv)
            newk += 1
        end
        newcolptr[j+1] = newk
    end
    SparseMatrixCSC(m, n, newcolptr, newrowval, newnzval)
end
  
function _sparselo(A::AbstractTriangular{Tv}, isunit::Bool) where {Tv}
    S = A.data
    rowval = rowvals(S)
    nzval = nonzeros(S)
    m, n = size(S)
    Ti = eltype(rowval)
    newcolptr = Vector{Ti}(undef, n+1)
    newrowval = Vector{Ti}(undef, nnz(S))
    newnzval = Vector{Tv}(undef, nnz(S))
    newcolptr[1] = 1
    newk = 1
    for j = 1:n
        if isunit
            newrowval[newk] = j
            newnzval[newk] = one(Tv)
            newk += 1
        end
        r = nzrange(S, j); r1 = r.start; r2 = r.stop
        if r1 <= r2
            r1 = searchsortedfirst(rowval, j, r1, r2, Base.Order.Forward)
            for k = r1:r2
                i = rowval[k]
                if i > j || i == j && !isunit
                    newrowval[newk] = i
                    newnzval[newk] = nzval[k]
                    newk += 1
                end
            end
        end
        newcolptr[j+1] = newk
    end
    SparseMatrixCSC(m, n, newcolptr, newrowval, newnzval)
end

function _sparse(fnzrange::Function, A::AbstractTriangular{Tv}, isunit::Bool) where {Tv}
    S = A.data
    rowval = rowvals(S)
    nzval = nonzeros(S)
    m, n = size(S)
    Ti = eltype(rowval)
    newcolptr = Vector{Ti}(undef, n+1)
    newrowval = Vector{Ti}(undef, nnz(S))
    newnzval = Vector{Tv}(undef, nnz(S))
    newcolptr[1] = 1
    uplo = fnzrange == nzrangeup
    newk = 1
    @inbounds for j = 1:n
        newkk = newk
        if isunit
            newk += !uplo
        end
        r = fnzrange(S, j); r1 = r.start; r2 = r.stop
        for k = r1:r2
            i = rowval[k]
            if i != j || i == j && !isunit
                newrowval[newk] = i
                newnzval[newk] = nzval[k]
                newk += 1
            end
        end
        if isunit
            uplo || (newkk = newk)
            newrowval[newkk] = j
            newnzval[newkk] = one(Tv)
            newk += uplo
        end
        newcolptr[j+1] = newk
    end
    nz = newcolptr[n+1] - 1
    resize!(newrowval, nz)
    resize!(newnzval, nz)
    SparseMatrixCSC(m, n, newcolptr, newrowval, newnzval)
end

function _sparseup(sA::Symmetric{Tv}, fadj::Function) where {Tv}
    A = sA.data
    rowval = rowvals(A)
    nzval = nonzeros(A)
    m, n = size(A)
    Ti = eltype(rowval)
    newcolptr = Vector{Ti}(undef, n+1)
    newrowval = Vector{Ti}(undef, 2nnz(A))
    newnzval = Vector{Tv}(undef, 2nnz(A))
    newcolptr[1] = 1
    for j = 1:n
        r = nzrange(A, j); r1 = r.start; r2 = r.stop
        r2 = searchsortedlast(rowval, j, r1, r2, Base.Order.Forward)
        newcolptr[j+1] = r2 - r1 + 1
        for k = r1:r2
            row = rowval[k]
            if row < j
                newcolptr[row+1] += 1
            end
        end
    end
    cumsum!(newcolptr, newcolptr)
    for j = 1:n
        newk = newcolptr[j]
        for k = nzrange(A, j)
            i = rowval[k]
            if i < j
                newrowval[newk] = i
                newnzval[newk] = nzval[k]
                newk += 1
                ni = newcolptr[i]
                newrowval[ni] = j
                newnzval[ni] = fadj(nzval[k])
                newcolptr[i] = ni + 1
            elseif i == j
                newrowval[newk] = i
                newnzval[newk] = real(nzval[k])
                newk += 1
            else
                break
            end
        end
        newcolptr[j] = newk
    end
    for j = n:-1:1
        newcolptr[j+1] = newcolptr[j]
    end
    nz = newcolptr[n+1] - 1
    newcolptr[1] = 1
    resize!(newrowval, nz)
    resize!(newnzval, nz)
    SparseMatrixCSC(m, n, newcolptr, newrowval, newnzval)
end
  
function _sparse(fnzrange::Function, fadj::Function, sA::Symmetric{Tv}) where {Tv}
    A = sA.data
    rowval = rowvals(A)
    nzval = nonzeros(A)
    m, n = size(A)
    Ti = eltype(rowval)
    newcolptr = Vector{Ti}(undef, n+1)
    newrowval = Vector{Ti}(undef, 2nnz(A))
    newnzval = Vector{Tv}(undef, 2nnz(A))
    newcolptr[1] = 1
    colrange = fnzrange === nzrangeup ? (1:n) : (n:-1:1)
    @inbounds for j = colrange
        r = fnzrange(A, j); r1 = r.start; r2 = r.stop
        newcolptr[j+1] = r2 - r1 + 1
        for k = r1:r2
            row = rowval[k]
            if row != j
                newcolptr[row+1] += 1
            end
        end
    end
    cumsum!(newcolptr, newcolptr)
    @inbounds for j = 1:n
        newk = newcolptr[j]
        for k = fnzrange(A, j)
            i = rowval[k]
            nzv = nzval[k]
            if i != j
                newrowval[newk] = i
                newnzval[newk] = nzv
                newk += 1
                ni = newcolptr[i]
                newrowval[ni] = j
                newnzval[ni] = fadj(nzv)
                newcolptr[i] = ni + 1
            else
                newrowval[newk] = i
                newnzval[newk] = real(nzv)
                newk += 1
            end
        end
        newcolptr[j] = newk
    end
    _sparse_gen(m, n, newcolptr, newrowval, newnzval)
end

function _sparse(fadj::Function, sA::Union{<:Transpose{Tv},<:Adjoint{Tv}}) where {Tv}
    A = sA.parent
    rowval = rowvals(A)
    nzval = nonzeros(A)
    m, n = size(A)
    Ti = eltype(rowval)
    newcolptr = zeros(Ti, m+1)
    newrowval = Vector{Ti}(undef, nnz(A))
    newnzval = Vector{Tv}(undef, nnz(A))
    for j = 1:n
        r = nzrange(A, j); r1 = r.start; r2 = r.stop
        for k = r1:r2
            row = rowval[k]
            newcolptr[row+1] += 1
        end
    end
    newcolptr[1] = 1
    cumsum!(newcolptr, newcolptr)
    for j = 1:n
        for k = nzrange(A, j)
            i = rowval[k]
            nzv = nzval[k]
            ni = newcolptr[i]
            newrowval[ni] = j
            newnzval[ni] = fadj(nzv)
            newcolptr[i] = ni + 1
        end
    end
    _sparse_gen(n, m, newcolptr, newrowval, newnzval)
end

function _sparse_gen(m, n, newcolptr, newrowval, newnzval)
    @inbounds for j = n:-1:1
        newcolptr[j+1] = newcolptr[j]
    end
    newcolptr[1] = 1
    nz = newcolptr[n+1] - 1
    resize!(newrowval, nz)
    resize!(newnzval, nz)
    SparseMatrixCSC(m, n, newcolptr, newrowval, newnzval)
end

