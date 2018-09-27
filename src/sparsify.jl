
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

sparsecsc(A::UpperTriangular{Tv,<:SparseMatrixCSC{Tv}}) where Tv = _sparseup(A, false)
sparsecsc(A::UnitUpperTriangular{Tv,<:SparseMatrixCSC{Tv}}) where Tv = _sparseup(A, true)
sparsecsc(A::LowerTriangular{Tv,<:SparseMatrixCSC{Tv}}) where Tv = _sparsedown(A, false)
sparsecsc(A::UnitLowerTriangular{Tv,<:SparseMatrixCSC{Tv}}) where Tv = _sparsedown(A, true)
    
function _sparseup(A::AbstractTriangular{Tv}, isunit::Bool) where Tv
    S = A.data
    colptr = S.colptr
    rowval = S.rowval
    nzval = S.nzval
    m, n = size(S)

    newcolptr = copy(colptr)
    newrowval = copy(rowval)
    newnzval = copy(nzval)

    newk = 1
    for j = 1:n
        for k = colptr[j]:colptr[j+1]-1
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
  
function _sparsedown(A::AbstractTriangular{Tv}, isunit::Bool) where Tv
    S = A.data
    colptr = S.colptr
    rowval = S.rowval
    nzval = S.nzval
    m, n = size(S)

    newcolptr = copy(colptr)
    newrowval = copy(rowval)
    newnzval = copy(nzval)

    newk = 1
    for j = 1:n
        if isunit
            newrowval[newk] = j
            newnzval[newk] = one(Tv)
            newk += 1
        end
        r1 = colptr[j]
        r2 = colptr[j+1]-1
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

