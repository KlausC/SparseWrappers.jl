import LinearAlgebra: AbstractTriangular, UnitLowerTriangular, UnitUpperTriangular
import SparseArrays: LowerTriangularPlain, UpperTriangularPlain

"""
    iswrsparse(S)

Returns `true` if `S` is sparse, and `false` otherwise.

# Examples
```jldoctest
julia> sv = sparsevec([1, 4], [2.3, 2.2], 10)
10-element SparseVector{Float64,Int64} with 2 stored entries:
  [1 ]  =  2.3
  [4 ]  =  2.2

julia> iswrsparse(sv)
true

julia> iswrsparse(Array(sv))
false

```
"""
iswrsparse(::T) where T<:AbstractArray = iswrsparse(T)
iswrsparse(T::Type) = walk_wrapper(_iswrsparse, T)
_iswrsparse(::Type) = false
_iswrsparse(::Type{<:AbstractSparseArray}) = true

iswrappedsparse(::Type{<:AbstractSparseArray}) = false
iswrappedsparse(T::Type) = iswrsparse(T)

indextype(::T) where T<:AbstractArray = indextype(T)
indextype(T::Type) = walk_wrapper(_indextype, T)
_indextype(::Type{<:AbstractSparseArray{<:Any,Ti}}) where Ti = Ti
_indextype(::Type) = Int

nnz_estimation(A::AbstractMatrix) = walk_wrapper(_nnz_estimation, A)
_nnz_estimation(::Any) = missing
_nnz_estimation(A::SparseMatrixCSC) = nnz(A)
_nnz_estimation(A::SparseVector) = nnz(A)
_nnz_estimation(A::Array) = length(A)

import LinearAlgebra: Symmetric, Hermitian, LowerTriangular, UnitLowerTriangular
import LinearAlgebra: UpperTriangular, UnitUpperTriangular, Transpose, Adjoint

isupper(A::AbstractArray) = walk_wrapper(_isupper, A)
_isupper(::Any) = missing
_isupper(::Union{UpperTriangular,UnitUpperTriangular}) = true
_isupper(A::Union{Symmetric,Hermitian}) = A.uplo == 'U'
_islower(::Union{LowerTriangular,UnitLowerTriangular}) = true
_islower(A::Union{Symmetric,Hermitian}) = A.uplo == 'L'

walk_wrapper(f::Function, x::Any) = f(x)
walk_wrapper_deep(f::Function, x::Any) = f(x, nothing)
for wr in (Conjugate, Symmetric, Hermitian,
                  LowerTriangular, UnitLowerTriangular,
                  UpperTriangular, UnitUpperTriangular,
                  Transpose, Adjoint,
                  SubArray)

    pl = wr === SubArray ? :($wr{<:Any,<:Any,T}) : :($wr{<:Any,T})
    @eval function walk_wrapper(f::Function, ::Type{<:$pl}) where T
        walk_wrapper(f, T)
    end
    @eval function walk_wrapper(f::Function, A::$pl) where T
        res = f(A)
        ismissing(res) && ( res = walk_wrapper(f, parent(A)) )
        ifelse(ismissing(res), false, res)
    end
end

"""
    inflate(A::AbstractMatrix)

Reduce depth in the case of nested wrappers of an abstract array.
If `dept(A) <= 1` convert to SparseMatrixCSC or Array.
Otherwise convert the deepest parents of the nesting and leave the rest as is.
"""
inflate(A::AbstractArray) = inflate1(A)
inflate1(A::AbstractArray) = iswrsparse(A) ? sparsecsc(A) : A isa Array ? A : Array(A)
for ty in (Symmetric, Hermitian)
    @eval function inflate(A::$ty)
        depth(A) == 1 ? inflate1(A) : $ty(inflate(parent(A)), up(A))
    end
end
function inflate(A::SubArray)
    depth(A) == 1 ? inflate1(A) : SubArray(inflate(parent(A)), A.indices)
end
for ty in ( LowerTriangular, UnitLowerTriangular,
            UpperTriangular, UnitUpperTriangular,
            Conjugate, Transpose, Adjoint)

    @eval function inflate(A::$ty)
        depth(A) == 1 ? inflate1(A) : $ty(inflate(parent(A)))
    end
end

"""
    sparsecsc(A::AbstractArray)

Return `A` if it is a `SparseMatrixCSC` or `SparseVector`, otherwise convert to that type
in an efficient manner.
"""
sparsecsc(A::AbstractArray) = sparse(A)
sparsecsc(A::SparseMatrixCSC) = A
sparsecsc(A::SparseVector) = A
sparsecsc(A::UpperTriangular{T,<:AbstractSparseMatrix}) where T = triu(A.data)
sparsecsc(A::UpperTriangular) = sparsecsc(UpperTriangular(sparsecsc(A.data)))
sparsecsc(A::UnitUpperTriangular) = sparsecsc(UnitUpperTriangular(sparsecsc(A.data)))
sparsecsc(A::LowerTriangular{T,<:AbstractSparseMatrix}) where T = tril(A.data)
sparsecsc(A::LowerTriangular) = sparsecsc(LowerTriangular(sparsecsc(A.data)))
sparsecsc(A::UnitLowerTriangular) = sparsecsc(UnitLowerTriangular(sparsecsc(A.data)))
sparsecsc(A::Symmetric) = sparsecsc(Symmetric(sparsecsc(A.data), up(A)))
sparsecsc(A::Hermitian) = sparsecsc(Hermitian(sparsecsc(A.data), up(A)))

sparsecsc(A::Transpose{<:Any,<:AbstractSparseMatrix}) = copy(A)
sparsecsc(A::Transpose) = sparsecsc(Transpose(sparsecsc(A.parent)))
sparsecsc(A::Adjoint{<:Any,<:AbstractSparseMatrix}) = copy(A)
sparsecsc(A::Adjoint) = sparsecsc(Adjoint(sparsecsc(A.parent)))

sparsecsc(A::Transpose{Tv,<:UpperTriangularPlain}) where Tv = _sparse(nzrangeup, transpose, A)
sparsecsc(A::Transpose{Tv,<:LowerTriangularPlain}) where Tv = _sparse(nzrangelo, transpose, A)
sparsecsc(A::Adjoint{Tv,<:UpperTriangularPlain}) where Tv = _sparse(nzrangeup, adjoint, A)
sparsecsc(A::Adjoint{Tv,<:LowerTriangularPlain}) where Tv = _sparse(nzrangelo, adjoint, A)

sparsecsc(A::UnitUpperTriangular{Tv,<:SparseMatrixCSC{Tv}}) where Tv = _sparse(nzrangeup, A, true)
sparsecsc(A::UnitLowerTriangular{Tv,<:SparseMatrixCSC{Tv}}) where Tv = _sparse(nzrangelo, A, true)
function sparsecsc(A::Symmetric{Tv,<:SparseMatrixCSC{Tv}}) where Tv
    _sparse(A.uplo == 'U' ? nzrangeup : nzrangelo, transpose, A)
end
function sparsecsc(A::Hermitian{Tv,<:SparseMatrixCSC{Tv}}) where Tv
    _sparse(A.uplo == 'U' ? nzrangeup : nzrangelo, adjoint, A)
end
sparsecsc(S::SubArray{<:Any,2,<:SparseMatrixCSC}) = getindex(S.parent,S.indices...)
function sparsecsc(S::Conjugate)
    A = sparsecsc(parent(S))
    SparseMatrixCSC(A.m, A.n, A.colptr, A.rowval, conj.(A.nzval))
end

# 2 cases: Unit(Upper|Lower)Triangular{Tv,SparseMatrixCSC}
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
            uplo && (newkk = newk)
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

# 4 cases: (Symmetric|Hermitian) variants (:U|:L)
function _sparse(fnzrange::Function, fadj::Function, sA::SparseMatrixCSCSymmHerm{Tv}) where {Tv}
    A = sA.data
    rowval = rowvals(A)
    nzval = nonzeros(A)
    m, n = size(A)
    Ti = eltype(rowval)
    newcolptr = Vector{Ti}(undef, n+1)
    diagmap = fadj == transpose ? identity : real

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
    nz = newcolptr[n+1] - 1
    newrowval = Vector{Ti}(undef, nz)
    newnzval = Vector{Tv}(undef, nz)
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
                newnzval[newk] = diagmap(nzv)
                newk += 1
            end
        end
        newcolptr[j] = newk
    end
    _sparse_gen(m, n, newcolptr, newrowval, newnzval)
end

# 8 cases: (Transpose|Adjoint){Tv,[Unit](Upper|Lower)Triangular}
function _sparse(fnzrange::Function, fadj::Function, taA::Union{Transpose{Tv,<:AbstractTriangular},Adjoint{Tv,<:AbstractTriangular}}) where {Tv}

    sA = taA.parent
    A = sA.data
    rowval = rowvals(A)
    nzval = nonzeros(A)
    m, n = size(A)
    Ti = eltype(rowval)
    newcolptr = Vector{Ti}(undef, n+1)
    unit = sA isa Union{UnitUpperTriangular,UnitLowerTriangular}
    uplo = A isa Union{UpperTriangular,UnitUpperTriangular}

    fill!(newcolptr, 1unit)
    newcolptr[1] = 1
    @inbounds for j = 1:n
        for k = fnzrange(A, j)
            i = rowval[k]
            if i != j || i == j && !unit
                newcolptr[i+1] += 1
            end
        end
    end
    cumsum!(newcolptr, newcolptr)
    nz = newcolptr[n+1] - 1
    newrowval = Vector{Ti}(undef, nz)
    newnzval = Vector{Tv}(undef, nz)

    @inbounds for j = 1:n
        if !uplo && unit
            ni = newcolptr[j]
            newrowval[ni] = j
            newnzval[ni] = fadj(one(Tv))
            newcolptr[j] = ni + 1
        end
        for k = fnzrange(A, j)
            i = rowval[k]
            nzv = nzval[k]
            if i != j || i == j && !unit
                ni = newcolptr[i]
                newrowval[ni] = j
                newnzval[ni] = fadj(nzv)
                newcolptr[i] = ni + 1
            end
        end
        if uplo && unit
            ni = newcolptr[j]
            newrowval[ni] = j
            newnzval[ni] = fadj(one(Tv))
            newcolptr[j] = ni + 1
        end
    end
    _sparse_gen(n, m, newcolptr, newrowval, newnzval)
end

# not used
# serves as a blueprint for accessing all nonzeros of an AbstractSparseMatrix
function sparsecopy(A::T) where T<:AbstractMatrix{Tv} where Tv
    iswrsparse(A) ? sparsecopy_sparse(A) : SparseMatrixCSC{Tv,indextype(A)}(A)
end

function sparsecopy_sparse(A::AbstractMatrix{Tv}) where {Tv}
    sA = sparseaccess(A)
    rowval = rowvals(sA)
    nzval = nonzeros(sA)
    m, n = size(A)
    nz = nnz_estimation(A)
    Ti = indextype(A)
    newcolptr = Vector{Ti}(undef, n+1)
    newrowval = Vector{Ti}(undef, nz)
    newnzval = Vector{Tv}(undef, nz)
    ip = 1
    for j = 1:n
        newcolptr[j] = ip
        if ip + m - 1 > nz
            nz += max(nz, m)
            resize!(newrowval, nz)
            resize!(newnzval, nz)
        end
        for k in nzrange(sA, j)
            newrowval[ip] = rowval[k]
            newnzval[ip] = nzval[k]
            ip += 1
        end
    end
    newcolptr[n+1] = ip
    ip -= 1
    resize!(newrowval, ip)
    resize!(newnzval, ip)
    SparseMatrixCSC(m, n, newcolptr, newrowval, newnzval)
end

function _sparse_gen(m, n, newcolptr, newrowval, newnzval)
    @inbounds for j = n:-1:1
        newcolptr[j+1] = newcolptr[j]
    end
    newcolptr[1] = 1
    SparseMatrixCSC(m, n, newcolptr, newrowval, newnzval)
end

