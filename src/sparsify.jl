
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

indextype(::T) where T<:AbstractArray = indextype(T)
indextype(T::Type) = walk_wrapper(_indextype, T)
_indextype(::Type{<:AbstractSparseArray{<:Any,Ti}}) where Ti = Ti
_indextype(::Type) = Int

nnz_estimation(A::AbstractMatrix) = walk_wrapper(_nnz_estimation, A)
_nnz_estimation(::Any) = missing
_nnz_estimation(A::SparseMatrixCSC) = nnz(A)
_nnz_estimation(A::SparseVector) = nnz(A)
_nnz_estimation(A::Array) = length(A)

depth(::T) where T<:AbstractArray = depth(T)
depth(T::Type) = walk_wrapper(_depth, T, x->x+1)
_depth(::Type) = 0

array_storage(::T) where T<:AbstractArray = array_storage(T)
array_storage(T::Type) = walk_wrapper(_array_storage, T)
_array_storage(T::Type{<:AbstractArray}) = T
_array_storage(T::Type) = Nothing

import LinearAlgebra: Symmetric, Hermitian, LowerTriangular, UnitLowerTriangular
import LinearAlgebra: UpperTriangular, UnitUpperTriangular, Transpose, Adjoint

isupper(A::AbstractArray) = walk_wrapper(_isupper, A)
_isupper(::Any) = missing
_isupper(::Union{UpperTriangular,UnitUpperTriangular}) = true
_isupper(A::Union{Symmetric,Hermitian}) = A.uplo == 'U'
_islower(::Union{LowerTriangular,UnitLowerTriangular}) = true
_islower(A::Union{Symmetric,Hermitian}) = A.uplo == 'L'

walk_wrapper(f::Function, x::Any, g::Function=identity) = f(x)
for wr in (Symmetric, Hermitian,
           LowerTriangular, UnitLowerTriangular, UpperTriangular, UnitUpperTriangular,
           Transpose, Adjoint,
           SubArray, Conjugate,
           Diagonal, Bidiagonal, Tridiagonal, SymTridiagonal, HermiteTridiagonal )

    pl = wr === SubArray ? :($wr{<:Any,<:Any,T}) : :($wr{<:Any,T})
    @eval function walk_wrapper(f::Function, ::Type{<:$pl}, g::Function=identity) where T
        g(walk_wrapper(f, T, g))
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
inflate(A::AbstractArray) = _inflate(A)
_inflate(A::AbstractArray) = iswrsparse(A) ? sparsecsc(A) : A isa Array ? A : Array(A)
for ty in (Symmetric, Hermitian)
    @eval function inflate(A::$ty)
        depth(A) == 1 ? _inflate(A) : $ty(inflate(parent(A)), up(A))
    end
end
function inflate(A::SubArray)
    depth(A) == 1 ? _inflate(A) : SubArray(inflate(parent(A)), A.indices)
end
for ty in ( LowerTriangular, UnitLowerTriangular,
            UpperTriangular, UnitUpperTriangular,
            Conjugate, Transpose, Adjoint)

    @eval function inflate(A::$ty)
        depth(A) == 1 ? _inflate(A) : $ty(inflate(parent(A)))
    end
end
for ty in (Diagonal, Bidiagonal, Tridiagonal, SymTridiagonal, HermiteTridiagonal)
    @eval inflate(A::$ty) = dropzeros!(sparse(A))
end

import SparseArrays.SparseMatrixCSC
for wr in (Symmetric, Hermitian,
           UpperTriangular, LowerTriangular, UnitUpperTriangular, UnitLowerTriangular,
           Transpose, Adjoint, SubArray, Conjugate,
           Bidiagonal, Tridiagonal, SymTridiagonal, HermiteTridiagonal)

    @eval SparseMatrixCSC(A::$wr) = sparsecsc(A)
    @eval SparseMatrixCSC{Tv}(A::$wr{Tv}) where Tv = sparsecsc(A)
    @eval SparseMatrixCSC{Tv}(A::$wr) where Tv = SparseMatrixCSC{Tv}(sparsecsc(A))
    @eval SparseMatrixCSC{Tv,Ti}(A::$wr) where {Tv,Ti} = SparseMatrixCSC{Tv,Ti}(sparsecsc(A))
end

"""
    unwrap(A::AbstractMatrix)

In case A is a wrapper type (`SubArray, Symmetric, Adjoint, SubArray, Triangular, Tridiagonal`, etc.)
convert to `Matrix` or `SparseMatrixCSC`, depending on final storage type of A.
For other types return A itself.
"""
unwrap(A::AbstractArray) = iswrsparse(A) ? convert(SparseMatrixCSC, A) : convert(Array, A)
unwrap(A::Any) = A

import Base.copy
copy(A::SubArray) = getindex(unwrap(parent(A)), A.indices...)

"""
    sparsecsc(A::AbstractArray)

Return `A` if it is a `SparseMatrixCSC` or `SparseVector`, otherwise convert to that type
in an efficient manner.
"""
function sparsecsc(@nospecialize A::AbstractArray)
    if iswrsparse(A)
        if depth(A) >= 1
            sparsecsc(inflate(A))
        else
            A
        end
    else
        Tv = eltype(A)
        invoke(SparseMatrixCSC{Tv,Int}, Tuple{AbstractMatrix}, A)
    end
end

sparsecsc(A::SparseVector) = A
sparsecsc(A::UpperTriangular{T,<:AbstractSparseMatrix}) where T = triu(A.data)
sparsecsc(A::LowerTriangular{T,<:AbstractSparseMatrix}) where T = tril(A.data)
function sparsecsc(A::Transpose{<:Any,<:AbstractSparseVector})
    B = parent(A);
    copy(reshape(transpose.(B), 1, size(B,1)))
end
function sparsecsc(A::Adjoint{<:Any,<:AbstractSparseVector})
    B = parent(A);
    copy(reshape(adjoint.(B), 1, size(B,1)))
end

sparsecsc(A::Transpose{<:Any,<:UpperTriangularPlain}) = _sparse(nzrangeup, transpose, A)
sparsecsc(A::Transpose{<:Any,<:LowerTriangularPlain}) = _sparse(nzrangelo, transpose, A)
sparsecsc(A::Adjoint{<:Any,<:UpperTriangularPlain}) = _sparse(nzrangeup, adjoint, A)
sparsecsc(A::Adjoint{<:Any,<:LowerTriangularPlain}) = _sparse(nzrangelo, adjoint, A)

function sparsecsc(A::AbstractTriangular{<:Any,<:SparseMatrixCSC})
    _sparse(nzrangeup, A)
end
function sparsecsc(A::Symmetric{<:Any,<:SparseMatrixCSC})
    _sparse(A.uplo == 'U' ? nzrangeup : nzrangelo, transpose, A)
end
function sparsecsc(A::Hermitian{<:Any,<:SparseMatrixCSC})
    _sparse(A.uplo == 'U' ? nzrangeup : nzrangelo, adjoint, A)
end
function sparsecsc(S::SubArray{<:Any,2,<:SparseMatrixCSC})
    getindex(S.parent,S.indices...)
end
function sparsecsc(S::Conjugate{<:Any,<:SparseMatrixCSC{Tv,Ti}}) where {Tv,Ti}
    A = parent(S)
    SparseMatrixCSC{Tv,Ti}(A.m, A.n, A.colptr, A.rowval, conj.(A.nzval))
end

# 2 cases: Unit(Upper|Lower)Triangular{Tv,SparseMatrixCSC}
function _sparse(fnzrange::Function, A::AbstractTriangular{Tv}) where {Tv}
    S = A.data
    rowval = rowvals(S)
    nzval = nonzeros(S)
    m, n = size(S)
    Ti = eltype(rowval)
    unit = sA isa Union{UnitUpperTriangular,UnitLowerTriangular}
    newcolptr = Vector{Ti}(undef, n+1)
    newrowval = Vector{Ti}(undef, nnz(S))
    newnzval = Vector{Tv}(undef, nnz(S))
    newcolptr[1] = 1
    uplo = fnzrange == nzrangeup
    newk = 1
    @inbounds for j = 1:n
        newkk = newk
        if unit
            newk += !uplo
        end
        r = fnzrange(S, j); r1 = r.start; r2 = r.stop
        for k = r1:r2
            i = rowval[k]
            if i != j || i == j && !unit
                newrowval[newk] = i
                newnzval[newk] = nzval[k]
                newk += 1
            end
        end
        if unit
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

