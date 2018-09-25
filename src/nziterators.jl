
struct NzIterator{T<:AbstractArray}
    filter::Function
    output::Function
    A::T
end

nziterator(A::AbstractMatrix) = NzIterator(A)

NzIterator(A::AbstractMatrix) = NzIterator(_f_all, identity, A)
NzIterator(A::UpperTriangular) = NzIterator(_f_upper, identity, A.data)
NzIterator(A::LowerTriangular) = NzIterator(_f_lower, identity, A.data)
NzIterator(A::Transpose) = NzIterator(_f_all, _out_transpose, A.parent)
NzIterator(A::Adjoint) = NzIterator(_f_all, _out_adjoint, A.parent)
NzIterator(A::Symmetric) = _nziterator(A, identity, _out_transpose)
NzIterator(A::Hermitian) = _nziterator(A, _out_realdiag, _out_adjoint)

_out_transpose((i, j, v)) = (j, i, transpose(v))
_out_adjoint((i, j, v)) = (j, i, adjoint(v))
_out_realdiag((i, j, v)) = (i, j, i == j ? real(v) : adjoint(v))
_f_all(i, j) = true
_f_upper(i,j) = i <= j
_f_upper_strict(i,j) = i < j
_f_lower(i,j) = i >= j
_f_lower_strict(i,j) = i > j

function _nziterator(A::Union{Symmetric,Hermitian}, out1::Function, out2::Function)
    filter1 = A.uplo == 'U' ? _f_upper : _f_lower
    filter2 = A.uplo == 'U' ? _f_upper_strict : _f_lower_strict

    Base.Iterators.flatten((NzIterator(filter1, out1, A.data),
                            NzIterator(filter2, out2, A.data)))
end

import Base: iterate

Base.IteratorSize(::Type{<:NzIterator}) = Base.SizeUnknown()
Base.show(io::IO, itr::NzIterator) = println(io, "Non-zero Iterator for $(typeof(itr.A))")

function iterate(itr::NzIterator{<:SparseMatrixCSC{Tv,Ti}}) where {Tv,Ti}
    A = itr.A
    m, n = size(A)
    m > 0 && n > 0 && nnz(A) > 0 || return nothing
    colptr = A.colptr
    rowval = A.rowval
    j = 1
    while colptr[j+1] <= 1
        j += 1
        j > n && break
    end
    j > n && return nothing
    iterate(itr, (0, colptr[j+1], j, n))
end
function iterate(itr::NzIterator{<:SparseMatrixCSC{Tv,Ti}}, (k, kmax, j, n)::NTuple{4,Ti}) where {Ti,Tv}

    colptr = itr.A.colptr
    rowval = itr.A.rowval
    nzval = itr.A.nzval
    filter = itr.filter
    while j <= n 
        k += 1
        while k >= kmax
            j += 1
            j > n && return nothing
            kmax = colptr[j+1]
        end
        i = rowval[k]
        nzv = nzval[k]
        nzv != 0 && filter(i,j) && return itr.output((i, j, nzv)), (k, kmax, j, n)
    end
    nothing
end
    
function iterate(itr::NzIterator{<:DenseMatrix})
    m, n = size(itr.A)
    m > 0 && n > 0 || return nothing
    iterate(itr, (0, m, 1, n))
end
function iterate(itr::NzIterator{<:DenseMatrix}, (i, m, j, n)::NTuple{4,Int})
    filter = itr.filter
    while j <= n
        i += 1
        if i > m
            j += 1
            i = 0
        else
            nzv = itr.A[i,j]
            nzv != 0 && filter(i, j) && return itr.output((i, j, nzv)), (i, m, j, n)
        end
    end
    nothing
end

function iterate(itr::NzIterator{<:Union{Diagonal,Bidiagonal,Tridiagonal,SymTridiagonal}})
    n = size(itr.A, 1)
    iterate(itr, (0, n))
end
function iterate(itr::NzIterator{<:Diagonal}, (i, n)::Tuple{Int,Int})
    while (i += 1) <= n
        nzv = itr.A.diag[i]
        nzv != 0 && filter(i, i) && return itr.output((i, i, nzv)), (i, n)
    end
end

function iterate(itr::NzIterator{<:Bidiagonal}, (i, n)::Tuple{Int,Int})
    k = itr.A.uplo == 'U' ? 1 : 0
    while (i += 1) <= 2n-1
        nzv, ii, jj = i <= n ? (itr.A.dv[i], i, i) : (itr.A.ev[i-n], i+1-k-n, i+k-n)
        nzv != 0 && itr.filter(ii, jj) && return itr.output((ii, jj, nzv)), (i, n)
    end
end

function iterate(itr::NzIterator{<:SymTridiagonal}, (i, n)::Tuple{Int,Int})
    while (i += 1) <= 3n-2
        nzv, ii, jj =
            i <= n ? (itr.A.dv[i], i, i) :
            i < 2n ? (itr.A.ev[i-n], i-n, i+1-n) :
                     (itr.A.ev[i-2n+1], i+2-2n, i-2n+1)

        nzv != 0 && itr.filter(ii, jj) && return itr.output((ii, jj, nzv)), (i, n)
    end
end

function iterate(itr::NzIterator{<:Tridiagonal}, (i, n)::Tuple{Int,Int})
    while (i += 1) <= 3n-2
        nzv, ii, jj =
            i <= n ? (itr.A.d[i], i, i) :
            i < 2n ? (itr.A.du[i-n], i-n, i+1-n) :
                     (itr.A.dl[i-2n+1], i-2n+2, i-2n+1)

        nzv != 0 && itr.filter(ii, jj) && return itr.output((ii, jj, nzv)), (i, n)
    end
end

function iterate(itr::NzIterator{<:AbstractMatrix})
    stack = NzIterator(itr.A)
    r = iterate(stack)
    _iterate(itr, r, stack)
end
function iterate(itr::NzIterator{<:AbstractMatrix}, (s, stack))
    r = iterate(stack, s)
    _iterate(itr, r, stack)
end
function _iterate(itr::NzIterator, r::Union{Nothing,Tuple}, stack)
    while r !== nothing
        (i, j, nz), s = r
        itr.filter(i, j) && return itr.output((i, j, nz)), (s, stack)
        r = iterate(stack, s)
    end
    nothing
end
