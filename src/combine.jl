
struct UniversalWrapper{T,S,F,C,D} <:AbstractMatrix{T}
    parent::S
end
indexflip(A::UniversalWrapper, i, j) = (i, j)
indexflip(A::UniversalWrapper{<:Any,<:Any,Union{Transpose,Adjoint}}, i, j) = (j, i)

mapper(::Type{<:Real},::Val) = identity
mapper(::Type{<:Complex},::Val{0}) = identity
mapper(::Type{<:Complex},::Val{1}) = conj
mapper(A::UniversalWrapper{T,<:Any,<:Any,C,:N}, i, j) where {T,C} = mapper(T, Val(C))
mapper(A::UniversalWrapper{T,<:Any,<:Any,C,:O}, i, j) where {T,C} = aij -> i == j ? one(T) : mapper(T, Val(C))(aij)
mapper(A::UniversalWrapper{T,<:Any,<:Any,C,:Z}, i, j) where {T,C} = aij -> i == j ? zero(T) : mapper(T, Val(C))(aij)
mapper(A::UniversalWrapper{T,<:Any,<:Any,C,:R}, i, j) where {T,C} = aij -> i == j ? real(aij) : mapper(T, Val(C))(aij)

Base.size(A::UniversalWrapper) = indexflip(A, size(A.parent))
Base.getindex(A::UniversalWrapper, i, j) = mapper(A, i, j)(getindex(A.parent, indexflip(A, i, j)...))

struct Conjugate{T,S} <:AbstractMatrix{T}
    parent::S
    Conjugate(A::AbstractMatrix{T}) where T = new{T,typeof(A)}(A)
end
Base.size(A::Conjugate) = size(A.parent)
Base.getindex(A::Conjugate,ind...) = conj.(getindex(A.parent, ind...))

struct HermiteTridiagonal{T,V<:AbstractVector{T}} <: AbstractMatrix{T}
    dv::V
    ev::V
end

"""
    combine wrappers of the following kinds to achieve

  1. try to reduce number of involved wrappers
  2. standardized sequence of combination
"""
WRAPPER_TYPES = [
                 Conjugate, Transpose, Adjoint, Symmetric, Hermitian,
                 UpperTriangular, LowerTriangular, UnitUpperTriangular, UnitLowerTriangular,
                 Diagonal, Bidiagonal, Tridiagonal, SymTridiagonal, HermiteTridiagonal]

up(B::Union{Symmetric,Hermitian,Bidiagonal}) = Symbol(B.uplo)
toggle(s::Symbol) = s == :U ? :L : :U

import LinearAlgebra: Transpose, Adjoint, Symmetric, Hermitian

# missing types: HermiteTridiagonal, Conjugate to be defined
Conjugate(B::Conjugate) = B.parent
Conjugate(B::Transpose) = Adjoint(B.parent)
Conjugate(B::Adjoint) = Transpose(B.parent)
# Conjugate(B::Symmetric) =
# Conjugate(B::Hermitian) =
# Conjugate(B::Diagonal) = conj(B)
# Conjugate(B::UpperTriangular) =
# Conjugate(B::LowerTriangular) =

Transpose(B::Conjugate) = Adjoint(B.parent)
Transpose(B::Transpose) = B.parent
Transpose(B::Adjoint) = Conjugate(B.parent)
Transpose(B::Symmetric) = B
Transpose(B::Hermitian) = Conjugate(B)
Transpose(B::Diagonal) = B
# Transpose(B::UpperTriangular) =
# Transpose(B::LowerTriangular) =

Adjoint(B::Conjugate) = Transpose(B.parent)
Adjoint(B::Transpose) = Conjugate(B.parent)
Adjoint(B::Adjoint) = B.parent
Adjoint(B::Symmetric) = Conjugate(B)
Adjoint(B::Hermitian) = B
Adjoint(B::Diagonal) = Conjugate(B)
# Adjoint(B::UpperTriangular) =
# Adjoint(B::LowerTriangular) =

Symmetric(B::Conjugate, uplo::Symbol=:U) = Conjugate(Symmetric(B.parent, uplo))
Symmetric(B::Transpose, uplo::Symbol=:U) = Symmetric(B.parent, toggle(uplo))
Symmetric(B::Adjoint, uplo::Symbol=:U) = Conjugate(Symmetric(B.parent, toggle(uplo)))
Symmetric(B::Symmetric, uplo::Symbol=:U) = B
function Symmetric(B::Hermitian, uplo::Symbol=:U)
    base = isreal(diag(B.data)) ? B.data : hermtotria(B)
    uplo == up(B) ? Symmetric(base, uplo) : Conjugate(Symmetric(base, up(B)))
end
Symmetric(B::Diagonal, uplo::Symbol=:U) = B
Symmetric(B::UpperTriangular, uplo::Symbol=:U) = uplo == :U ? Symmetric(B.data, :U) : Diagonal(diag(B.data))
Symmetric(B::LowerTriangular, uplo::Symbol=:L) = uplo == :L ? Symmetric(B.data, :L) : Diagonal(diag(B.data))
Symmetric(B::UnitUpperTriangular, uplo::Symbol=:U) = uplo == :U ? Symmetric(sparsecsc(B), :U) : Diagonal(one.(diag(B.data)))
Symmetric(B::UnitLowerTriangular, uplo::Symbol=:L) = uplo == :L ? Symmetric(sparsecsc(B), :L) : Diagonal(one.(real(diag(B.data))))

Hermitian(B::Conjugate, uplo::Symbol=:U) = Conjugate(Hermitian(B.parent, uplo))
Hermitian(B::Transpose, uplo::Symbol=:U) = Conjugate(Hermitian(B.parent, toggle(uplo)))
Hermitian(B::Adjoint, uplo::Symbol=:U) = Hermitian(B.parent, toggle(uplo))
Hermitian(B::Symmetric, uplo::Symbol=:U) = uplo == up(B) ? Hermitian(B.data, uplo) : Conjugate(Hermitian(B.data, up(B)))
Hermitian(B::Hermitian, uplo::Symbol=:U) = B
Hermitian(B::Diagonal, uplo::Symbol=:U) = real(B)
Hermitian(B::UpperTriangular, uplo::Symbol=:U) = uplo == :U ? Hermitian(B.data, :U) : Diagonal(real(diag(B.data)))
Hermitian(B::LowerTriangular, uplo::Symbol=:L) = uplo == :L ? Hermitian(B.data, :L) : Diagonal(real(diag(B.data)))
Hermitian(B::UnitUpperTriangular, uplo::Symbol=:U) = uplo == :U ? Hermitian(sparsecsc(B), :U) : Diagonal(one.(diag(B.data)))
Hermitian(B::UnitLowerTriangular, uplo::Symbol=:L) = uplo == :L ? Hermitian(sparsecsc(B), :L) : Diagonal(one.(real(diag(B.data))))

UpperTriangular(B::Conjugate) = Conjugate(UpperTriangular(B.parent))
UpperTriangular(B::Transpose) = Transpose(LowerTriangular(B.parent))
UpperTriangular(B::Adjoint) = Adjoint(LowerTriangular(B.parent))
UpperTriangular(B::Symmetric) = up(B) == :U ? UpperTriangular(B.data) : Transpose(LowerTriangular(B.data))
function UpperTriangular(B::Hermitian)
    base = isreal(diag(B.data)) ? B.data : hermtotria(B)
    up(B) == :U ? UpperTriangular(base) : Adjoint(LowerTriangular(base))
end
UpperTriangular(B::Diagonal) = B
## UpperTriangular(B::UpperTriangular) = B
UpperTriangular(B::LowerTriangular) = Diagonal(diag(B))
UpperTriangular(B::UnitUpperTriangular) = B
UpperTriangular(B::UnitLowerTriangular) = Diagonal(one.(diag(B.data)))

LowerTriangular(B::Conjugate) = Conjugate(LowerTriangular(B.parent))
LowerTriangular(B::Transpose) = Transpose(UpperTriangular(B.parent))
LowerTriangular(B::Adjoint) = Adjoint(UpperTriangular(B.parent))
LowerTriangular(B::Symmetric) = up(B) == :L ? LowerTriangular(B.data) : Transpose(UpperTriangular(B.data))
function LowerTriangular(B::Hermitian)
    base = isreal(diag(B.data)) ? B.data : hermtotria(B)
    up(B) == :L ? LowerTriangular(base) : Adjoint(UpperTriangular(base))
end
LowerTriangular(B::Diagonal) = B
## LowerTriangular(B::LowerTriangular) = B
LowerTriangular(B::UpperTriangular) = Diagonal(diag(B.data))
LowerTriangular(B::UnitLowerTriangular) = B
LowerTriangular(B::UnitUpperTriangular) = Diagonal(one.(diag(B.data)))

UnitUpperTriangular(B::Conjugate) = Conjugate(UnitUpperTriangular(B.parent))
UnitUpperTriangular(B::Transpose) = Transpose(UnitLowerTriangular(B.parent))
UnitUpperTriangular(B::Adjoint) = Adjoint(UnitLowerTriangular(B.parent))
UnitUpperTriangular(B::Symmetric) = up(B) == :U ? UnitUpperTriangular(B.data) : Transpose(UnitLowerTriangular(B.data))
function UnitUpperTriangular(B::Hermitian)
    base = isreal(diag(B.data)) ? B.data : hermtotria(B)
    up(B) == :U ? UnitUpperTriangular(base) : Adjoint(UnitLowerTriangular(base))
end
UnitUpperTriangular(B::Diagonal) = B
## UnitUpperTriangular(B::UnitUpperTriangular) = B
UnitUpperTriangular(B::UnitLowerTriangular) = Diagonal(one.(diag(B.data)))
UnitUpperTriangular(B::UpperTriangular) = UnitUpperTriangular(B.data)
UnitUpperTriangular(B::LowerTriangular) = Diagonal(one.(diag(B.data)))

UnitLowerTriangular(B::Conjugate) = Conjugate(UnitLowerTriangular(B.parent))
UnitLowerTriangular(B::Transpose) = Transpose(UnitUpperTriangular(B.parent))
UnitLowerTriangular(B::Adjoint) = Adjoint(UnitUpperTriangular(B.parent))
UnitLowerTriangular(B::Symmetric) = up(B) == :L ? UnitLowerTriangular(B.data) : Transpose(UnitUpperTriangular(B.data))
function UnitLowerTriangular(B::Hermitian)
    base = isreal(diag(B.data)) ? B.data : hermtotria(B)
    up(B) == :L ? UnitLowerTriangular(base) : Adjoint(UnitUpperTriangular(base))
end
UnitLowerTriangular(B::Diagonal) = B
## UnitLowerTriangular(B::UnitLowerTriangular) = B
UnitLowerTriangular(B::UnitUpperTriangular) = Diagonal(one.(diag(B.data)))
UnitLowerTriangular(B::LowerTriangular) = UnitLowerTriangular(B.data)
UnitLowerTriangular(B::UpperTriangular) = Diagonal(one.(diag(B.data)))

Diagonal(B::Conjugate) = Conjugate(Diagonal(diag(B.parent)))
Diagonal(B::Transpose) = Diagonal(diag(B.parent))
Diagonal(B::Adjoint) = Conjugate(Diagonal(diag(B.parent)))
Diagonal(B::Symmetric) = Diagonal(diag(B.data))
Diagonal(B::Hermitian) = Diagonal(real(diag(B.data)))
## Diagonal(B::Diagonal) = B
Diagonal(B::UpperTriangular) = Diagonal(diag(B.data))
Diagonal(B::LowerTriangular) = Diagonal(diag(B.data))
Diagonal(B::UnitUpperTriangular) = Diagonal(one.(diag(B.data)))
Diagonal(B::UnitLowerTriangular) = Diagonal(one.(diag(B.data)))

function hermtotria(B::Hermitian)
    A = sparsecsc((up(B) == :U ? UpperTriangular : LowerTriangular)(B.data))
    for i = 1:size(A, 1)
        A[i,i] = real(A[i,i])
    end
    A
end

#
# verification of type depth of combinations of wrappers.
# does the combiniation of wrappers define a finite group? 
#
wrappertype(A) = begin wr = _wrt(A); isempty(wr) ? [identity] : wr end
_wr(A::T) where T = T.name.wrapper

_wrt(A::Union{Conjugate,Transpose,Adjoint}) = Any[ _wr(A); _wrt(A.parent)]
_wrt(A::Union{Symmetric,Hermitian}) = Any[ (_wr(A), up(A)); _wrt(A.data)]
_wrt(A::Union{UpperTriangular,LowerTriangular}) = Any[ _wr(A); _wrt(A.data)]
_wrt(A::Union{UnitUpperTriangular,UnitLowerTriangular}) = Any[ _wr(A); _wrt(A.data)]
_wrt(A::Union{Diagonal,Tridiagonal,SymTridiagonal}) = Any[ _wr(A) ]
_wrt(A::Bidiagonal) = Any[ (_wr(A), up(A)) ]
_wrt(A::Any) = []

depth(A::Union{Conjugate,Transpose,Adjoint}) = depth(A.parent) + 1
depth(A::Union{Symmetric,Hermitian}) = depth(A.data) + 1
depth(A::Union{UpperTriangular,LowerTriangular}) = depth(A.data) + 1
depth(A::Diagonal) = 1
depth(::Any) = 0

apply(T) = A -> T(A)
apply(t::Tuple) = A -> t[1](A, t[2])
apply(a::Vector) = A -> (x = apply(a[end])(A); length(a) == 1 ? x : apply(a[1:end-1])(x))

WRAPPERS_BASE = [identity, Transpose, Adjoint, Conjugate,
                (Symmetric, :U), (Symmetric, :L), (Hermitian, :U), (Hermitian, :L),
                UpperTriangular, LowerTriangular,
                UnitUpperTriangular, UnitLowerTriangular, Diagonal
               ]

WRAPPERS_COMPLETE = [WRAPPERS_BASE...,
                     Any[Transpose, UpperTriangular], Any[Transpose, LowerTriangular],
                     Any[Transpose, UnitUpperTriangular], Any[Transpose, UnitLowerTriangular],
                     Any[Adjoint, UpperTriangular], Any[Adjoint, LowerTriangular],
                     Any[Adjoint, UnitUpperTriangular], Any[Adjoint, UnitLowerTriangular],
                     Any[Conjugate, UpperTriangular], Any[Conjugate, LowerTriangular],
                     Any[Conjugate, UnitUpperTriangular], Any[Conjugate, UnitLowerTriangular],
                     [Conjugate, (Symmetric, :U)], [Conjugate, (Symmetric, :L)],
                     [Conjugate, (Hermitian, :U)], [Conjugate, (Hermitian, :L)],
                     Any[Conjugate, Diagonal]
                    ]

