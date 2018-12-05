export symmetric, hermitian, conjugate, diagonal, upper_triangular, lower_triangular
export unit_upper_triangular, unit_lower_triangular

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

"""
    combine wrappers of the following kinds to achieve

  1. try to reduce number of involved wrappers
  2. standardized sequence of combination
"""
WRAPPER_TYPES = [
                 Conjugate, Transpose, Adjoint, Symmetric, Hermitian,
                 UpperTriangular, LowerTriangular, UnitUpperTriangular, UnitLowerTriangular,
                 Diagonal, Bidiagonal, Tridiagonal, SymTridiagonal, HermiteTridiagonal]


import LinearAlgebra: Transpose, Adjoint, Symmetric, Hermitian
import LinearAlgebra: transpose, adjoint, symmetric, hermitian

# missing types: HermiteTridiagonal, Conjugate to be defined
conjugate(B::Conjugate) = B.parent
conjugate(B::Transpose) = adjoint(B.parent)
conjugate(B::Adjoint) = transpose(B.parent)
conjugate(B::Diagonal) = Diagonal(conj.(B.diag))
conjugate(B::AbstractMatrix{<:Real}) = B
conjugate(B::AbstractMatrix) = Conjugate(B)

transpose(B::Conjugate) = adjoint(B.parent)
## transpose(B::Transpose) = B.parent
transpose(B::Adjoint) = conjugate(B.parent)
# transpose(B::Symmetric) = B
# transpose(B::Hermitian) = conjugate(B)
# transpose(B::Diagonal) = B
## transpose(B::AbstractMatrix) = Transpose(B)

adjoint(B::Conjugate) = transpose(B.parent)
adjoint(B::Transpose) = conjugate(B.parent)
# adjoint(B::Adjoint) = B.parent
# adjoint(B::Symmetric) = conjugate(B)
## adjoint(B::Hermitian) = B
# adjoint(B::Diagonal) = conjugate(B)
## adjoint(B::AbstractMatrix) = Adjoint(B)

symmetric(B::Conjugate, uplo::Symbol=:U) = conjugate(symmetric(B.parent, uplo))
symmetric(B::Transpose, uplo::Symbol=:U) = symmetric(B.parent, toggle(uplo))
symmetric(B::Adjoint, uplo::Symbol=:U) = conjugate(symmetric(B.parent, toggle(uplo)))
## symmetric(B::Symmetric, uplo::Symbol=:U) = B
function symmetric(B::Hermitian, uplo::Symbol=:U)
    base = isreal(diag(B.data)) ? B.data : hermtotria(B)
    uplo == up(B) ? symmetric(base, uplo) : conjugate(symmetric(base, up(B)))
end
symmetric(B::Diagonal, uplo::Symbol=:U) = B
symmetric(B::UpperTriangular, uplo::Symbol=:U) = uplo == :U ? symmetric(B.data, :U) : Diagonal(diag(B.data))
symmetric(B::LowerTriangular, uplo::Symbol=:L) = uplo == :L ? symmetric(B.data, :L) : Diagonal(diag(B.data))
symmetric(B::UnitUpperTriangular, uplo::Symbol=:U) = uplo == :U ? symmetric(sparsecsc(B), :U) : Diagonal(oneunit.(diag(B.data)))
symmetric(B::UnitLowerTriangular, uplo::Symbol=:L) = uplo == :L ? symmetric(sparsecsc(B), :L) : Diagonal(oneunit.(real(diag(B.data))))
symmetric(B::Hermitian{<:Real}, uplo::Symbol=:U) = Symmetric(B.data, uplo)
# symmetric(B::AbstractMatrix, uplo::Symbol=:U) = Symmetric(B, uplo)

hermitian(B::Conjugate, uplo::Symbol=:U) = conjugate(hermitian(B.parent, uplo))
hermitian(B::Transpose, uplo::Symbol=:U) = conjugate(hermitian(B.parent, toggle(uplo)))
hermitian(B::Adjoint, uplo::Symbol=:U) = hermitian(B.parent, toggle(uplo))
hermitian(B::Symmetric, uplo::Symbol=:U) = uplo == up(B) ? hermitian(B.data, uplo) : conjugate(hermitian(B.data, up(B)))
hermitian(B::Hermitian, uplo::Symbol=:U) = B
hermitian(B::Diagonal, uplo::Symbol=:U) = real(B)
hermitian(B::UpperTriangular, uplo::Symbol=:U) = uplo == :U ? hermitian(B.data, :U) : Diagonal(real(diag(B.data)))
hermitian(B::LowerTriangular, uplo::Symbol=:L) = uplo == :L ? hermitian(B.data, :L) : Diagonal(real(diag(B.data)))
hermitian(B::UnitUpperTriangular, uplo::Symbol=:U) = uplo == :U ? hermitian(sparsecsc(B), :U) : Diagonal(oneunit.(diag(B.data)))
hermitian(B::UnitLowerTriangular, uplo::Symbol=:L) = uplo == :L ? hermitian(sparsecsc(B), :L) : Diagonal(oneunit.(real(diag(B.data))))
hermitian(B::AbstractMatrix{<:Real}, uplo::Symbol=:U) = symmetric(B, uplo)
# hermitian(B::AbstractMatrix, uplo::Symbol=:U) = Hermitian(B, uplo)

upper_triangular(B::Conjugate) = conjugate(upper_triangular(B.parent))
upper_triangular(B::Transpose) = transpose(lower_triangular(B.parent))
upper_triangular(B::Adjoint) = adjoint(lower_triangular(B.parent))
upper_triangular(B::Symmetric) = up(B) == :U ? upper_triangular(B.data) : transpose(lower_triangular(B.data))
function upper_triangular(B::Hermitian)
    base = isreal(diag(B.data)) ? B.data : hermtotria(B)
    up(B) == :U ? upper_triangular(base) : adjoint(lower_triangular(base))
end
upper_triangular(B::Diagonal) = B
upper_triangular(B::UpperTriangular) = B
upper_triangular(B::LowerTriangular) = Diagonal(diag(B))
upper_triangular(B::UnitUpperTriangular) = B
upper_triangular(B::UnitLowerTriangular) = Diagonal(oneunit.(diag(B.data)))
upper_triangular(B::AbstractMatrix) = UpperTriangular(B)

lower_triangular(B::Conjugate) = conjugate(lower_triangular(B.parent))
lower_triangular(B::Transpose) = transpose(upper_triangular(B.parent))
lower_triangular(B::Adjoint) = Adjoint(upper_triangular(B.parent))
lower_triangular(B::Symmetric) = up(B) == :L ? lower_triangular(B.data) : transpose(upper_triangular(B.data))
function lower_triangular(B::Hermitian)
    base = isreal(diag(B.data)) ? B.data : hermtotria(B)
    up(B) == :L ? lower_triangular(base) : adjoint(upper_triangular(base))
end
lower_triangular(B::Diagonal) = B
lower_Triangular(B::LowerTriangular) = B
lower_triangular(B::UpperTriangular) = Diagonal(diag(B.data))
lower_triangular(B::UnitLowerTriangular) = B
lower_triangular(B::UnitUpperTriangular) = Diagonal(oneunit.(diag(B.data)))
lower_triangular(B::AbstractMatrix) = LowerTriangular(B)

unit_upper_triangular(B::Conjugate) = Conjugate(unit_upper_triangular(B.parent))
unit_upper_triangular(B::Transpose) = Transpose(UnitLowerTriangular(B.parent))
unit_upper_triangular(B::Adjoint) = Adjoint(UnitLowerTriangular(B.parent))
unit_upper_triangular(B::Symmetric) = up(B) == :U ? unit_upper_triangular(B.data) : transpose(unit_lower_triangular(B.data))
function unit_upper_triangular(B::Hermitian)
    base = isreal(diag(B.data)) ? B.data : hermtotria(B)
    up(B) == :U ? unit_upper_triangular(base) : adjoint(unit_lower_triangular(base))
end
unit_upper_triangular(B::Diagonal) = Diagonal(oneunit.(diag(B)))
unit_upper_triangular(B::UnitUpperTriangular) = B
unit_upper_triangular(B::UnitLowerTriangular) = Diagonal(oneunit.(diag(B.data)))
unit_upper_triangular(B::UpperTriangular) = unit_upper_triangular(B.data)
unit_upper_triangular(B::LowerTriangular) = Diagonal(oneunit.(diag(B.data)))
unit_upper_triangular(B::AbstractMatrix) = UnitUpperTriangular(B)

unit_lower_triangular(B::Conjugate) = Conjugate(unit_lower_triangular(B.parent))
unit_lower_triangular(B::Transpose) = Transpose(unit_upper_triangular(B.parent))
unit_lower_triangular(B::Adjoint) = Adjoint(unit_upper_triangular(B.parent))
unit_lower_triangular(B::Symmetric) = up(B) == :L ? unit_lower_triangular(B.data) : Transpose(unit_upper_triangular(B.data))
function unit_lower_triangular(B::Hermitian)
    base = isreal(diag(B.data)) ? B.data : hermtotria(B)
    up(B) == :L ? unit_lower_triangular(base) : adjoint(unit_upper_triangular(base))
end
unit_lower_triangular(B::Diagonal) = Diagonal(oneunit.(diag(B)))
unit_lower_triangular(B::UnitLowerTriangular) = B
unit_lower_triangular(B::UnitUpperTriangular) = Diagonal(oneunit.(diag(B.data)))
unit_lower_triangular(B::LowerTriangular) = unit_lower_triangular(B.data)
unit_lower_triangular(B::UpperTriangular) = Diagonal(oneunit.(diag(B.data)))
unit_lower_triangular(B::AbstractMatrix) = UnitLowerTriangular(B)

diagonal(B::Conjugate) = conjugate(Diagonal(diag(B.parent)))
diagonal(B::Transpose) = Diagonal(diag(B.parent))
diagonal(B::Adjoint) = conjugate(Diagonal(diag(B.parent)))
diagonal(B::Symmetric) = Diagonal(diag(B.data))
diagonal(B::Hermitian) = Diagonal(real(diag(B.data)))
diagonal(B::Diagonal) = B
diagonal(B::UpperTriangular) = Diagonal(diag(B.data))
diagonal(B::LowerTriangular) = Diagonal(diag(B.data))
diagonal(B::UnitUpperTriangular) = Diagonal(oneunit.(diag(B.data)))
diagonal(B::UnitLowerTriangular) = Diagonal(oneunit.(diag(B.data)))
diagonal(B::AbstractMatrix) = Diagonal(diag(B))

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

