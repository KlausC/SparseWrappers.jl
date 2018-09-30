
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

up(B::Union{Symmetric,Hermitian}) = B.uplo == 'U' ? :U : :L
toggle(s::Symbol) = s == :U ? :L : :U

import LinearAlgebra: Transpose, Adjoint, Symmetric, Hermitian

# missing types: HermiteTridiagonal, Conjugate to be defined
Conjugate(B::Conjugate) = B.parent
Conjugate(B::Transpose) = Adjoint(B.parent)
Conjugate(B::Adjoint) = Transpose(B.parent)
# Conjugate(B::Symmetric) =
# Conjugate(B::Hermitian) =
# Conjugate(B::UpperTriangular) =
# Conjugate(B::LowerTriangular) =

Transpose(B::Conjugate) = Adjoint(B.parent)
Transpose(B::Transpose) = B.parent
Transpose(B::Adjoint) = Conjugate(B.parent)
Transpose(B::Symmetric) = B
Transpose(B::Hermitian) = Conjugate(B)
# Transpose(B::UpperTriangular) =
# Transpose(B::LowerTriangular) =

Adjoint(B::Conjugate) = Transpose(B.parent)
Adjoint(B::Transpose) = Conjugate(B.parent)
Adjoint(B::Adjoint) = B.parent
Adjoint(B::Symmetric) = Conjugate(B)
Adjoint(B::Hermitian) = B
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
Symmetric(B::UpperTriangular, uplo::Symbol=:U) = uplo == :U ? Symmetric(B.data, :U) : Diagonal(diag(B.data))
Symmetric(B::LowerTriangular, uplo::Symbol=:L) = uplo == :L ? Symmetric(B.data, :L) : Diagonal(diag(B.data))
Symmetric(B::UnitUpperTriangular, uplo::Symbol=:U) = uplo == :U ? Symmetric(sparsecsc(B), :U) : Diagonal(one.(diag(B.data)))
Symmetric(B::UnitLowerTriangular, uplo::Symbol=:L) = uplo == :L ? Symmetric(sparsecsc(B), :L) : Diagonal(one.(real(diag(B.data))))

Hermitian(B::Conjugate, uplo::Symbol=:U) = Conjugate(Hermitian(B.parent, uplo))
Hermitian(B::Transpose, uplo::Symbol=:U) = Conjugate(Hermitian(B.parent, toggle(uplo)))
Hermitian(B::Adjoint, uplo::Symbol=:U) = Hermitian(B.parent, toggle(uplo))
Hermitian(B::Symmetric, uplo::Symbol=:U) = uplo == up(B) ? Hermitian(B.data, uplo) : Conjugate(Hermitian(B.data, up(B)))
Hermitian(B::Hermitian, uplo::Symbol=:U) = B
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
## LowerTriangular(B::LowerTriangular) = B
LowerTriangular(B::UpperTriangular) = Diagonal(diag(B.data))
LowerTriangular(B::UnitLowerTriangular) = B
LowerTriangular(B::UnitUpperTriangular) = Diagonal(one.(diag(B.data)))

Diagonal(B::Conjugate) = Conjugate(Diagonal(diag(B.parent)))
Diagonal(B::Transpose) = Diagonal(diag(B.parent))
Diagonal(B::Adjoint) = Diagonal(conj.(diag(B.parent)))
Diagonal(B::Symmetric) = Diagonal(diag(B.data))
Diagonal(B::Hermitian) = Diagonal(real(diag(B.data)))
## Diagonal(B::Diagonal) = B
Diagonal(B::UpperTriangular) = Diagonal(diag(B.data))
Diagonal(B::LowerTriangular) = Diagonal(diag(B.data))
Diagonal(B::UnitUpperTriangular) = Diagonal(one.(diag(B.data)))
Diagonal(B::UnitLowerTriangular) = Diagonal(one.(diag(B.data)))

function hermtotria(B::Hermitian)
    println("hermtotria start")
    A = sparsecsc((up(B) == :U ? UpperTriangular : LowerTriangular)(B.data))
    for i = 1:size(A, 1)
        A[i,i] = real(A[i,i])
    end
    println("hermtotria end")
    A
end
