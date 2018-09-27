
struct Conjugate <:AbstractMatrix
    parent
end

struct HermiteTridiagonal <:AbstractMatrix
    dv
    ev
end

"""
    combine wrappers of the following kinds to achieve

  1. try to reduce number of involved wrappers
  2. standardized sequence of combination
"""
WRAPPER_TYPES = [
                 Conjugate, Transpose, Adjoint, Symmetric, Hermitian,
                 UpperTriangular, LowerTriangular, UnitUpperTriangular, UnitLowerTriangular,
                 Diagonal, Bidiagonal, Tridiagonal, SymTridiagonali, HermiteTridiagonal]

up(B::Union{Symmetric,Hermitian}) = B.uplo == 'U' ? :U : :L
toggle(s::Symbol) = s == :U ? :L : :U

# missing types: HermiteTridiagonal, Conjugate to be defined
Conjugate(B::Conjugate) = B
Conjugate(B::Transpose) = Adjoint(B)
Conjugate(B::Adjoint) = Transpose(B)
# Conjugate(B::Symmetric) =
# Conjugate(B::Hermitian) =

Transpose(B::Conjugate) = Adjoint(B)
Transpose(B::Transpose) = B.parent
Transpose(B::Adjoint) = Conjugate(B)
Transpose(B::Symmetric) = B
Transpose(B::Hermitian) = Conjugate(B)

Adjoint(B::Conjugate) = Transpose(B)
Adjoint(B::Transpose) = Conjugate(B)
Adjoint(B::Adjoint) = B.parent
Adjoint(B::Symmetric) = Conjugate(B)
Adjoint(B::Hermitian) = B

Symmetric(B::Conjugate, uplo=:U) = Conjugate(Symmetric(B.parent, uplo))
Symmetric(B::Transpose, uplo=:U) = Symmetric(B.parent, toggle(uplo))
Symmetric(B::Adjoint, uplo=:U) = Conjugate(Symmetric(B.parent, toggle(uplo)))
Symmetric(B::Symmetric, uplo=:U) = B
Symmetric(B::Hermitian, uplo=:U) = uplo == up(B) ? Symmetric(B.parent, uplo) : Conjugate(Symmetric(B.parent, up(B)))

Hermitian(B::Conjugate, uplo=:U) = Conjugate(Hermitian(B.parent, uplo))
Hermitian(B::Transpose, uplo=:U) = Conjugate(Hermitian(B.parent, toggle(uplo)))
Hermitian(B::Adjoint, uplo=:U) = Hermitian(B.parent, toogle(uplo))
Hermitian(B::Symmetric, uplo=:U) = uplo == up(B) ? Hermitian(B.parent, uplo) : Conjugate(Hermitian(B.parent, up(B)))
Hermitian(B::Hermitian, uplo=:U) = B


