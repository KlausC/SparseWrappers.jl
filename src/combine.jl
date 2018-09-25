
struct Conjugate
    parent
end

struct HermiteTridiagonal
    dv
    ev
end

"""
    combine wrappers of the following kinds to achieve

  1. try to reduce number of involved wrappers
  2. standardized sequence of combination
"""
WRAPPER_TYPES = [
                 identity, Conjugate,
                 Transpose, Adjoint,
                 Symmetric, Hermitian,
                 UpperTriangular, LowerTriangular, UnitUpperTriangular, UnitLowerTriangular,
                 Diagonal, Bidiagonal, Tridiagonal, SymTridiagonali, HermiteTridiagonal]

# missing types: HermiteTridiagonal, Conjugate to be defined
Conjugate(B::Conjugate) = B
Conjugate(B::Transpose) = Adjoint(B)
Conjugate(B::Adjoint) = Transpose(B)
Conjugate(B::Symmetric) = Hermitian(B, :U)
Conjugate(B::Hermitian) = Symmetric(B, :U)

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

