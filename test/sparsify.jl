
n = 100
A = sprandn(ComplexF64, n, n, 0.1)
sp = sparsecsc
ma = Matrix
@test sp(A) === A

@testset "wrapped once $wr" for wr in
    (Conjugate, adjoint, transpose, symmetric, hermitian,
     upper_triangular, lower_triangular, unit_upper_triangular, unit_lower_triangular)

    @test sp(wr(A)) == ma(wr(ma(A)))
end
@testset "wrapped once $wr $uplo" for wr in (symmetric, hermitian), uplo in (:U, :L)
    @test sp(wr(A, uplo)) == ma(wr(ma(A), uplo))
end
@testset "view ($I,$J)" for I in (:, 5:2:10), J in (1:1:20, 20:10:100)  
    @test sp(view(A, I, J)) == ma(view(ma(A), I, J))
end

types1 = [Diagonal, Tridiagonal, Transpose, Adjoint,
               UpperTriangular, LowerTriangular, UnitUpperTriangular, UnitLowerTriangular,
               Symmetric, Hermitian]

types2 = [(view, (2:10, :)), (Bidiagonal, :U), (Bidiagonal, :L),
                (Symmetric, :U), (Symmetric, :L), (Hermitian, :U), (Hermitian, :L)]

@testset "depth $ty" for ty in types1
    @test depth(ty(A)) == 1
end
@testset "depth $ty $args" for (ty, args) in types2
    args = args isa Tuple ? args : (args,)
    @test depth(ty(A, args...)) == 1
end

@test depth(A) == 0
@test depth([1.7; 0]) == 0


