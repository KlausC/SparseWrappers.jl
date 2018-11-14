
begin
    rng = Random.MersenneTwister(1)
    n = 1000
    p = 0.02
    q = 1 - sqrt(1-p)
    Areal = sprandn(rng, n, n, p)
    Breal = randn(rng, n)
    Bcomplex = Breal + randn(rng, n) * im
    Acomplex = sprandn(rng, n, n, q) + sprandn(rng, n, n, q) * im
    @testset "symmetric/Hermitian sparse multiply with $S($U)" for S in (Symmetric, Hermitian), U in (:U, :L), (A, B) in ((Areal,Breal), (Acomplex,Bcomplex))
        Asym = S(A, U)
        As = sparse(Asym) # takes most time
        @test which(mul!, (typeof(B), typeof(Asym), typeof(B))).module in
            (SparseArrays, SparseWrappers)
        @test norm(Asym * B - As * B, Inf) <= eps() * n * p * 2
    end
end
