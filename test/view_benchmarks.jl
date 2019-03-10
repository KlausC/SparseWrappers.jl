# benchmarking several functions on views
using BenchmarkTools
using SparseArrays

import SparseArrays:   blockdiag, nnz, findnz, ftranspose
import Base:    isone

if !hasmethod(blockdiag, (AbstractMatrix,))
    blockdiag(A::AbstractMatrix...) = blockdiag(sparse.(A)...)
end
if !hasmethod(nnz, (AbstractMatrix,))
    nnz(A::AbstractMatrix) = nnz(sparse(A))
end
if !hasmethod(findnz, (AbstractMatrix,))
    findnz(A::AbstractMatrix) = findnz(sparse(A))
end
if !hasmethod(ftranspose, (AbstractMatrix,Any))
    ftranspose(A::AbstractMatrix, f) = ftranspose(sparse(A), f)
end
isone(A::AbstractMatrix) = isone(sparse(A))

function view_benchmarks(n::Integer=1000)

    x_findall(x) = findall(Base.Fix2(isless, 0.0), x)
    x_ftranspos(x) = ftranspose(x, transpose)
    fname(f) = string(nameof(f))

    suite = BenchmarkGroup()

    rng = MersenneTwister(20190131)
    res = []
    n1, n2 = (n ÷ 10) + 1, n - n ÷ 10
    B = []
    A = sprand(rng, n, n, 0.01)
    push!(B, ("CSC", A))
    push!(B, ("CSCView-slice", view(A, :, n1:n2)))
    push!(B, ("CSCInterface-1", view(A, 1:n, n1:n2)))
    push!(B, ("CSCInterface-2", view(A, n1:n2, n2:-1:n1)))
    push!(B, ("CSCView-rev", view(A, n2:-1:n1, n1:n2)))
    N = axes(B, 1)

    for f in (nnz, findnz, x_ftranspos, Matrix )
        suite[fname(f)] = BenchmarkGroup(["views", string(n)])
        for i in N
            name, X = B[i]
            suite[fname(f)][name] = @benchmarkable $f($X)
        end
    end

    for f in (x_findall, iszero, isone)
        suite[fname(f)] = BenchmarkGroup(["views", string(n)])
        for i in N
            name, X = B[i]
            suite[fname(f)][name] = @benchmarkable $f($X)
        end
    end

    for f in (mapreduce,)
        suite[fname(f)] = BenchmarkGroup(["views", string(n)])
        for i in N
            name, X = B[i]
            suite[fname(f)][name*"dims1"] = @benchmarkable $f(floor, +, $X, dims=1)
            suite[fname(f)][name*"dims2"] = @benchmarkable $f(floor, +, $X, dims=2)
        end
    end

    for f in (hcat, vcat, blockdiag)
        suite[fname(f)] = BenchmarkGroup(["views", string(n)])
        for i in N
            name, X = B[i]
            suite[fname(f)][name] = @benchmarkable $f($X, $X)
        end
    end

    AS = A + A'
    AS = [AS AS]
    B = []
    push!(B, ("Symm", AS))
    push!(B, ("SymmView-slice", view(AS, :, 1:n)))
    push!(B, ("SymmInterface-2", view(AS, n1:n2, n1:n2)))
    push!(B, ("SymmView-rev", view(AS, n2:-1:n1, n2:-1:n1)))
    N = axes(B, 1)
    for f in (issymmetric, ishermitian)
        suite[fname(f)] = BenchmarkGroup(["views", string(n)])
        for i in N
            name, X = B[i]
            suite[fname(f)][name] = @benchmarkable $f($X)
        end
    end

    ASU = triu(A)
    ASL = tril(A)
    B = []
    push!(B, ("Triu", ASU))
    push!(B, ("Triu-slice", view(ASU, :, 1:n)))
    push!(B, ("TriuInterface", view(ASU, n1:n2, n1:n2)))
    push!(B, ("Triu-rev", view(ASL, n2:-1:n1, n2:-1:n1)))
    N = axes(B, 1)
    for f in (istriu,)
        suite[fname(f)] = BenchmarkGroup(["views", string(n)])
        for i in N
            name, X = B[i]
            suite[fname(f)][name] = @benchmarkable $f($X)
        end
    end

    B = []
    push!(B, ("Tril", ASL))
    push!(B, ("Tril-slice", view(ASL, :, 1:n)))
    push!(B, ("TrilInterface", view(ASL, n1:n2, n1:n2)))
    push!(B, ("Tril-rev", view(ASU, n2:-1:n1, n2:-1:n1)))
    N = axes(B, 1)
    for f in (istril,)
        suite[fname(f)] = BenchmarkGroup(["views", string(n)])
        for i in N
            name, X = B[i]
            suite[fname(f)][name] = @benchmarkable $f($X)
        end
    end

    suite
end

