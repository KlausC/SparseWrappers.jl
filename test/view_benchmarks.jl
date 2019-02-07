# benchmarking several functions on views
using BenchmarkTools
function run_view_benchmarks(n::Integer=1000)

    suite = BenchmarkGroup()

    rng = MersenneTwister(20190131)
    res = []
    n1, n2 = (n ÷ 10) + 1, n - n ÷ 10
    B = []
    A = sprand(rng, n, n, 0.01)
    push!(B, A)
    push!(B, view(A, :, n1:n2))         # SparseMatrixCSCView
    push!(B, view(A, 1:n, n1:n2))       # SparseMatrixCSCInterface \ SparseMatrixCSCView
    push!(B, view(A, n1:n2, n2:-1:n1))  # SparseMatrixCSCInterface \ SparseMatrixCSCView
    push!(B, view(A, n2:-1:n1, n1:n2))  # SparseMatrixCSCViewAll \ SparseMatricCSCInterface
    N = axes(B, 1)

    x_findall(x) = findall(Base.Fix2(isless, 0.0), x)
    x_ftranspos(x) = SparseArrays.ftranspose(x, transpose)
    for f in (nnz, findnz, x_ftranspos, Matrix )
        suite[nameof(f)] = BenchmarkGroup(["views", string(n)])
        for i in N
            X = B[i]
            suite[nameof(f)][i] = @benchmarkable $f($X)
        end
    end

    for f in (x_findall, iszero, isone)
        suite[nameof(f)] = BenchmarkGroup(["views", string(n)])
        for i in N
            X = B[i]
            suite[nameof(f)][i] = @benchmarkable $f($X)
        end
    end

    for f in (mapreduce,)
        suite[nameof(f)] = BenchmarkGroup(["views", string(n)])
        for i in N
            X = B[i]
            suite[nameof(f)][i, 1] = @benchmarkable $f(floor, +, $X, dims=1)
            suite[nameof(f)][i, 2] = @benchmarkable $f(floor, +, $X, dims=2)
        end
    end

    for f in (hcat, vcat, blockdiag)
        suite[nameof(f)] = BenchmarkGroup(["views", string(n)])
        for i in N
            X = B[i]
            suite[nameof(f)][i] = @benchmarkable $f($X, $X)
        end
    end

    AS = A + A'
    AS = [AS AS]
    B = []
    push!(B, AS)
    push!(B, view(AS, :, 1:n))
    push!(B, view(AS, n1:n2, n1:n2))
    push!(B, view(AS, n2:-1:n1, n2:-1:n1))
    N = axes(B, 1)
    for f in (issymmetric, ishermitian)
        suite[nameof(f)] = BenchmarkGroup(["views", string(n)])
        for i in N
            X = B[i]
            suite[nameof(f)][i] = @benchmarkable $f($X)
        end
    end

    AS = triu(A)
    B = []
    push!(B, AS)
    push!(B, view(AS, :, 1:n))
    push!(B, view(AS, n1:n2, n1:n2))
    push!(B, view(AS, n2:-1:n1, n2:-1:n1))
    N = axes(B, 1)
    for f in (istriu,)
        suite[nameof(f)] = BenchmarkGroup(["views", string(n)])
        for i in N
            X = B[i]
            suite[nameof(f)][i] = @benchmarkable $f($X)
        end
    end

    AS = tril(A)
    B = []
    push!(B, AS)
    push!(B, view(AS, :, 1:n))
    push!(B, view(AS, n1:n2, n1:n2))
    push!(B, view(AS, n2:-1:n1, n2:-1:n1))
    N = axes(B, 1)
    for f in (istril,)
        suite[nameof(f)] = BenchmarkGroup(["views", string(n)])
        for i in N
            X = B[i]
            suite[nameof(f)][i] = @benchmarkable $f($X)
        end
    end

    suite
end

