
using BenchmarkTools

function timefun(m, p, n, q, k=1, rep=4)
    #q = -expm1(log1p(-qs)/n) / p
    A = sprand(m, n, p)
    B = sprand(n, k, q)
    Corig = SparseWrappers.spmatmul_orig(A, B)
    beorig = @benchmark SparseWrappers.spmatmul_orig($A, $B) samples=rep
    bealt = @benchmark SparseWrappers.spmatmul_alt($A, $B) samples=rep
    torig = minimum(beorig)
    talt = minimum(bealt)
    (m=m, nnzA=nnz(A), n=n, nnzB=nnz(B), k=k, nnzC=nnz(Corig), torig=torig, talt=talt)
end

function sample(cnt::Integer, seed=0, oplimit=10^9, slimit=10^9)
    Random.seed!(seed)
    res = []
    i = 0
    while i < cnt
        m = 10^rand(3:6)
        n = 10^rand(3:6)
        k = 10^rand(0:1)
        p = exp10(rand(-6:-2))
        q = exp10(rand(-6:-1))
        nnzA = Int(ceil(m * n * p))
        nnzB = Int(ceil(n * k * q))
        nnzC = estimate_mulsize(m, nnzA, n, nnzB, k)
        space = (nnzA + nnzB + nnzC)*(sizeof(Int)+sizeof(Float64)) + (n + 2k)*(sizeof(Int))
        op1 = Int(ceil(m * n * k * p * q))
        op2 = Int(ceil(m * k))
        if n * p > 1 && n * q > 1 && op1 <= oplimit && op2 <= oplimit && space <= slimit
            i += 1
            println("$i: m=$m n=$n k=$k p=$p q=$q nnzA=$nnzA nnzB=$nnzB nnzC=$nnzC op1=$op1 op2=$op2 space=$space")
            push!(res, timefun(m, p, n, q, k))
        end
    end
    res
end

op1(r) = r.nnzA * r.nnzB / r.n
op2(r) = r.m * r.k
space(r) = r.nnzC * (sizeof(Int) + sizeof(Float64)) + r.k * sizeof(Int)

nlogn(n::Integer) = ilog2(n) * n

# convert vector of 7-tuples to array
function model_matrix(res::Vector)
    A = vcat((collect(hcat(1.0, op1(r), op2(r), r.nnzC/r.k, r.k*nlogn(r.nnzC÷r.k)) for r = res))...)
    B = vcat((collect(hcat(r.torig, r.talt) for r = res))...)
    W = [r.torig for r = res]
    A, getproperty.(B, :time), getproperty.(W, :time)
end

q(qs, n, p ) = -expm1(log1p(-qs)/n) / p
rhs(qs, k, n, p) = sprand(n, k, q(qs, n, p))
mat(n, p ) = sprandn(n, n, p)

function timefun2(matmul::Function, A::SparseMatrixCSC, qs::Float64, k::Integer)
    n = size(A, 2)
    p = nnz(A) / length(A)
    B = rhs(qs, k, n, p)
    be = @benchmark $matmul($A, $B) evals=10
    minimum(be).time
end

function sample2(matmul::Function, A, qs, kvec::AbstractVector{<:Integer})
    timefun2.(matmul, Ref(A), qs, kvec)
end

function sample1(matmul::Function, n::Integer, p::Float64, kvec::AbstractVector, qsvec::AbstractVector)
    Random.seed!(0)
    A = mat(n, p)
    res = Float64[]
    for qs in qsvec
        append!(res, sample2(matmul, A, qs, kvec))
    end
    reshape(res, length(kvec), length(qsvec))
end


#test environment to find speed of sorting vs. scanning

function sort_or_scan(rowval::Vector, nzval::Vector, xb::Vector, len::Integer, m::Integer, nnz::Integer)

    len >= m >= nnz || throw(ArgumentError("len($len) >= m($m) >= nnz($nnz) requested"))
    println("sort_or_scan($len, $m, $nnz)")
    rowval, nzval, xb, ip0, ip, m = nzsetup(rowval, nzval, xb, len, m, nnz)

    bescan = @benchmark runscan(rowval, nzval, xb, $ip0, $ip, $m) setup=begin
                                rowval, nzval, xb = copy($rowval), copy($nzval), copy($xb)
                            end samples=10 evals=1

    besort = @benchmark runsort(rowval, nzval, xb, $ip0, $ip, $m) setup=begin
                                rowval, nzval, xb = copy($rowval), copy($nzval), copy($xb)
                            end samples=10 evals=1

    getproperty.(minimum.([bescan, besort]), :time)
end

function runsort(rowvalC, nzvalC, xb, ip0, ip, m)
    let ip0=ip0, ip=ip, k0 = ip0 - 1
        sort!(rowvalC, ip0, ip-1, QuickSort, Base.Order.Forward)
        for vp = ip0:ip-1
            k = rowvalC[vp]
            xb[k] = false
            nzvalC[vp] = nzvalC[k+k0]
        end
    end
end

function runscan(rowvalC, nzvalC, xb, ip0, ip, m)
    let ip0=ip0, m=m, k0 = ip0 - 1
        for k = 1:m
            if xb[k]
                xb[k] = false
                rowvalC[ip0] = k
                nzvalC[ip0] = nzvalC[k+k0]
                ip0 += 1
            end
        end
    end
end

function nzsetup(rowval, nzval, xb, len, m, nnz, Tv=Float64, Ti=Int)
    @assert 0 <= nnz <= m <= len
    Random.seed!(65+len)
    length(nzval) < len && resize!(nzval, len)
    length(rowval) < len && resize!(rowval, len)
    length(xb) < m && resize!(xb, m)
    fill!(xb, false)
    ip0 = rand(1:len-m) 
    k0 = ip0 - 1
    for j = 1:nnz
        k = rand(1:m)
        xb[k] = true
        nzval[k+k0] = randn(Tv)
        rowval[j+k0] = k
    end
    ip = k0 + nnz
    rowval, nzval, xb, ip0, ip, m 
end          

samplex(rowval, nzval, xb, m::Integer, nz::Integer) = sort_or_scan(rowval, nzval, xb, max(10^7,2^m*9÷8), 2^m, 2^nz)
samplex(rowval, nzval, xb, m::Integer) = samplex.(Ref(rowval), Ref(nzval), Ref(xb), m, max(0,m-8):m)
samplex(r::UnitRange) = samplex.(Ref(Int[]), Ref(Float64[]), Ref(Bool[]), r)

"""
    samplex_plot(r::UnitRange, res::Vector{Vector{Vector{Float64}}})

usage:
    res = samplex(0:10)
    xscanm, scannnz, sortm, sortnnz, xm, ym = samplex_sort(0:10, res)

    scatter(scanm, scannnz)

    scatter!(sortm, sortnnz)

    plot!(xm, ym)

    See also sort_or_scan.png in data directory.
"""
function samplex_plot(r::UnitRange, res::Vector{<:Vector})
    sortm = Int[]
    sortnnz = Int[]
    scanm = Int[]
    scannnz = Int[]
    j = 0
    for m = r
        cm = res[j+=1]
        k = length(cm)
        for nz = m:-1:m-k+1
            data = cm[k]
            k -= 1
            if data[1] < data[2]
                push!(scanm, m)
                push!(scannnz, nz)
            else
                push!(sortm, m)
                push!(sortnnz, nz)
            end
        end
    end

    ym = collect(1:length(res)-7)
    xm = log2.(3 * nlogn.((^).(2, ym)))

    scanm, scannnz, sortm, sortnnz, xm, ym
end

nlogn(x) = x <= 0 ? zero(x) : ilog2(x) * x


