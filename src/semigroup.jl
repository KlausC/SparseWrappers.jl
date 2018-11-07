
# verify transformation semigroup generated by wrappers
#
REAL_ID = [4 5 6; 3 4 5; 2 3 4]
COMPLEX_ID = REAL_ID * (1 + im)

#identity
#transpose
#adjoint
symmup(A) = Symmetric(A, :U)
symmlo(A) = Symmetric(A, :L)
hermup(A) = Hermitian(A, :U)
hermlo(A) = Hermitian(A, :L)
triaup(A) = UpperTriangular(A)
trialo(A) = LowerTriangular(A)
triaupu(A) = UnitUpperTriangular(A)
trialou(A) = UnitLowerTriangular(A)
diagonal(A) = Diagonal(A)
bidiup(A) = Bidiagonal(A, :U)
bidilo(A) = Bidiagonal(A, :L)
tridi(A) = Tridiagonal(A)
symmtridiup(A) = SymTridiagonal(Symmetric(A, :U))
symmtridilo(A) = SymTridiagonal(Symmetric(A, :L))
hermtridiup(A) = Hermitian(Tridiagonal(A), :U)
hermtridilo(A) = Hermitian(Tridiagonal(A), :L)

REAL_GEN = [identity, transpose, symmup, symmlo, triaup, trialo, triaupu, trialou,
            diagonal, bidiup, bidilo, tridi, symmtridiup, symmtridilo]

COMPLEX_GEN = [REAL_GEN..., adjoint, hermup, hermlo, hermtridiup, hermtridilo]

function closegen(ID::T, GEN::Vector) where T<:AbstractMatrix
    dic = Dict{T,Any}(ID => Set{Any}())
    modified = true
    len(set::Set) = isempty(set) ? 0 : length(first(set))
    setkg(k, gen) = isempty(dic[k]) ? (gen,) : (gen, first(dic[k])...)
    while modified
        modified = false
        keysprev = collect(keys(dic))
        for gen in GEN
            for k in keysprev
                gk = Matrix(gen(k))
                if !haskey(dic, gk)
                    dic[gk] = Set{Any}(Ref(setkg(k, gen)))
                    modified = true
                elseif len(dic[gk]) == len(dic[k]) + 1
                    push!(dic[gk], setkg(k, gen))
                end
            end
        end
    end
    dic
end

closereal() = closegen(REAL_ID, REAL_GEN)
closecomplex() = closegen(COMPLEX_ID, COMPLEX_GEN)
