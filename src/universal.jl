"""
    UniversalWrapper(::AbstractMatrix)

UniversalWrapper generalizes the well-known wrappers `oiangular, Symmetric, Adjoint,
Diagonal, SubArray, ...` in a more flexible way.
Each wrapper is considered as an operation performed on a subset of the set of square
matrices, applied to such a matrix (called parent).
The set of operators with the standard composition form a semi-group.
It is the aim of this structure to represent all combinations in one object.
The types are distinguished by a big amount of type parameters to enable easy dispatch
on all properties.

Type parameters are a triple of integers {Up,Di,Lo}  (for upper, diagonal, lower)

Up - relies to triangle above and excluding diagonal
Di - relies to diagonal elements of parent matrix
Lo - relies to triangle below and excluding diagonal

Values
Up/Lo:
-  0 constant zero (ignore A[i,j])
-  1 reproduce A[i,j]
- -1 replace A[i,j] by conj.(A[i,j])
-  2 replace A[i,j] by transpose(A[j,i])
- -2 replace A[i,j] by adjoint(A[j,i])
Di:
-  0 constant one
-  1 reproduce A[i,i]
- -1 replace A[i,i] by conj.(A[i,i])
-  2 replace A[i,i] by (A[i,i] + transpose(A[i,i]))/2 (enforce symmetric)
- -2 replace A[i,i] by conj.((A[i,i] + transpose(A[i,i]))/2) (enforce symmetric)
-  3 replace A[i,i] by (A[i,i] + adjoint(A[i,i]))/2 (enforce hermitian)
"""
struct UniversalWrapper{Tv,P<:AbstractMatrix{Tv},Up,Di,Lo} <: AbstractMatrix{Tv}
   parent::P
   UniversalWrapper{Tv,P,Up,Di,Lo}(p::P) where {Tv,P<:AbstractMatrix{Tv},Up,Di,Lo} = new{Tv,P,Up,Di,Lo}(p)
end

function UniversalWrapper{Tv1,P1,Up1,Di1,Lo1}(p::UniversalWrapper{Tv,P,Up2,Di2,Lo2}) where {Tv1,P1,Up1,Di1,Lo1,Tv,P,Up2,Di2,Lo2}
    Up, Di, Lo = compose(Up1, Di1, Lo1, Up2, Di2, Lo2)
    UniversalWrapper{Tv,P,Up,Di,Lo}(parent(p))
end

@inline function universal(Up1, Di1, Lo1, p::UniversalWrapper{Tv,P,Up2,Di2,Lo2}) where {Tv,P,Up2,Di2,Lo2}
    Up, Di, Lo = compose(Up1, Di1, Lo1, Up2, Di2, Lo2)
    UniversalWrapper{Tv,P,Up,Di,Lo}(parent(p))
end

transpose(p::UniversalWrapper) = universal(2, 2, 2, p)
adjoint(p::UniversalWrapper) = universal(-2, -1, -2, p)
unituppertria(p::UniversalWrapper) = universal(1, 0, 0, p)
unitlowertria(p::UniversalWrapper) = universal(0, 0, 1, p)
uppertria(p::UniversalWrapper) = universal(1, 1, 0, p)
lowertria(p::UniversalWrapper) = universal(0, 1, 1, p)
uppersymm(p::UniversalWrapper) = universal(1, 2, 2, p)
lowersymm(p::UniversalWrapper) = universal(2, 2, 1, p)
upperherm(p::UniversalWrapper) = universal(1, 3, -2, p)
lowerherm(p::UniversalWrapper) = universal(-2, 3, 1, p)

for gen in (:transpose, :adjoint, :unituppertria, :unitlowertria, :uppertria, :lowertria,
            :uppersymm, :lowersymm, :upperherm, :lowerherm)

    @eval ($gen)(p::AbstractMatrix) = ($gen)(UniversalWrapper(p))
end

UniversalWrapper(p::UniversalWrapper) = p
UniversalWrapper(p::P) where {Tv,P<:AbstractMatrix{Tv}} = UniversalWrapper{Tv,P,1,1,1}(p)
UniversalWrapper(p::Transpose) = transpose(_upp(p)) 
UniversalWrapper(p::Adjoint) = adjoint(_upp(p)) 
UniversalWrapper(p::UnitUpperTriangular) = unituppertria(_upp(p)) 
UniversalWrapper(p::UnitLowerTriangular) = unitlowertria(_upp(p)) 
UniversalWrapper(p::UpperTriangular) = uppertria(_upp(p)) 
UniversalWrapper(p::LowerTriangular) = lowertria(_upp(p)) 
UniversalWrapper(p::Symmetric) = p.uplo == 'U' ?  uppersymm(_upp(p)) : lowersymm(_upp(p))
UniversalWrapper(p::Hermitian) = p.uplo == 'U' ?  upperherm(_upp(p)) : lowerherm(_upp(p))

_upp(p) = UniversalWrapper(parent(p))

Base.parent(A::UniversalWrapper) = A.parent
Base.size(A::UniversalWrapper) = size(parent(A))
function Base.getindex(U::UniversalWrapper{Tv,P,Up,Di,Lo}, i::Integer,j::Integer) where {Tv,P,Up,Di,Lo}
    A = parent(U)
    if i < j
        Up ==  1 ? A[i,j] :
        Up == -1 ? conj.(A[i,j]) :
        Up ==  2 ? transpose(A[j,i]) :
        Up == -2 ? adjoint(A[j,i]) : zero(A[i,j])
    elseif i > j
        Lo ==  1 ? A[i,j] :
        Lo == -1 ? conj.(A[i,j]) :
        Lo ==  2 ? transpose(A[j,i]) :
        Lo == -2 ? adjoint(A[j,i]) : zero(A[i,j])
    else # i == j
        Di ==  1 ? A[i,j] :
        Di == -1 ? conj.(A[i,j]) :
        Di ==  2 ? tosymm(A[i,j]) :
        Di == -2 ? conj.(tosymm(A[i,j])) :
        Di ==  3 ? toherm(A[i,j]) : one(A[i,j])
    end
end

tosymm(a::Real) = a
toherm(a::Real) = a
tosymm(a::Complex) = a
toherm(a::Complex) = real(a)
tosymm(a::AbstractArray) = (a + transpose(a)) / 2
toherm(a::AbstractArray) = (a + adjoint(a)) / 2

@inline function compose_ul(Up1, Up2, Lo2)
    Up1 ==  1 ? Up2 :
    Up1 == -1 ? -Up2 :
    Up1 ==  2 ? cominv(Lo2) :
    Up1 == -2 ? -cominv(Lo2) : 0
end
@inline function compose_di(Di1, Di2)
    Di1 ==  1 ? Di2 :
    Di1 == -1 ? (Di2 == 3 ? Di2 : -Di2) :
    Di1 ==  2 ? (Di2 == 0 ? 0 : Di2 < 0 ? -2 : Di2 < 3 ?  2 : 3) :
    Di1 == -2 ? (Di2 == 0 ? 0 : Di2 < 0 ?  2 : Di2 < 3 ? -2 : 3) :
    Di1 ==  3 ? (Di2 == 0 ? 0 : Di1) : 0
end
@inline function cominv(Up)
    Up ==  1 ?  2 :
    Up == -1 ? -2 :
    Up ==  2 ?  1 :
    Up == -2 ? -1 : 0
end

@inline function compose(Up1, Di1, Lo1, Up2, Di2, Lo2)
    compose_ul(Up1, Up2, Lo2), compose_di(Di1, Di2), compose_ul(Lo1, Lo2, Up2)
end

# check associativity of composition
function check_assoc()
    for Up1 in -2:2, Lo1 in -2:2
        for Up2 in -2:2, Lo2 in -2:2
            for Up3 in -2:2, Lo3 in -2:2
                if compose(compose(Up1, 0, Lo1, Up2, 0, Lo2)..., Up3, 0, Lo3) !=
                   compose(Up1, 0, Lo1, compose(Up2, 0, Lo2, Up3, 0, Lo3)...)
                    println("($Up1 0 $Lo1) ($Up2, 0, $Lo2) ($Up3, 0, $Lo3)")
                end
            end
        end
    end
    for Di1 in -2:3, Di2 in -2:3, Di3 in -2:3
        if compose(compose(0, Di1, 0, 0, Di2, 0)..., 0, Di3, 0) !=
           compose(0, Di1, 0, compose(0, Di2, 0, 0, Di3, 0)...)
            println("(0, $Di1, 0) (0, $Di2, 0) (0, $Di3, 0)")
        end
    end
end

