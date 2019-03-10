"""
    UniversalWrapper(::AbstractMatrix)

UniversalWrapper generalizes the well-known wrappers `*Triangular, Symmetric, Adjoint,
Diagonal, ...` in a more flexible way.
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
if A[i,i] isa AbstractMatrix convert to Wrapper with same Up,Di,Lo.
otherwise:
-  0 constant one
-  1 A[i,i]
- -1 conj(A[i,i])
-  2 real(A[i,i])
"""
struct UniversalWrapper{Tv,P<:AbstractMatrix{Tv},Up,Di,Lo} <: AbstractMatrix{Tv}
   parent::P
   function UniversalWrapper{Tv,P,Up,Di,Lo}(p::P) where {Tv,P<:AbstractMatrix{Tv},Up,Di,Lo}
       check_updilo(Up, Di, Lo) || throw(ArgumentError("Up/Lo = $Up/$Lo invalid"))
       new{Tv,P,Up,Di,Lo}(p)
   end
end

function UniversalWrapper{Tv1,P1,Up1,Di1,Lo1}(p::UniversalWrapper{Tv,P,Up2,Di2,Lo2}) where {Tv1,P1,Up1,Di1,Lo1,Tv,P,Up2,Di2,Lo2}
    Up, Di, Lo = compose(Tv, Up1, Di1, Lo1, Up2, Di2, Lo2)
    UniversalWrapper{Tv,P,Up,Di,Lo}(parent(p))
end

@inline function universal(Up1, Di1, Lo1, ::Type{UniversalWrapper{Tv,P,Up2,Di2,Lo2}}) where {Tv,P,Up2,Di2,Lo2}
    Up, Di, Lo = compose(Tv, Up1, Di1, Lo1, Up2, Di2, Lo2)
    UniversalWrapper{Tv,P,Up,Di,Lo}
end
@inline universal(Up1, Di1, Lo1, p::UniversalWrapper{Tv,P,Up2,Di2,Lo2}) where {Tv,P,Up2,Di2,Lo2} = universal(Up1, Di1, Lo1, typeof(p))(parent(p))

@inline universal(Up, Di, Lo, a::AbstractMatrix) = universal(Up, Di, Lo, UniversalWrapper(a))

@inline universal(Up, Di, Lo, T::Type) = universal(Up, Di, Lo, UniversalWrapper{T,Matrix{T},1,1,1})

@inline universal(t::Tuple{Int,Int,Int}, x) = universal(t..., x)

transpose(p::UniversalWrapper) = universal(2, 1, 2, p)
adjoint(p::UniversalWrapper) = universal(-2, -1, -2, p)
unituppertria(p::UniversalWrapper) = universal(1, 0, 0, p)
unitlowertria(p::UniversalWrapper) = universal(0, 0, 1, p)
uppertria(p::UniversalWrapper) = universal(1, 1, 0, p)
lowertria(p::UniversalWrapper) = universal(0, 1, 1, p)
uppersymm(p::UniversalWrapper) = universal(1, 1, 2, p)
lowersymm(p::UniversalWrapper) = universal(2, 1, 1, p)
upperherm(p::UniversalWrapper) = universal(1, 2, -2, p)
lowerherm(p::UniversalWrapper) = universal(-2, 2, 1, p)
conjugate(p::UniversalWrapper) = universal(-1, -1, -1, p)
diagonal(p::UniversalWrapper) = universal(0, 1, 0, p)
Base.one(p::UniversalWrapper) = universal(0, 0, 0, p)


for gen in (:transpose, :adjoint, :unituppertria, :unitlowertria, :uppertria, :lowertria,
            :uppersymm, :lowersymm, :upperherm, :lowerherm, :conjugate)

    @eval ($gen)(p::AbstractMatrix) = ($gen)(UniversalWrapper(p))
end

function_to_wrapper(f::Function, T::Type=Complex) = typeof(f(UniversalWrapper(zeros(T,2,2))))

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

function Base.getindex(U::UniversalWrapper{Tv,<:Any,Up,Di,Lo}, i::Integer,j::Integer) where {Tv,Up,Di,Lo}
    A = parent(U)
    if i < j
        Up ==  1 ? A[i,j] :
        Up == -1 ? conj.(A[i,j]) :
        Up ==  2 ? transpose(A[j,i]) :
        Up == -2 ? adjoint(A[j,i]) :
        zero(A[i,j])
    elseif i > j
        Lo ==  1 ? A[i,j] :
        Lo == -1 ? conj.(A[i,j]) :
        Lo ==  2 ? transpose(A[j,i]) :
        Lo == -2 ? adjoint(A[j,i]) :
        zero(A[i,j])
    else # i == j
        Tv <: AbstractArray && return convert_element(U, A[i,i])
        Di ==  1 ? A[i,i] :
        Di == -1 ? conj(A[i,i]) :
        Di ==  2 ? real(A[i,i]) :
        one(A[i,i])
    end
end

function check_updilo(Up::Integer, Di::Integer, Lo::Integer)
    true
    # Up == 0 || Lo == 0 || Up + Lo != 0
    # Up == 0 || Lo == 0 || Di != 2 || abs(Up+Lo) == 1 || throw(ArgumentError("Up/Di/Lo = $Up/$Di/$Lo invalid"))
    # Di != 0 || Up == 0 || Lo == 0 || throw(ArgumentError("Up/Di/Lo = $Up/$Di/$Lo invalid"))
end

function convert_element(u::UniversalWrapper{<:Any,<:Any,Up,Di,Lo}, a::X) where {Up,Di,Lo,Tv,X<:UniversalWrapper{Tv}}
    universal(Up, Di, Lo, a)
end
function convert_element(u::UniversalWrapper{<:Any,<:Any,Up,Di,Lo}, a::X) where {Up,Di,Lo,Tv,X<:AbstractMatrix{Tv}}
    unwrap(UniversalWrapper{Tv,X,Up,Di,Lo}(a))
end
iswrsparse(::Type{<:UniversalWrapper{<:Any,P,Up,Di,Lo}}) where {P,Up,Di,Lo} = iswrsparse(P) 

@inline function compose_ul(Up1, Up2, Lo2)
    Up1 ==  1 ? Up2 :
    Up1 == -1 ? -Up2 :
    Up1 ==  2 ? cominv(Lo2) :
    Up1 == -2 ? -cominv(Lo2) :
    0
end
@inline function compose_di(Di1, Di2)
    Di1 ==  1 ? Di2 :
    Di1 == -1 ? (Di2 == 2 ? Di2 : -Di2) :
    Di1 ==  2 ? Di2 == 0 ? 0 : Di1 :
    0
end
@inline cominv(Up) = sign(Up) * 3 - Up

function compose(::Type{<:Real}, arg...)
    Up, Di, Lo = compose(arg...)
    abs(Up), min(abs(Di), 1), abs(Lo)
end
compose(::Type, args...) = compose(args...)

@inline function compose(Up1, Di1, Lo1, Up2, Di2, Lo2)
    compose_ul(Up1, Up2, Lo2), compose_di(Di1, Di2), compose_ul(Lo1, Lo2, Up2)
end

import Base.∘
function ∘(::Type{<:UniversalWrapper{T,S,Up1,Di1,Lo1}},::Type{<:UniversalWrapper{T,S,Up2,Di2,Lo2}}) where {T,S,Up1,Di1,Lo1,T2,Up2,Di2,Lo2}

    UniversalWrapper{T,S,compose(T, Up1, Di1, Lo1, Up2, Di2, Lo2)...}
end

parameters(::Type{<:UniversalWrapper{<:Any,<:Any,Up,Di,Lo}}) where {Up,Di,Lo} = (Up, Di, Lo)

all_parameters(::Type) = [ (Up, Di, Lo) for Up = -2:2 for Di = -1:2 for Lo = -2:2 if check_updilo(Up, Di, Lo)]
all_parameters(::Type{<:Real}) = [ (Up, Di, Lo) for Up = 0:2 for Di = 0:1 for Lo = 0:2 if check_updilo(Up, Di, Lo)]

function generate(g::Union{Set,AbstractVector}, T::Type = Complex)
    g = Set([ x isa Function ? function_to_wrapper(x, T) : x for x in g])
    res = copy(g)
    n = 0
    while length(res) != n
        n = length(res)
        res2 = copy(res)
        for v in res2, u in g
            uv =  u ∘ v
            if !(uv in res)
                # println("$u ∘ $v = $uv")
                push!(res, uv)
            end
        end
    end
    collect(res)
end

function norm_conj(U::Type{<:UniversalWrapper{T,S,Up,Di,Lo}}) where {T,S,Up,Di,Lo}
   V = universal(-1,-1,-1, U)
   Up2, Di2, Lo2 = parameters(V)
   cmp(u, d, l) = (count((u,d,l) .< 0), count((u,d,l) .== -1), d, u, l) 
   cmp(Up, Di, Lo) < cmp(Up2, Di2, Lo2) ? U : V
end

check_well_defined() = isempty(not_well_defined())
function not_well_defined()
    res = []
    A = UniversalWrapper([1+1im 2+2im; 3+3im 4+4im])
    for updilo2 in all_parameters(Complex)
        U2A = universal(updilo2..., A)
        MU2A = universal(1, 1, 1, Matrix(U2A))
        for updilo1 in all_parameters(Complex)
            if universal(updilo1..., U2A) != universal(updilo1..., MU2A)
                push!(res, (updilo1, updilo2))
            end
        end
    end
    res
end

# check associativity of composition (move to test)
check_assoc() = isempty(non_associatives()) 
function non_associatives()
    res = []
    for Up1 in -2:2, Lo1 in -2:2
        for Up2 in -2:2, Lo2 in -2:2
            for Up3 in -2:2, Lo3 in -2:2
                if compose(compose(Up1, 0, Lo1, Up2, 0, Lo2)..., Up3, 0, Lo3) !=
                   compose(Up1, 0, Lo1, compose(Up2, 0, Lo2, Up3, 0, Lo3)...)
                    push!(res, ((Up1,0,Lo1), (Up2,0,Lo2), (Up3,0,Lo3)))
                end
            end
        end
    end
    for Di1 in -1:2, Di2 in -1:2, Di3 in -1:2
        if compose(compose(0, Di1, 0, 0, Di2, 0)..., 0, Di3, 0) !=
           compose(0, Di1, 0, compose(0, Di2, 0, 0, Di3, 0)...)
           push!(res, ((0,Di1,0), (0,Di2,0), (0,Di3,0)))
        end
    end
    res
end

check_conjugate_commutative() = relation_empty(!commutes, universal(-1,-1,-1,Complex))
function relation(f::Function, u::Type{<:UniversalWrapper})
    [v for v in universal.(all_parameters(Complex), Complex) if f(u, v) ]
end

relation_empty(f::Function, u::Type{<:UniversalWrapper}) = isempty(relation(f, u))

function commutator()
    [v for v in universal.(all_parameters(Complex), Complex) if relation_empty(!commutes, v)]
end

function inverses(u::Type{<:UniversalWrapper})
    e = universal(1, 1, 1, Complex)
    [v for v in universal.(all_parameters(Complex), Complex) if  v ∘ u == e == u ∘ v]
end

function invertibles()
    [v for v in universal.(all_parameters(Complex), Complex) if !isempty(inverses(v))]
end

function commutes(u::Type{<:UniversalWrapper}, v::Type{<:UniversalWrapper})
    u ∘ v == v ∘ u
end

