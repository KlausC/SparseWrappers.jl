
export AbstractWrappedArray, basetype
export WTranspose, WAdjoint, WSymmetric, WUpperTriangular

abstract type AbstractWrappedArray{T,N,P,S} <: AbstractArray{T,N}; end

basetype(::Type{P}) where P<:AbstractWrappedArray{T,N,X,S} where {T,N,X,S} = S
basetype(::Type{P}) where P = P
basetype(x::T) where T = basetype(T)

SparseArrays.issparse(::Type{P}) where P<:AbstractWrappedArray = issparse(basetype(P))
SparseArrays.issparse(::Type{<:Union{SparseVector,SparseMatrixCSC}}) = true
SparseArrays.issparse(::Type) = false
SparseArrays.issparse(x::T) where T<:AbstractArray = issparse(T)

# the following concrete wrapper types are demo replacements for the standard ones. 
# an additional type parameter is appended to the original
# actually the original definitions in LinearAlgebra and Base have to be modified 

# Transpose
struct WTranspose{T,P,S} <:AbstractWrappedArray{T,2,P,S}
    parent::P
    WTranspose(A::P) where P<:AbstractArray{T,N} where {T,N} = new{T,P,basetype(P)}(A)
end
Base.size(x::WTranspose) = reverse(size(x.parent))
Base.getindex(x::WTranspose,I,J,K...) = getindex(x.parent, J, I, K...)

# Adjoint
struct WAdjoint{T,P,S} <:AbstractWrappedArray{T,2,P,S}
    parent::P
    WAdjoint(A::P) where P<:AbstractArray{T,N} where {T,N} = new{T,P,basetype(P)}(A)
end
Base.size(x::WAdjoint) = reverse(size(x.parent))
Base.getindex(x::WAdjoint,I,J,K...) = adjoint(getindex(x.parent, J, I, K...))

# Symmetric
struct WSymmetric{T,P,S} <:AbstractWrappedArray{T,2,P,S}
    parent::P
    uplo::Char
    WSymmetric(A::P,ul::Symbol) where P<:AbstractArray{T,N} where {T,N} = new{T,P,basetype(P)}(A, ul == :U ? 'U' : 'L')
end
Base.size(x::WSymmetric) = size(x.parent)
Base.getindex(x::WSymmetric,I...) = getindex(Symmetric(x.parent, x.uplo == 'U' ? :U : :L), I...)

# UpperTriangular
struct WUpperTriangular{T,P,S} <:AbstractWrappedArray{T,2,P,S}
    parent::P
    UpperTriangular(A::P) where P<:AbstractArray{T,N} where {T,N} = new{T,P,basetype(P)}(A)
end
Base.size(x::WUpperTriangular) = size(x.parent)
Base.getindex(x::WUpperTriangular,I...) = getindex(UpperTriangular(x.parent), I...)

# SubArray





