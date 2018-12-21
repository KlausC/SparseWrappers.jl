
"""
    MMatrix(A::AbstractMatrix)

MMatrix is a type concentrating on the mathematical properties of a Matrix.
That is in opposition to wrappers like Symmetric, Adjoint, etc., which have a bias 
on the internal representation.
Matrix(Symmetric(A)) has type `MMatrix{T, SparseMatrixCSC{T}, :symmetricup}`, which allows to
specialize methods by the storage type and the symmetry property.
"""
struct MMatrix{T,N,S<:AbstractArray{<:Any,N},P} <:AbstractArray{T,N}
    parent::S
end

Base.parent(ma::MMatrix) = typed_parent(ma)
Base.show(io::IO, ma::MMatrix) = show(io, typed_parent(ma))

MMatrix(a::MMatrix{T,N,X,P}) where {T,N,X,P} = MMatrix{T,N,X,P}(parent(a))
MMatrix(a::S) where {T,N,S<:StridedArray{T,N}} = MMatrix{T,N,S,:general}(a)
MMatrix(a::S) where {T,S<:SparseMatrixCSC{T}} = MMatrix{T,2,S,:general}(a)
MMatrix(a::S) where {T,S<:SparseVector{T}} = MMatrix{T,1,S,:general}(a)
function MMatrix(a::S) where {T,X,S<:Transpose{T,X}}
    T<:Complex && throw(ArgumentError("transpose of complex not supported")) 
    return MMatrix{T,2,X,:transpose}(parent(a)) 
end
function MMatrix(a::S) where {T,N,X<:AbstractArray{T,N},S<:Adjoint{T,X}}
    MMatrix{T,N,X,ifelse(T<:Complex,:adjoint,:transpose)}(parent(a)) 
end
function MMatrix(a::S) where {T,X,S<:Symmetric{T,X}}
    T<:Complex && throw(ArgumentError("symmetric of complex not supported")) 
    (a.uplo == 'U' ? MMatrix{T,2,X,:symmetricup} : MMatrix{T,2,X,:symmetriclow})(parent(a))
end
function MMatrix(a::S) where {T,X,S<:Hermitian{T,X}}
    if T<:Complex
        isreal(diag(a)) || throw(ArgumentError("Hermitian needs real diagonal")) 
        (a.uplo == 'U' ? MMatrix{T,2,X,:hermitianup} : MMatrix{T,2,X,:hermitianlow})(parent(a))
    else
        (a.uplo == 'U' ? MMatrix{T,2,X,:isymmetricup} : MMatrix{T,2,X,:symmetriclow})(parent(a))
    end
end

# conversions to base types
# Matrix uses AbstractMatrix
SparseMatrixCSC{Tv,Ti}(a::MMatrix) where {Tv,Ti} = SparseMatrixCSC{Tv,Ti}(parent(a))
unwrap(a::MMatrix) = unwrap(parent(a))
iswrsparse(a::MMatrix) = iswrsparse(a.parent)

Base.convert(::Type{MMatrix}, a::AbstractMatrix) = MMatrix(a)
Base.eltype(::MMatrix{T}) where T = T
mmtype(::MMatrix{<:Any,<:Any,P}) where P = P
mmstorage_type(::MMatrix{<:Any,X}) where X = X

typed_parent(a::MMatrix{<:Any,<:Any,<:Any,:general}) = a.parent
typed_parent(a::MMatrix{<:Any,<:Any,<:Any,:transpose}) = transpose(a.parent)
typed_parent(a::MMatrix{<:Any,<:Any,<:Any,:adjoint}) = adjoint(a.parent)
typed_parent(a::MMatrix{<:Any,<:Any,<:Any,:symmetricup}) = symmetric(a.parent)
typed_parent(a::MMatrix{<:Any,<:Any,<:Any,:symmetriclow}) = symmetric(a.parent, :L)
typed_parent(a::MMatrix{<:Any,<:Any,<:Any,:hermitianup}) = hermitian(a.parent)
typed_parent(a::MMatrix{<:Any,<:Any,<:Any,:hermitianlow}) = hermitian(a.parent, :L)

# inversion
adjoint(a::MMatrix) = MMatrix(adjoint(parent(a)))
transpose(a::MMatrix) = MMatrix(transpose(parent(a)))
symmetric(a::MMatrix, s::Symbol=:U) = MMatrix(symmetric(parent(a), s))
hermitian(a::MMatrix, s::Symbol=:U) = MMatrix(hermitian(parent(a), s))

# AbstractArray interface
Base.getindex(a::MMatrix, i...) = getindex(parent(a), i...)
Base.setindex!(a::MMatrix, v, i...) = setindex!(parent(a), v, i...)
Base.size(a::MMatrix) = size(parent(a))

# fallback to Matrix or SparseMatrixCSC 
import Base: +, -, *, /, \
for op in (:+, :-, :*, :/, :\)
    for ty in (Number, AbstractVector, AbstractMatrix)
        @eval ($op)(a::MMatrix, b::$ty) = MMatrix(($op)(unwrap(a), unwrap(b))) 
    end
    for ty in (Number, Adjoint{<:Any,<:AbstractVector}, Transpose{<:Any,<:AbstractVector}, AbstractMatrix)
        @eval ($op)(a::$ty, b::MMatrix) = MMatrix(($op)(unwrap(a), unwrap(b))) 
    end 
    @eval ($op)(a::MMatrix, b::MMatrix) = MMatrix(($op)(unwrap(a), unwrap(b))) 
end

