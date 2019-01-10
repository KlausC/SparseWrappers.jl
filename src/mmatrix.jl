"""
    MMatrix(A::AbstractMatrix)

MMatrix is a type concentrating on the mathematical properties of a Matrix.
That is in opposition to wrappers like Symmetric, Adjoint, etc., which have a bias 
on the internal representation.
MMatrix(Symmetric(A)) has type `MMatrix{T,SparseMatrixCSC{T},:upper,:symm,:ident,:ident}`,
which allows to specialize methods by the storage type and the symmetry property.

The type prameters after the storage type are in detail:
  - Bands (:all, :upper, :lower, :diagonal)
  - Moves (:ident, :transpose, :symmetry)
  - Transform (:ident, :conjugate)
  - Diag (:ident, :hermitian, :ones)

Some are restricted by element type of the matrix, so :conjugate and :hermitian are not
supported for Real arguments. :diagonal implies Moves(:ident).
Other combinations of type parameters describe differnt combinations of standard wrappers.
TODO: We will see how the open semigroup generated by the standard wrappers looks like.
"""
struct MMatrix{Tv,N,S<:AbstractArray{<:Any,N},B,M,T,D} <:AbstractArray{Tv,N}
    parent::S
end

const StorageMatrix{Tv} = Union{StridedArray{Tv},SparseMatrixCSC{Tv}}
const StorageArray{Tv} = Union{StorageArray{Tv},SparseVector{Tv}}

Base.parent(ma::MMatrix) = wparent(ma)
Base.show(io::IO, ma::MMatrix) = show(io, wparent(ma))

# MMatrix of storage types
MMatrix(a::S) where {Tv,S<:StorageMatrix{Tv}} = MMatrix{Tv,2,S,:all,:ident,:ident,:ident}(a)
MMatrix(a::S) where {Tv,S<:SparseVector{Tv}} = MMatrix{Tv,1,S,:all,:ident,:ident,:ident}(a)
function MMatrix(a::S) where {Tv,X<:StorageArray{Tv},S<:Transpose{Tv,X}}
    T<:Real || throw(ArgumentError("transpose of complex not supported")) 
    return MMatrix{T,2,X,:all,:transpose,:ident,:ident}(parent(a)) 
end
function MMatrix(a::S) where {Tv<:Real,N,X<:StorageArray{Tv},S<:Adjoint{Tv,X}}
    MMatrix{T,N,X,:all,:transpose,:ident,:ident}(parent(a)) 
end
function MMatrix(a::S) where {Tv,N,X<:StorageArray{Tv},S<:Adjoint{Tv,X}}
    MMatrix{T,N,X,:all,:transpose,:conjugate,:hermitian}(parent(a)) 
end
function MMatrix(a::S) where {Tv,X<:StorageMatrix,S<:Symmetric{Tv,X}}
    T<:Real || throw(ArgumentError("symmetric of complex not supported")) 
    uplo = a.uplo == 'U' ? :upper : lower
    MMatrix{Tv,2,X,uplo,:symmetric,:ident,:ident}(parent(a))
end
function MMatrix(a::S) where {Tv,X<:StorageMatrix,S<:Hermitian{Tv,X}}
    if ! (Tv<:Real)
        isreal(diag(a)) || throw(ArgumentError("Hermitian needs real diagonal")) 
        uplo = a.uplo == 'U' ? :upper : lower
        MMatrix{Tv,2,X,uplo,:symmetric,:conjugate,:hermitian}(parent(a))
    else
        uplo = a.uplo == 'U' ? :upper : lower
        MMatrix{Tv,2,X,uplo,:symmetric,:ident,:ident}(parent(a))
    end
end
function MMatrix(a::S) where {Tv,X<:StorageMatrix,S<:AbstractTriangular{Tv,X}}
    uplo = S isa Union{UpperTriangular,UnitUpperTriangular} ? :upper : :lower
    unit = S isa Union{UnitUpperTriangular,UnitLowerTriangular} :ones : :ident
    MMatrix{Tv,2,X,uplo,:ident,:ident,unit}(parent(a))
end

# MMatrix of wrapped types
for wr in (Transpose,Adjoint,Symmetric,Hermitian,
           UpperTriangular,LowerTriangular,UnitUpperTriangular,UnitLowerTriangular)

    @eval MMatrix(a::S) where {Tv,X,S<:$wr{Tv,X}} = MMatrix(MMatrix(parent(a)))
end
# conversions to base types
# Matrix uses AbstractMatrix
SparseMatrixCSC{Tv,Ti}(a::MMatrix) where {Tv,Ti} = SparseMatrixCSC{Tv,Ti}(wparent(a))
unwrap(a::MMatrix) = unwrap(wparent(a))
iswrsparse(a::MMatrix) = iswrsparse(a.parent)

Base.convert(::Type{MMatrix}, a::AbstractMatrix) = MMatrix(a)
Base.eltype(::MMatrix{T}) where T = T
mmtype(::MMatrix{<:Any,<:Any,B,M,T,D}) where {B,M,T,D} = (B,M,T,D)
mmstorage_type(::MMatrix{<:Any,X}) where X = X

wparent(a::MMatrix{<:Any,<:Any,<:Any,:all,:ident,:ident,:ident}) = a.parent
wparent(a::MMatrix{<:Any,<:Any,<:Any,:all,:transpose,:ident,:ident}) = transpose(a.parent)
wparent(a::MMatrix{<:Any,<:Any,<:Any,:all,:transpose,:conjugate,:ident}) = adjoint(a.parent)
wparent(a::MMatrix{<:Any,<:Any,<:Any,:upper,:symmetry,:ident,:ident}) = symmetric(a.parent)
wparent(a::MMatrix{<:Any,<:Any,<:Any,:lower,:symmetry,:ident,:ident}) = symmetric(a.parent, :L)
wparent(a::MMatrix{<:Any,<:Any,<:Any,:upper,:symmetry,:conjugate,:hermitian}) = hermitian(a.parent)
wparent(a::MMatrix{<:Any,<:Any,<:Any,:lower,:symmetry,:conjugate,:hermitian}) = hermitian(a.parent, :L)
wparent(a::MMatrix{<:Any,<:Any,<:Any,:upper,:ident,:ident,:ident}) = UpperTriangular(a.parent)
wparent(a::MMatrix{<:Any,<:Any,<:Any,:lower,:ident,:ident,:ident}) = LowerTriangular(a.parent)
wparent(a::MMatrix{<:Any,<:Any,<:Any,:upper,:ident,:ident,:ones}) = UnitUpperTriangular(a.parent)
wparent(a::MMatrix{<:Any,<:Any,<:Any,:lower,:ident,:ident,:ones}) = UnitLowerTriangular(a.parent)

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

