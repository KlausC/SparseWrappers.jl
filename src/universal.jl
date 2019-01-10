abstract type Bands end
struct AllBands <: Bands; end   # all parrent array
struct Upper <: Bands; end      # select upper bands of square matrix
struct Lower <: Bands; end      # select lower bands of square matrix
struct DiagBand <:Bands; end    # select diagonal of square matrix

abstract type Moves end
struct NoMove <: Moves; end     # original poition of elements
struct Transposing <: Moves; end # transposed position for off-diagonal
struct Symmetry <: Moves; end   # symmetric position for off-diagonal

abstract type Transform end 
struct NoTransform <: Transform; end    # unchanged (for real spaces)
struct ConjTransform <: Transform; end  # conjugated (for complex spaces)

abstract type Diag end
struct NoDiag <: Diag; end  # diagonal element unchanged
struct HermDiag <:Diag; end # diagolal elements transformed to real
struct OneDiag <: Diag; end # diagnal elements transformed to ones

"""
    UniversalWrapper(::AbstractMatrix)

UniversalWrapper generalizes the well-known wrappers `UpperTriangular, Symmetric, Adjoint,
Diagonal, SubArray, ...` in a more flexible way.
Each wrapper is considered as an operation performed on a subset of the set of square
matrices, applied to such a matrix (called parent).
The set of operators with the standard composition form a semi-group.
It is the aim of this structure to represent all combinations in one object.
The types are distinguished by a big amount of type parameters to enable easy dispatch
on all properties.
"""
struct UniversalWrapper{Tv,N,P<:AbstractArray{Tv,N},B<:Bands,M<:Moves,Tr<:Transform,D<:Diag} <: AbstractArray{Tv,N}
   parent::P
end

UniversalWrapper(A::P) where {Tv,N,P<:StridedArray{Tv,N}} = UniversalWrapper{Tv,N,P,AllBands,NoMove,NoTransform,NoDiag}(A)

UniversalWrapper(A::P) where {Tv,N,P<:Transpose{Tv,<:AbstractArray{Tv,N}}} = UniversalWrapper{Tv,N,P,AllBands,Transposing,NoTransform,NoDiag}(A)

UniversalWrapper(A::P) where {Tv,N,P<:Adjoint{Tv,<:AbstractArray{Tv,N}}} = UniversalWrapper{Tv,N,P,AllBands,Transposing,ConjTransform,HermDiag}(A)

UniversalWrapper(A::P) where {Tv,N,P<:AbstractTriangular{Tv,<:AbstractArray{Tv,2}}} = UniversalWrapper{Tv,2,P,ifelse(P<:Union{UpperTriangular,UnitUpperTriangular},Upper,Lower),NoMove,NoTransform,ifelse(P<:Union{UnitUpperTriangular,UnitLowerTriangular},OneDiag,NoDiag)}(A)

UniversalWrapper(A::P) where {Tv,N,P<:Symmetric{Tv,<:AbstractArray{Tv,2}}} = UniversalWrapper{Tv,2,P,ifelse(A.uplo=='U',Upper,Lower),Symmetry,NoTransform,NoDiag}(A)

UniversalWrapper(A::P) where {Tv,N,P<:Hermitian{Tv,<:AbstractArray{Tv,2}}} = UniversalWrapper{Tv,2,P,ifelse(A.uplo=='U',Upper,Lower),Symmetry,ConjTransform,HermDiag}(A)


