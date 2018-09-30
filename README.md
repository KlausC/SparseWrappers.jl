# SparseWrappers.jl
sandbox to improve operations of linear algebra wrapping abstract matrices

There are some Wrappert types, which provide a specialized view to an arbitrary `AbstractMatrix`.
There are at least two important aspects that justify their existence

  1. Storage space economy
  2. Specialization of algorithms applied on them

The existing wrappers of that kind are

  1. UpperTriangular
  2. LowerTriangular
  3. UnitUpperTriangular
  4. UnitLowerTriangular
  5. Symmetric(_, :U)
  6. Symmetric(_, :L)
  7. Hermitian(_, :U)
  8. Hermitian(_, :L)
  9. Transpose
  10. Adjoint

Items 1. - 4. have a common supertype `LinearAlgebra.AbstractTriangular`.

Items 5. - 6. and 7. - 8. share a common type, while their specialization to `:U` and
`:L` is implemented by a data field `uplo`.

Items 1. - 8. have a common field `data`, which contains the referred abstract matrix.
Items 9. - 10. use field `parent` to refer to the abstract matrix.

The types of all items are subtypes of `AbstractMatrix`. So it is possible to combine them
arbitrarily. For example `Symmetric(Hermitian(UnitUpperTriangular([3+im 4+im;5 6]), :U), :U)` is a valid combination of wrappers. Intuitively, it should be equal to `[3 4+im;4+im 6]`.

The ability to combine those wrappers liberately leads to an unlimited number of types,
which makes it hard to design specialized algoritms for them, which are efficient.

This project was set up in order to improve this situation

##### Restrict Number of Types of Combinations of Wrappers

  1. Identify a limited set of wrapped types, which can express all results of wrappers.
  2. Maybe additional types to be added to above list
  3. Maybe restrict support to "useful" combinations (is Symmetric(A) useful
   if A is complex?)


##### Selected Operations for Wrapped Types

  1. sparse(A::X) for all wrapped types X should be as efficient as    sparse(::Adjoint(SparseMatrixCSC)))
  2. all unary and binary operations with wrapped types of sparse matrices should take 
  advantage of the sparsity structure, as far as possible
  3. if no specialized algorithms are availble, in the case of wrapped sparse matrices,
  the fallback should avoid the generic methods for `AbstractMatrix`, but use corresponding
  methods for `AbstractSparseMatrix`.


##### Issues

  1. Adjoint(Transpose)
  2. Hermitian(A) when diag(A) is not real should throw exception
  3. Symmetric(Symmetric, :U), :L) should be Diagonal
  4. Symmetric(Hermitian, :U), :L) should be Diagonal


  
 




  

