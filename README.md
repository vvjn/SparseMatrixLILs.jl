# SparseMatrixLILs

[![Build
 Status](https://travis-ci.org/vvjn/SparseMatrixLILs.jl.svg?branch=master)](https://travis-ci.org/vvjn/SparseMatrixLILs.jl)
 [![codecov.io](http://codecov.io/github/vvjn/SparseMatrixLILs.jl/coverage.svg?branch=master)](http://codecov.io/github/vvjn/SparseMatrixLILs.jl?branch=master)


This package implements a sparse matrix data structure stored as a
list of lists (namely, `Vector{SparseVector{Tv,Ti}}`). 

The `SparseMatrixCSC` data structure needs to copy/update the entire
matrix every time that a non-stored value is updated, most of the
time. That is, when `setindex!` is used to update a non-stored
position, `SparseMatrixCSC` has a time complexity of `O(nnz)`, where
`nnz` is the number of non-zeros.

The list of lists (LIL) sparse matrix format, `SparseMatrixLIL` allows
for faster incremental sparse matrix construction. `setindex!` for
`SparseMatrixLIL` has a time complexity of `O(d)` when you update a
non-stored value, where `d` is the number of non-zeros in a column of
the matrix.

This package is primarily meant to be used by [DynaWAVE.jl](https://github.com/vvjn/DynaWAVE.jl).


# Installation

```julia
Pkg.clone("https://github.com/vvjn/SparseMatrixLILs.jl")
```
