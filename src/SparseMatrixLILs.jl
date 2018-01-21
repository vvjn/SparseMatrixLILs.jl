module SparseMatrixLILs

import Base: getindex, size, setindex!, length, findnz, nnz, +, -, map, map!, /, \, *
export SparseMatrixLIL

"""
Store sparse matrix as as vector of sparse vectors (list of lists)
This is to enable more efficient updates (i.e. setindex!)
Each vector represents a COLUMN on the matrix
"""
immutable SparseMatrixLIL{Tv,Ti<:Integer} <: AbstractSparseMatrix{Tv,Ti}
    data :: Vector{SparseVector{Tv,Ti}}
    m :: Int
    n :: Int
end

"""
        SparseMatrixLIL(A :: SparseMatrixLIL)

Returns A without copying.

        SparseMatrixLIL(A :: SparseMatrixCSC{Tv,Ti})

Converts to `SparseMatrixLIL` format (copying).

        SparseMatrixLIL(A :: AbstractMatrix{Tv})

Converts to `SparseMatrixLIL` format (copying).
"""
SparseMatrixLIL(A :: SparseMatrixLIL) = A
SparseMatrixLIL(A :: SparseMatrixCSC{Tv,Ti}) where {Tv,Ti<:Integer} =
    SparseMatrixLIL{Tv,Ti}(map(v -> A[:,v], 1:size(A,2)), size(A)...)
SparseMatrixLIL(A :: AbstractMatrix{Tv}) where {Tv} =
    SparseMatrixLIL{Tv,Int}(map(v -> sparsevec(A[:,v]), 1:size(A,2)), size(A)...)

size(A::SparseMatrixLIL) = (A.m,A.n)
size(A::SparseMatrixLIL,d::Integer) = (A.m,A.n)[d]
length(A::SparseMatrixLIL) = A.m * A.n

@inline getindex(A::SparseMatrixLIL, i::Integer, j::Integer) = A.data[j][i]
getindex(A::SparseMatrixLIL, I::AbstractVector, j::Integer) = SparseMatrixLIL([A.data[j][I]])

function getindex(A::SparseMatrixLIL{Tv,Ti}, I::AbstractVector, J::AbstractVector) where {Tv,Ti<:Integer}
    data = Vector{SparseVector{Tv,Ti}}(length(J))
    for ji in 1:length(J)
        data[ji] = A.data[J[ji]][I]
    end
    SparseMatrixLIL{Tv,Ti}(data, length(I), length(J))
end

@inline setindex!(A::SparseMatrixLIL, x, i::Integer, j::Integer) = A.data[j][i] = x
@inline setindex!(A::SparseMatrixLIL, x::AbstractVector, I::AbstractVector, j::Integer) = A.data[j][I] = x
@inline function setindex!(A::SparseMatrixLIL{Tv,Ti}, x::AbstractMatrix, I::AbstractVector, J::AbstractVector) where {Tv,Ti<:Integer}
    for ji in 1:length(J)
        A.data[J[ji]][I] = view(x, :, ji)
    end
    x
end

function nnz(A::SparseMatrixLIL{Tv,Ti}) where {Tv,Ti<:Integer}
    n = 0
    for j = 1:A.n
        n += nnz(A.data[j])
    end
    n
end

function findnz(S::SparseMatrixLIL{Tv,Ti}) where {Tv,Ti}
    numnz = nnz(S)
    I = Vector{Ti}(numnz)
    J = Vector{Ti}(numnz)
    V = Vector{Tv}(numnz)

    count = 1
    for col = 1:S.n
        nzval = S.data[col].nzval
        nzind = S.data[col].nzind
        for u = 1:length(nzind)            
            if nzval[u] != 0
                I[count] = nzind[u]
                J[count] = col
                V[count] = nzval[u]
                count += 1
            end
        end
    end

    count -= 1
    if numnz != count
        deleteat!(I, (count+1):numnz)
        deleteat!(J, (count+1):numnz)
        deleteat!(V, (count+1):numnz)
    end

    return (I, J, V)
end

map(f::Tf, A::SparseMatrixLIL) where {Tf} =
    SparseMatrixLIL(map(v -> map(f,v), A.data), A.m, A.n)

map(f::Tf, A::SparseMatrixLIL, Bs::Vararg{SparseMatrixLIL,N}) where {Tf,N} =
    SparseMatrixLIL(map((v...) -> map(f,v...), A.data, map(B -> B.data, Bs)...), A.m, A.n)

function map!(f::Tf, C::SparseMatrixLIL, A::SparseMatrixLIL, Bs::Vararg{SparseMatrixLIL,N}) where {Tf,N}
    for i = 1:C.n
        map!(f, C.data[i], A.data[i], map(B -> B.data[i], Bs)...)
    end
    C
end

broadcast(f::Tf, A::SparseMatrixLIL) where {Tf} =
    SparseMatrixLIL(map(v -> map(f,v), A.data), A.m, A.n)

function broadcast(f::Tf, A::SparseMatrixLIL, Bs::Vararg{SparseMatrixLIL,N}) where {Tf,N}
    #shape = Base.Broadcast.broadcast_indices(A.data, map(B -> B.data, Bs)...)
    #n = Base.Broadcast.broadcast_shape(size(A.data), map(B -> size(B.data), Bs)...)[1]
    #keeps, Idefaults = Base.Broadcast.map_newindexer(shape, A, Bs)
    for B in Bs
        if A.n != B.n
            error("not implemented")
        end
    end
    data = Vector{SparseVector{Tv,Ti}}(A.n)
    for i = 1:A.n
        data[i] = broadcast(f, A.data[i], map(B -> B.data[i], Bs)...)
    end
    # SparseMatrixLIL(map((v...) -> map(f,v...), A.data, map(B -> B.data, Bs)...), A.m, A.n)
    SparseMatrixLIL(data, length(data[1]), A.n)
end

function broadcast!(f::Tf, C::SparseMatrixLIL, A::SparseMatrixLIL, Bs::Vararg{SparseMatrixLIL,N}) where {Tf,N}
    for i = 1:C.n
        broadcast!(f, C.data[i], A.data[i], map(B -> B.data[i], Bs)...)
    end
    C
end

## Binary arithmetic and boolean operators
(+)(A::SparseMatrixLIL, B::SparseMatrixLIL) = map(+, A, B)
(-)(A::SparseMatrixLIL, B::SparseMatrixLIL) = map(-, A, B)
(+)(A::SparseMatrixLIL, B::Array) = Array(A) + B
(+)(A::Array, B::SparseMatrixLIL) = A + Array(B)
(-)(A::SparseMatrixLIL, B::Array) = Array(A) - B
(-)(A::Array, B::SparseMatrixLIL) = A - Array(B)

for f in (:/, :\, :*)
    if f != :/
        @eval ($f)(A::Number, B::SparseMatrixLIL) = SparseMatrixLIL(map(v -> ($f)(A,v), B.data), B.m, B.n)
    end
    if f != :\
        @eval ($f)(A::SparseMatrixLIL, B::Number) = SparseMatrixLIL(map(v -> ($f)(v,B), A.data), A.m, A.n)
    end
end

end
