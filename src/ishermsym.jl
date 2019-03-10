import SparseArrays: SparseMatrixCSC, SparseMatrixCSCInterface, is_hermsym
is_hermsym(A::Number, check::Function) = A == check(A)

function is_hermsym3(A::SparseMatrixCSCInterface{Tv,Ti}, check::Function) where {Tv,Ti}
    m, n = size(A)
    m == n || return false
    rowval = rowvals(A)
    nzval = nonzeros(A)
    start = Vector{Int}(undef, n)
    toend = A isa SparseMatrixCSC || last(A.indices[1]) == size(parent(A), 1)
    stop = toend ? parent(A).colptr : Vector{Ti}(undef, n+1)
    for col = 1:n
        nzr = nzrange(A, col)
        if !toend
            stop[col+1] = last(nzr) + 1
        end
        start[col] = stop[col+1]
        for j in nzr
            row = rowval[j]
            row == 0 && continue
            if row >= col
                if row == col
                    is_hermsym(nzval[j], check) || return false
                    start[col] = j + 1
                 else
                    start[col] = j 
                end
                break
            else # row < col
                nz = nzval[j]
                found = false
                for k = start[row]:stop[row+1]-1
                    mcol = rowval[k]
                    if mcol < col
                        iszero(nzval[k]) || return false
                    elseif mcol == col
                        nzval[k] == check(nz) || return false
                        start[row] = k + 1
                        found = true
                        break
                    else
                        start[row] = k
                        break
                    end
                end
                found || iszero(nz) || return false
            end
        end
    end
    # return all(iszero(nzval[k]) for row = 1:n, k = start[row]:stop[row+1]-1)
    for row = 1:n, k = start[row]:stop[row+1]-1
        iszero(nzval[k]) || return false
    end
    return true
end

function is_hermsym2(A::SubArray{<:Any,2,<:SparseMatrixCSC,<:Tuple{<:StepRange{<:Integer,<:Integer},<:AbstractVector{<:Integer}},false}, check::Function)

    st = step(A.indices[1])
    if st == 1
        ind = (first(A.indices[1]):last(A.indices[1]), A.indices[2])
    elseif st == -1
        ind = (last(A.indices[1]):first(A.indices[1]), reverse(A.indices[2]))
    else
        throw(ArgumentError("view step of first index is $st not in {1, -1}"))
    end
    is_hermsym2(view(parent(A), ind...), check)
end

function is_hermsym2(A::SparseMatrixCSCInterface, check::Function)
    m, n = size(A)
    if m != n; return false; end

    rowval = rowvals(A)
    nzval = nonzeros(A)
    tracker, stop = tracker_stop(A)
    for col = 1:n
        # `tracker` is updated such that, for symmetric matrices,
        # the loop below starts from an element at or below the
        # diagonal element of column `col`"
        for p = tracker[col]:stop[col+1]-1
            val = nzval[p]
            row = rowval[p]

            # Ignore stored zeros
            if iszero(val)
                continue
            end

            # If the matrix was symmetric we should have updated
            # the tracker to start at the diagonal or below. Here
            # we are above the diagonal so the matrix can't be symmetric.
            if row < col
                return false
            end

            # Diagonal element
            if row == col
                if !is_hermsym(val, check)
                    return false
                end
            else
                offset = tracker[row]

                # If the matrix is unsymmetric, there might not exist
                # a rowval[offset]
                if offset > length(rowval)
                    return false
                end

                row2 = rowval[offset]

                # row2 can be less than col if the tracker didn't
                # get updated due to stored zeros in previous elements.
                # We therefore "catch up" here while making sure that
                # the elements are actually zero.
                while row2 < col
                    if !iszero(nzval[offset])
                        return false
                    end
                    offset += 1
                    row2 = rowval[offset]
                    tracker[row] += 1
                end

                # Non zero A[i,j] exists but A[j,i] does not exist
                if row2 > col
                    return false
                end

                # A[i,j] and A[j,i] exists
                if row2 == col
                    if val != check(nzval[offset])
                        return false
                    end
                    tracker[row] += 1
                end
            end
        end
    end
    return true
end

function tracker_stop(A::SparseMatrixCSC)
    copy(A.colptr), A.colptr
end
function tracker_stop(A::SparseMatrixCSCInterface{Tv,Ti}) where {Tv,Ti}
    n = size(A, 2)
    stop = Vector{Ti}(undef, n+1)
    stop[1] = 1
    tracker = Vector{Ti}(undef, n)
    for col = 1:n
        nzr = nzrange(A, col)
        tracker[col] = first(nzr)
        stop[col+1] = last(nzr) + 1
    end
    tracker, stop
end

