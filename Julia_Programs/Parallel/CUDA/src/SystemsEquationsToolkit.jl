# Libraries
using LinearAlgebra: Tridiagonal
# Functions 
# System Equation Linear Sparse
function system_matrix_sparse( systemDimension::Integer, mtxA::AbstractMatrix )
    vtrDiagonalMain = 3*ones(systemDimension)
    vtrDiagonalUpperLower = -1*ones(systemDimension-1)
    mtxA .= convert(Matrix, Tridiagonal(vtrDiagonalUpperLower, vtrDiagonalMain, vtrDiagonalUpperLower))
    for indexI = 1:systemDimension
        if mtxA[indexI, systemDimension-indexI+1] == 0.0
            mtxA[indexI, systemDimension-indexI+1] = 0.5
        end
    end
    return nothing
end
# 
function vector_independent_term( systemDimension::Integer, vtrB::AbstractVector )
    vtrB[1] = 2.5
    vtrB[systemDimension] = 2.5
    position::Integer = floor(systemDimension/2)
    for indexI = 2:systemDimension-1
        if indexI == position || indexI == position+1
            vtrB[indexI] = 1.0
        else
            vtrB[indexI] = 1.5
        end
    end
    return nothing
end