# Libraries
using LinearAlgebra
# Functions 
function system_matrix(sysDim::Int64)
    diagonalMain = 3*ones(sysDim)
    diagonalUpperLower = -1*ones(sysDim-1)
    mtxA = convert(Matrix{Float64}, Tridiagonal(diagonalUpperLower, diagonalMain, diagonalUpperLower))
    for i = 1:sysDim
        if mtxA[i, sysDim - i + 1] == 0.0
            mtxA[i, sysDim - i + 1] = 0.5
        end
    end
    return mtxA
end

function vector_independant_terms(sysDim::Int64)
    vtrB = Vector{Float64}(undef, sysDim)
    vtrB[1] = 2.5
    vtrB[sysDim] = 2.5
    position::Int64 = floor(sysDim/2)
    for i = 2:sysDim-1
        if i == position || i == position+1
            vtrB[i] = 1.0
        else
            vtrB[i] = 1.5
        end
    end
    return vtrB
end

function conjugate_gradient_method( sysDim::Int64,
                                    mtxA::Matrix{Float64},
                                    vtrB::Vector{Float64},
                                    vtrExactSol::Vector{Float64},
                                    vtrX_initial::Vector{Float64},
                                    tolerance::Float64 )
    maximumIteration = sysDim
    methodError = 0.0
    iterationNumber = 0

    vtrX_current = vtrX_initial
    vtrR_current = vtrB - mtxA * vtrX_current
    vtrD_current = vtrR_current
    vtrAproxSol = Vector{Float64}(undef, sysDim)
    
    for k = 1:maximumIteration
        vectorAD = mtxA * vtrD_current
        alpha = dot(vtrR_current, vtrR_current) / dot(vectorAD, vtrD_current)
        vtrX_next = vtrX_current + alpha * vtrD_current
        methodError = norm(vtrX_next - vtrExactSol) / norm(vtrX_current)
        # Convergence Criteria
        if methodError <= tolerance
            iterationNumber = k
            vtrAproxSol = vtrX_next
            break
        end
        vtrR_next = vtrR_current - alpha * vectorAD
        betha = dot(vtrR_next, vtrR_next) / dot(vtrR_current, vtrR_current)
        vtrD_next = vtrR_next + betha * vtrD_current
        # Update
        vtrX_current = vtrX_next
        vtrR_current = vtrR_next
        vtrD_current = vtrD_next
    end

    return methodError, iterationNumber
end
