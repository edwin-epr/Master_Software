# =================================================================
# Methods for solve Systems of Equations Linears
# =================================================================
using LinearAlgebra: dot, norm
using SparseArrays: AbstractSparseMatrix 
# =================================================================
# Jacobi Method
# =================================================================
function jacobi_method( 
    matrixD::AbstractMatrix,
    matrixL::AbstractMatrix,
    matrixU::AbstractMatrix,
    vectorb::AbstractVector,
    vectorx_exactsolution::AbstractVector,
    vectorx_initial::AbstractVector,
    maximumiteration::Integer,
    tolerance::AbstractFloat )
iterationnumber = 0
methoderror = 0.0
vectorx_current = vectorx_initial
matrixsparseinvD = sparse(inv(matrixD))
matrixsparseT = sparse(matrixL + matrixU)
for indexk = 1:maximumiteration
    vectorx_next = matrixsparseinvD * ( vectorb - matrixsparseT * vectorx_current )
    methoderror = norm( vectorx_next - vectorx_exactsolution) / norm(vectorx_next)
    # Stop Condition
    if methoderror <= tolerance
        iterationnumber = indexk
        break
    end
    # Update 
    vectorx_current = vectorx_next
end
# Method information
println("The number of iterations is: $(iterationnumber)")
println("The method error is: $(methoderror)")
return nothing
end
# =================================================================
# Jacobi Method (for numerical expirement)
# =================================================================
function jacobi_method!( 
    matrixD::AbstractMatrix,
    matrixL::AbstractMatrix,
    matrixU::AbstractMatrix,
    vectorb::AbstractVector,
    vectorx_exactsolution::AbstractVector,
    vectorx_initial::AbstractVector,
    maximumiteration::Integer, 
    informationprint::Bool )
methoderror = 0.0
vectorx_current = vectorx_initial
matrixsparseinvD = sparse(inv(matrixD))
matrixsparseT = sparse(matrixL + matrixU)
for indexk = 1:maximumiteration
    vectorx_next = matrixsparseinvD * ( vectorb - matrixsparseT * vectorx_current )
    # For mesure method error
    if indexk == maximumiteration
        methoderror = norm( vectorx_next - vectorx_exactsolution) / norm(vectorx_next)
        break
    end
    # Update 
    vectorx_current = vectorx_next
end
# Method information
if informationprint == true
    println("For $(maximumiteration) iterations")
    println("the method error is: $(methoderror)")
    return methoderror
end
return nothing
end
# =================================================================
# Gauss-Seidel Method
# =================================================================
function gauss_seidel_method( 
    matrixD::AbstractMatrix,
    matrixL::AbstractMatrix,
    matrixU::AbstractMatrix,
    vectorb::AbstractVector,
    vectorx_exactsolution::AbstractVector,
    vectorx_initial::AbstractVector,
    maximumiteration::Integer,
    tolerance::AbstractFloat )
iterationnumber = 0
methoderror = 0.0
vectorx_current = vectorx_initial
matrixT = matrixD+matrixL
matrixInvT = inv(matrixT)
matrixUSparse = sparse(matrixU)
for indexk = 1:maximumiteration
    vectorx_next = matrixInvT * ( vectorb - matrixUSparse * vectorx_current )
    methoderror = norm( vectorx_next - vectorx_exactsolution ) / norm( vectorx_next )
    # Stop Condition
    if methoderror <= tolerance
        iterationnumber = indexk
        break
    end
    # Update
    vectorx_current = vectorx_next
end
# Method information
println("The number of iterations is: $(iterationnumber)")
println("The method error is: $(methoderror)")
return nothing
end
# =================================================================
# Gauss-Seidel Method (for numerical expirements)
# =================================================================
function gauss_seidel_method!( 
    matrixD::AbstractMatrix,
    matrixL::AbstractMatrix,
    matrixU::AbstractMatrix,
    vectorb::AbstractVector,
    vectorx_exactsolution::AbstractVector,
    vectorx_initial::AbstractVector,
    maximumiteration::Integer,
    informationprint::Bool )
methoderror = 0.0
vectorx_current = vectorx_initial
matrixT = matrixD+matrixL
matrixInvT = inv(matrixT)
matrixUSparse = sparse(matrixU)
for indexk = 1:maximumiteration
    vectorx_next = matrixInvT * ( vectorb - matrixUSparse * vectorx_current )
    # Stop Condition
    if indexk == maximumiteration
        methoderror = norm( vectorx_next - vectorx_exactsolution ) / norm( vectorx_next )
        break
    end
    # Update
    vectorx_current = vectorx_next
end
# Method information
if informationprint == true
    println("For $(maximumiteration) iterations")
    println("the method error is: $(methoderror)")
    return methoderror
end
return nothing
end
# =================================================================
# Sucessive Over-Relaxation Method
# =================================================================
function sor_method( 
    matrixD::AbstractMatrix,
    matrixL::AbstractMatrix,
    matrixU::AbstractMatrix,
    vectorb::AbstractVector,
    vectorx_exactsolution::AbstractVector,
    vectorx_initial::AbstractVector,
    maximumiteration::Integer,
    relaxationParameter::AbstractFloat,
    tolerance::AbstractFloat )
    omega = relaxationParameter
    iterationnumber = 0
    methoderror = 0.0
    vectorx_current = vectorx_initial
    matrixT = matrixD + omega * matrixL
    matrixInvT = inv(matrixT)
    matrixOmegaDSparse = sparse((1 - omega) * matrixD)
    matrixOmegaUSparse = sparse((omega * matrixU))
    vectorF = omega * matrixInvT * vectorb
    for indexk = 1:maximumiteration
        vectorx_next = matrixInvT * (matrixOmegaDSparse * vectorx_current - matrixOmegaUSparse * vectorx_current) + vectorF
        methoderror = norm( vectorx_next - vectorx_exactsolution ) / norm( vectorx_next )
        # Stop Condition
        if methoderror <= tolerance
            iterationnumber = indexk
            break
        end
        # Update
        vectorx_current = vectorx_next
    end
    # Method information
    println("The number of iterations is: $(iterationnumber)")
    println("The method error is: $(methoderror)")
    return nothing
end
# =================================================================
# Sucessive Over-Relaxation Method (for numerical expirements)
# =================================================================
function sor_method!( 
    matrixD::AbstractMatrix,
    matrixL::AbstractMatrix,
    matrixU::AbstractMatrix,
    vectorb::AbstractVector,
    vectorx_exactsolution::AbstractVector,
    vectorx_initial::AbstractVector,
    maximumiteration::Integer,
    relaxationParameter::AbstractFloat,
    informationprint::Bool )
    omega = relaxationParameter
    iterationnumber = 0
    methoderror = 0.0
    vectorx_current = vectorx_initial
    matrixT = matrixD + omega * matrixL
    matrixInvT = inv(matrixT)
    matrixOmegaDSparse = sparse((1 - omega) * matrixD)
    matrixOmegaUSparse = sparse((omega * matrixU))
    vectorF = omega * matrixInvT * vectorb
    for indexk = 1:maximumiteration
        vectorx_next = matrixInvT * (matrixOmegaDSparse * vectorx_current - matrixOmegaUSparse * vectorx_current) + vectorF
        # Stop Condition
        if indexk == maximumiteration
            methoderror = norm( vectorx_next - vectorx_exactsolution ) / norm( vectorx_next )
            break
        end
        # Update
        vectorx_current = vectorx_next
    end
    # Method information
    if informationprint == true
        println("For $(maximumiteration) iterations")
        println("the method error is: $(methoderror)")
        return methoderror
    end
    return nothing
end
# =================================================================
# Conjugate Gradient Method
# =================================================================
function conjugate_gradient_method( 
    systemdimension::Integer,
    matrixA::AbstractSparseMatrix,
    vectorb::AbstractVector,
    vectorx_exactsolution::AbstractVector,
    vectorx_initial::AbstractVector,
    tolerance::AbstractFloat )
    maximumiteration = systemdimension
    methoderror = 0.0
    iterationnumber = 0
    vectorx_current = vectorx_initial
    vectorR_current = vectorb - matrixA * vectorx_current
    vectorD_current = vectorR_current
    for indexk = 1:maximumiteration
        vectorAD = matrixA * vectorD_current
        alpha = dot(vectorR_current, vectorR_current) / dot(vectorAD, vectorD_current)
        vectorx_next = vectorx_current + alpha * vectorD_current
        # Stop Condition
        methoderror = norm(vectorx_next - vectorx_exactsolution) / norm(vectorx_next)
        if methoderror <= tolerance
            iterationnumber = indexk
            break
        end
        vectorR_next = vectorR_current - alpha * vectorAD
        betha = dot(vectorR_next, vectorR_next) / dot(vectorR_current, vectorR_current)
        vectorD_next = vectorR_next + betha * vectorD_current
        # Update
        vectorx_current = vectorx_next
        vectorR_current = vectorR_next
        vectorD_current = vectorD_next
    end
    # Method information
    println("The iteration number is: $(iterationnumber).")
    println("The method error is: $(methoderror).")
    return nothing
end
# =================================================================
# Conjugate Gradient Method (for numerical expirements)
# =================================================================
function conjugate_gradient_method!( 
    matrixA::AbstractSparseMatrix,
    vectorb::AbstractVector,
    vectorx_exactsolution::AbstractVector,
    vectorx_initial::AbstractVector,
    maximumiteration::Integer,
    informationprint::Bool )
    methoderror = 0.0
    vectorx_current = vectorx_initial
    vectorR_current = vectorb - matrixA * vectorx_current
    vectorD_current = vectorR_current
    for indexk = 1:maximumiteration
        vectorAD = matrixA * vectorD_current
        alpha = dot(vectorR_current, vectorR_current) / dot(vectorAD, vectorD_current)
        vectorx_next = vectorx_current + alpha * vectorD_current
        # Stop Condition
        if indexk == maximumiteration
            methoderror = norm(vectorx_next - vectorx_exactsolution) / norm(vectorx_next)
            break
        end
        vectorR_next = vectorR_current - alpha * vectorAD
        betha = dot(vectorR_next, vectorR_next) / dot(vectorR_current, vectorR_current)
        vectorD_next = vectorR_next + betha * vectorD_current
        # Update
        vectorx_current = vectorx_next
        vectorR_current = vectorR_next
        vectorD_current = vectorD_next
    end
    # Output variables
    if informationprint == true
        println("For $(maximumiteration) iterations")
        println("the method error is: $(methoderror)")
        return methoderror
    end
    return nothing
end
# =================================================================
# Biconjugate Gradient Stabilized Method
# =================================================================
function biconjugate_gradient_stabilized_method( 
    systemdimension::Integer,
    matrixA::AbstractSparseMatrix,
    vectorb::AbstractVector,
    vectorx_exactsolution::AbstractVector,
    vectorx_initial::AbstractVector,
    tolerance::AbstractFloat )
    maximumiteration = systemdimension
    methoderror = 0.0
    iterationnumber = 0
    vectorx_current = vectorx_initial
    vectorR_current = vectorb - matrixA * vectorx_current
    vectorD_current = vectorR_current
    vectorRAst = vectorR_current
    for indexk = 1:maximumiteration
        vectorAD = matrixA * vectorD_current
        alpha = dot(vectorR_current, vectorRAst) / dot(vectorAD, vectorRAst)
        vectorS = vectorR_current - alpha * vectorAD
        vectorAS = matrixA * vectorS
        omega = dot(vectorAS, vectorS) / dot(vectorAS, vectorAS)
        vectorx_next = vectorx_current + alpha * vectorD_current + omega * vectorS
        # Stop Condition
        methoderror = norm(vectorx_next - vectorx_exactsolution) / norm(vectorx_next)
        if methoderror <= tolerance
            iterationnumber = indexk
            break
        end
        vectorR_next = vectorS - omega * vectorAS
        betha = (dot(vectorR_next, vectorRAst) / dot(vectorR_current, vectorRAst)) * (alpha / omega)
        vectorD_next = vectorR_next + betha * (vectorD_current - omega * vectorAD)
        # Update
        vectorx_current = vectorx_next
        vectorR_current = vectorR_next
        vectorD_current = vectorD_next
    end
    # Output variables
    println("The iteration number is: $(iterationnumber).")
    println("The method error is: $(methoderror).")
    return nothing
end
# =================================================================
# Biconjugate Gradient Stabilized Method (for numerical expirements)
# =================================================================
function biconjugate_gradient_stabilized_method!( 
    matrixA::AbstractSparseMatrix,
    vectorb::AbstractVector,
    vectorx_exactsolution::AbstractVector,
    vectorx_initial::AbstractVector,
    maximumiteration::Integer,
    informationprint::Bool )
    methoderror = 0.0
    vectorx_current = vectorx_initial
    vectorR_current = vectorb - matrixA * vectorx_current
    vectorD_current = vectorR_current
    vectorRAst = vectorR_current
    for indexk = 1:maximumiteration
        vectorAD = matrixA * vectorD_current
        alpha = dot(vectorR_current, vectorRAst) / dot(vectorAD, vectorRAst)
        vectorS = vectorR_current - alpha * vectorAD
        vectorAS = matrixA * vectorS
        omega = dot(vectorAS, vectorS) / dot(vectorAS, vectorAS)
        vectorx_next = vectorx_current + alpha * vectorD_current + omega * vectorS
        # Stop Condition
        if indexk == maximumiteration
            methoderror = norm(vectorx_next - vectorx_exactsolution) / norm(vectorx_next)
            break
        end
        vectorR_next = vectorS - omega * vectorAS
        betha = (dot(vectorR_next, vectorRAst) / dot(vectorR_current, vectorRAst)) * (alpha / omega)
        vectorD_next = vectorR_next + betha * (vectorD_current - omega * vectorAD)
        # Update
        vectorx_current = vectorx_next
        vectorR_current = vectorR_next
        vectorD_current = vectorD_next
    end
    # Output variables
    if informationprint == true
        println("For $(maximumiteration) iterations")
        println("the method error is: $(methoderror)")
        return methoderror
    end
    return nothing
end
# =================================================================
# Restarted Generalized Minimal Residual Method
# =================================================================
function restarted_generalized_minimal_residual_method( 
    systemdimension::Integer,
    matrixA::AbstractSparseMatrix,
    vectorb::AbstractVector,
    vectorx_initial::AbstractVector,
    vectorx_exactsolution::AbstractVector,
    restartparameter::Integer,
    tolerance::AbstractFloat )
    matrixH = Matrix{Float64}(undef, restartparameter+1, restartparameter)
    matrixQ = Matrix{Float64}(undef, systemdimension, restartparameter+1)
    vectorx_current = vectorx_initial
    methoderror = 0.0
    flag = true
    matrixdimension = restartparameter
    indexk = 0
    while flag == true
        # println("Reinicio:",k)
        vectorR_start = vectorb - matrixA * vectorx_current
        betha = norm(vectorR_start)
        matrixQ[:,1] = vectorR_start / betha
        vectorE = zeros(restartparameter+1)
        vectorE[1] = 1
        vectorbethaE = betha*vectorE
        vectorcoseno = Vector{Float64}(undef, restartparameter)
        vectorseno = Vector{Float64}(undef, restartparameter)
        kappa = 0.38;
        for indexj = 1:restartparameter
            vectorVJ = matrixA * matrixQ[:,indexj]
            normVJ = norm(vectorVJ)
            for indexi = 1:indexj
                matrixH[indexi, indexj] = dot(vectorVJ, matrixQ[:,indexi])
                vectorVJ = vectorVJ - matrixH[indexi, indexj] * matrixQ[:,indexi]
            end
            if norm(vectorVJ) / normVJ <= kappa
                for indexi = 1:indexj
                    vectorP = dot(matrixQ[:, indexi], vectorVJ)
                    vectorVJ = vectorVJ - vectorP * matrixQ[:, indexi]
                    matrixH[indexi, indexj] = matrixH[indexi, indexj] + vectorP
                end
            end
            matrixH[indexj+1, indexj] = norm(vectorVJ)
            if abs(matrixH[indexj+1, indexj]) < tolerance
                println("fin")
                matrixdimension = indexj
                println("$(matrixdimension)")
                break
            end
            matrixQ[:, indexj+1] = vectorVJ / matrixH[indexj+1, indexj]
            for indexi = 1:indexj-1
                auxiliarvalue = matrixH[indexi, indexj]
                matrixH[indexi, indexj] = vectorcoseno[indexi] * auxiliarvalue + vectorseno[indexi] * matrixH[indexi+1, indexj]
                matrixH[indexi+1, indexj] = -vectorseno[indexi] * auxiliarvalue + vectorcoseno[indexi] * matrixH[indexi+1, indexj]
            end
            if (abs(matrixH[indexj, indexj]) > abs(matrixH[indexj+1, indexj]))
                auxiliarvalue1 = matrixH[indexj+1, indexj] / matrixH[indexj, indexj]
                auxiliarvalue2 = sign(matrixH[indexj, indexj]) * sqrt(1 + auxiliarvalue1^2)
                vectorcoseno[indexj] = 1 / auxiliarvalue2
                vectorseno[indexj] = auxiliarvalue1 * vectorcoseno[indexj]
            else
                auxiliarvalue1 =  matrixH[indexj, indexj] / matrixH[indexj+1, indexj]
                auxiliarvalue2 = sign(matrixH[indexj+1, indexj]) * sqrt(1 + auxiliarvalue1^2)
                vectorseno[indexj] = 1 / auxiliarvalue2
                vectorcoseno[indexj] = auxiliarvalue1 * vectorseno[indexj]
            end
            valueHjj = matrixH[indexj, indexj]
            matrixH[indexj, indexj] = vectorcoseno[indexj] * valueHjj + vectorseno[indexj] * matrixH[indexj+1, indexj]
            matrixH[indexj+1, indexj] = -vectorseno[indexj] * valueHjj + vectorcoseno[indexj] * matrixH[indexj+1, indexj]
            vectorbethaE[indexj+1] = -vectorseno[indexj]*vectorbethaE[indexj]
            vectorbethaE[indexj] = vectorcoseno[indexj] * vectorbethaE[indexj]
        end
        for indexi = 1:matrixdimension-1
            auxiliarvalue = matrixH[indexi, matrixdimension]
            matrixH[indexi, matrixdimension] = vectorcoseno[indexi] * auxiliarvalue + vectorseno[indexi] * matrixH[indexi+1, matrixdimension]
            matrixH[indexi+1, matrixdimension] = -vectorseno[indexi] * auxiliarvalue + vectorcoseno[indexi] * matrixH[indexi+1, matrixdimension]
        end
        vectory = backward_substitution(matrixdimension, matrixH, vectorbethaE )
        vectorx_aproximateSolution = vectorx_current + matrixQ[:,1:matrixdimension]*vectory
        methoderror = norm(vectorx_aproximateSolution - vectorx_exactsolution) / (norm(vectorx_aproximateSolution))
        if methoderror < tolerance
            break
        end
        vectorx_current = vectorx_aproximateSolution
        indexk += 1
    end
    println("El número de reinicios es de $(indexk).")
    println("El error del método es: $(methoderror).")
    return nothing
end
# =================================================================
# Restarted Generalized Minimal Residual Method (for numerical expirements)
# =================================================================
function restarted_generalized_minimal_residual_method!( 
    systemdimension::Integer,
    matrixA::AbstractSparseMatrix,
    vectorb::AbstractVector,
    vectorx_initial::AbstractVector,
    vectorx_exactsolution::AbstractVector,
    restartparameter::Integer,
    maximumrestart::Integer,
    tolerance::AbstractFloat,
    informationprint::Bool )
    matrixH = Matrix{Float64}(undef, restartparameter+1, restartparameter)
    matrixQ = Matrix{Float64}(undef, systemdimension, restartparameter+1)
    vectorx_current = vectorx_initial
    methoderror = 0.0
    flag = true
    matrixdimension = restartparameter
    indexk = 0
    while flag == true
        # println("Reinicio:",k)
        vectorR_start = vectorb - matrixA * vectorx_current
        betha = norm(vectorR_start)
        matrixQ[:,1] = vectorR_start / betha
        vectorE = zeros(restartparameter+1)
        vectorE[1] = 1
        vectorbethaE = betha*vectorE
        vectorcoseno = Vector{Float64}(undef, restartparameter)
        vectorseno = Vector{Float64}(undef, restartparameter)
        kappa = 0.38;
        for indexj = 1:restartparameter
            vectorVJ = matrixA * matrixQ[:,indexj]
            normVJ = norm(vectorVJ)
            for indexi = 1:indexj
                matrixH[indexi, indexj] = dot(vectorVJ, matrixQ[:,indexi])
                vectorVJ = vectorVJ - matrixH[indexi, indexj] * matrixQ[:,indexi]
            end
            if norm(vectorVJ) / normVJ <= kappa
                for indexi = 1:indexj
                    vectorP = dot(matrixQ[:, indexi], vectorVJ)
                    vectorVJ = vectorVJ - vectorP * matrixQ[:, indexi]
                    matrixH[indexi, indexj] = matrixH[indexi, indexj] + vectorP
                end
            end
            matrixH[indexj+1, indexj] = norm(vectorVJ)
            if abs(matrixH[indexj+1, indexj]) < tolerance
                println("fin")
                matrixdimension = indexj
                println("$(matrixdimension)")
                break
            end
            matrixQ[:, indexj+1] = vectorVJ / matrixH[indexj+1, indexj]
            for indexi = 1:indexj-1
                auxiliarvalue = matrixH[indexi, indexj]
                matrixH[indexi, indexj] = vectorcoseno[indexi] * auxiliarvalue + vectorseno[indexi] * matrixH[indexi+1, indexj]
                matrixH[indexi+1, indexj] = -vectorseno[indexi] * auxiliarvalue + vectorcoseno[indexi] * matrixH[indexi+1, indexj]
            end
            if (abs(matrixH[indexj, indexj]) > abs(matrixH[indexj+1, indexj]))
                auxiliarvalue1 = matrixH[indexj+1, indexj] / matrixH[indexj, indexj]
                auxiliarvalue2 = sign(matrixH[indexj, indexj]) * sqrt(1 + auxiliarvalue1^2)
                vectorcoseno[indexj] = 1 / auxiliarvalue2
                vectorseno[indexj] = auxiliarvalue1 * vectorcoseno[indexj]
            else
                auxiliarvalue1 =  matrixH[indexj, indexj] / matrixH[indexj+1, indexj]
                auxiliarvalue2 = sign(matrixH[indexj+1, indexj]) * sqrt(1 + auxiliarvalue1^2)
                vectorseno[indexj] = 1 / auxiliarvalue2
                vectorcoseno[indexj] = auxiliarvalue1 * vectorseno[indexj]
            end
            valueHjj = matrixH[indexj, indexj]
            matrixH[indexj, indexj] = vectorcoseno[indexj] * valueHjj + vectorseno[indexj] * matrixH[indexj+1, indexj]
            matrixH[indexj+1, indexj] = -vectorseno[indexj] * valueHjj + vectorcoseno[indexj] * matrixH[indexj+1, indexj]
            vectorbethaE[indexj+1] = -vectorseno[indexj]*vectorbethaE[indexj]
            vectorbethaE[indexj] = vectorcoseno[indexj] * vectorbethaE[indexj]
        end
        for indexi = 1:matrixdimension-1
            auxiliarvalue = matrixH[indexi, matrixdimension]
            matrixH[indexi, matrixdimension] = vectorcoseno[indexi] * auxiliarvalue + vectorseno[indexi] * matrixH[indexi+1, matrixdimension]
            matrixH[indexi+1, matrixdimension] = -vectorseno[indexi] * auxiliarvalue + vectorcoseno[indexi] * matrixH[indexi+1, matrixdimension]
        end
        vectory = backward_substitution(matrixdimension, matrixH, vectorbethaE )
        vectorx_aproximateSolution = vectorx_current + matrixQ[:,1:matrixdimension]*vectory
        if indexk == maximumrestart
            methoderror = norm(vectorx_aproximateSolution - vectorx_exactsolution) / (norm(vectorx_aproximateSolution))
            break
        end
        vectorx_current = vectorx_aproximateSolution
        indexk += 1
    end
    # Output variables
    if informationprint == true
        println("For $(maximumrestart) restart with a restart parameter of $(restartparameter)")
        println("the method error is: $(methoderror)")
        return methoderror
    end
    return nothing
end
#######################################################################################################################
# Auxiliars Functions
#######################################################################################################################
function backward_substitution( systemdimension::Integer, matrixU::AbstractMatrix, vectorb::AbstractVector )
    vectorx = Vector{Float64}(undef, systemdimension)
    vectorx[systemdimension] = vectorb[systemdimension] / matrixU[systemdimension, systemdimension]
    for indexi = systemdimension-1:-1:1
        accumulator = dot(matrixU[indexi, indexi+1:systemdimension], vectorx[indexi+1:systemdimension])
        vectorx[indexi] = (vectorb[indexi] - accumulator) / matrixU[indexi, indexi]
    end
    return vectorx
end