# =================================================================
# Methods for solve Systems of Equations Linear
# =================================================================
using LinearAlgebra: dot, norm
# =================================================================
# Jacobi Method
# =================================================================
function jacobi_method(
    matrixD::AbstractMatrix,
    matrixL::AbstractMatrix,
    matrixU::AbstractMatrix,
    vectorB::AbstractVector,
    vectorX_exactSolution::AbstractVector,
    vectorX_initial::AbstractVector,
    maximumIteration::Integer,
    tolerance::AbstractFloat)

    iterationNumber = 0
    methodError = 0.0
    vectorX_current = vectorX_initial
    inverseMatrixD = inv(matrixD)
    matrixT = matrixL + matrixU

    for k = 1:maximumIteration
        vectorX_next = inverseMatrixD * (vectorB - matrixT * vectorX_current)
        methodError = norm(vectorX_next - vectorX_exactSolution) / norm(vectorX_next)
        # Stop Condition
        if methodError <= tolerance
            iterationNumber = k
            break
        end
        # Update 
        vectorX_current = vectorX_next
    end

    # Method information
    println("The number of iterations is: $(iterationNumber)")
    println("The method error is: $(methodError)")
    return nothing
end
# =================================================================
# Jacobi Method (for numerical experiment)
# =================================================================
function jacobi_method!(
    inverseMatrixD::AbstractMatrix,
    matrixT::AbstractMatrix,
    vectorB::AbstractVector,
    vectorX_exactSolution::AbstractVector,
    vectorX_initial::AbstractVector,
    maximumIteration::Integer,
    informationPrint::Bool)
    """
        inverseMatrixD:     inverse matrix of D.

        matrixT:            matrix T = L + U.
    """

    methodError = 0.0
    vectorX_current = vectorX_initial

    for k = 1:maximumIteration
        vectorX_next = inverseMatrixD * (vectorB - matrixT * vectorX_current)
        # For mesure method error
        if k == maximumIteration
            methodError = norm(vectorX_next - vectorX_exactSolution) / norm(vectorX_next)
            break
        end
        # Update 
        vectorX_current = vectorX_next
    end

    # Method information
    if informationPrint == true
        println("For $(maximumIteration) iterations")
        println("the method error is: $(methodError)")
        return methodError
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
    vectorB::AbstractVector,
    vectorX_exactSolution::AbstractVector,
    vectorX_initial::AbstractVector,
    maximumIteration::Integer,
    tolerance::AbstractFloat)

    iterationNumber = 0
    methodError = 0.0
    vectorX_current = vectorX_initial
    matrixT = matrixD + matrixL
    inverseMatrixT = inv(matrixT)

    for k = 1:maximumIteration
        vectorX_next = inverseMatrixT * (vectorB - matrixU * vectorX_current)
        methodError = norm(vectorX_next - vectorX_exactSolution) / norm(vectorX_next)
        # Stop Condition
        if methodError <= tolerance
            iterationNumber = k
            break
        end
        # Update
        vectorX_current = vectorX_next
    end

    # Method information
    println("The number of iterations is: $(iterationNumber)")
    println("The method error is: $(methodError)")
    return nothing
end
# =================================================================
# Gauss-Seidel Method (for numerical experiments)
# =================================================================
function gauss_seidel_method!(
    inverseMatrixT::AbstractMatrix,
    matrixU::AbstractMatrix,
    vectorB::AbstractVector,
    vectorX_exactSolution::AbstractVector,
    vectorX_initial::AbstractVector,
    maximumIteration::Integer,
    informationPrint::Bool)
    """
        inverseMatrixT: inverse matrix T, where matrix T = D + L
    """

    methodError = 0.0
    vectorX_current = vectorX_initial

    for k = 1:maximumIteration
        vectorX_next = inverseMatrixT * (vectorB - matrixU * vectorX_current)
        # Stop Condition
        if k == maximumIteration
            methodError = norm(vectorX_next - vectorX_exactSolution) / norm(vectorX_next)
            break
        end
        # Update
        vectorX_current = vectorX_next
    end

    # Method information
    if informationPrint == true
        println("For $(maximumIteration) iterations")
        println("the method error is: $(methodError)")
        return methodError
    end
    return nothing
end
# =================================================================
# Successive Over-Relaxation Method
# =================================================================
function sor_method(
    matrixD::AbstractMatrix,
    matrixL::AbstractMatrix,
    matrixU::AbstractMatrix,
    vectorB::AbstractVector,
    vectorX_exactSolution::AbstractVector,
    vectorX_initial::AbstractVector,
    maximumIteration::Integer,
    relaxationParameter::AbstractFloat,
    tolerance::AbstractFloat)

    omega = relaxationParameter
    iterationNumber = 0
    methodError = 0.0
    vectorX_current = vectorX_initial
    matrixT = matrixD + omega * matrixL
    inverseMatrixT = inv(matrixT)
    matrixOmegaD = (1 - omega) * matrixD
    matrixOmegaU = omega * matrixU
    vectorF = omega * matrixInvT * vectorB

    for k = 1:maximumIteration
        vectorX_next = inverseMatrixT * (matrixOmegaD * vectorX_current - matrixOmegaU * vectorX_current) + vectorF
        methodError = norm(vectorX_next - vectorX_exactSolution) / norm(vectorX_next)
        # Stop Condition
        if methodError <= tolerance
            iterationNumber = k
            break
        end
        # Update
        vectorX_current = vectorX_next
    end

    # Method information
    println("The number of iterations is: $(iterationNumber)")
    println("The method error is: $(methodError)")
    return nothing
end
# =================================================================
# Successive Over-Relaxation Method (for numerical experiments)
# =================================================================
function sor_method!(
    inverseMatrixT::AbstractMatrix,
    matrixOmegaD::AbstractMatrix,
    matrixOmegaU::AbstractMatrix,
    vectorF::AbstractVector,
    vectorX_exactSolution::AbstractVector,
    vectorX_initial::AbstractVector,
    maximumIteration::Integer,
    relaxationParameter::AbstractFloat,
    informationPrint::Bool)
    """
        omega = relaxationParameter

        inverseMatrixT: inverse matrix T, where matrix T = D + omega * L.

        matrixOmegaD:   matrix omegaD = (1 - omega) * D.

        matrixOmegaU:   matrix omegaU = omega * U.

        vectorF:        omega * (D + omega * L)^{-1} * b.
    """

    methodError = 0.0
    vectorX_current = vectorX_initial

    for k = 1:maximumIteration
        vectorX_next = inverseMatrixT * (matrixOmegaD * vectorX_current - matrixOmegaU * vectorX_current) + vectorF
        # Stop Condition
        if k == maximumIteration
            methodError = norm(vectorX_next - vectorX_exactSolution) / norm(vectorX_next)
            break
        end
        # Update
        vectorX_current = vectorX_next
    end

    # Method information
    if informationPrint == true
        println("For a relaxation parameter of $(relaxationParameter)")
        println("get $(maximumIteration) iterations")
        println("and the method error is: $(methodError).")
        return methodError
    end
    return nothing
end
# =================================================================
# Conjugate Gradient Method
# =================================================================
function conjugate_gradient_method(
    systemDimension::Integer,
    matrixA::AbstractMatrix,
    vectorB::AbstractVector,
    vectorX_exactSolution::AbstractVector,
    vectorX_initial::AbstractVector,
    tolerance::AbstractFloat)

    maximumIteration = systemDimension
    methodError = 0.0
    iterationNumber = 0
    vectorX_current = vectorX_initial
    vectorR_current = vectorB - matrixA * vectorX_current
    vectorD_current = vectorR_current

    for k = 1:maximumIteration
        vectorAD = matrixA * vectorD_current
        alpha = dot(vectorR_current, vectorR_current) / dot(vectorAD, vectorD_current)
        vectorX_next = vectorX_current + alpha * vectorD_current
        # Stop Condition
        methodError = norm(vectorX_next - vectorX_exactSolution) / norm(vectorX_next)
        if methodError <= tolerance
            iterationNumber = k
            break
        end
        vectorR_next = vectorR_current - alpha * vectorAD
        beta = dot(vectorR_next, vectorR_next) / dot(vectorR_current, vectorR_current)
        vectorD_next = vectorR_next + beta * vectorD_current
        # Update
        vectorX_current = vectorX_next
        vectorR_current = vectorR_next
        vectorD_current = vectorD_next
    end

    # Method information
    println("The iteration number is: $(iterationNumber).")
    println("The method error is: $(methodError).")
    return nothing
end
# =================================================================
# Conjugate Gradient Method (for numerical experiments)
# =================================================================
function conjugate_gradient_method!(
    matrixA::AbstractMatrix,
    vectorB::AbstractVector,
    vectorX_exactSolution::AbstractVector,
    vectorX_initial::AbstractVector,
    maximumIteration::Integer,
    informationPrint::Bool)

    methodError = 0.0
    vectorX_current = vectorX_initial
    vectorR_current = vectorB - matrixA * vectorX_current
    vectorD_current = vectorR_current

    for k = 1:maximumIteration
        vectorAD = matrixA * vectorD_current
        alpha = dot(vectorR_current, vectorR_current) / dot(vectorAD, vectorD_current)
        vectorX_next = vectorX_current + alpha * vectorD_current
        # Stop Condition
        if k == maximumIteration
            methodError = norm(vectorX_next - vectorX_exactSolution) / norm(vectorX_next)
            break
        end
        vectorR_next = vectorR_current - alpha * vectorAD
        beta = dot(vectorR_next, vectorR_next) / dot(vectorR_current, vectorR_current)
        vectorD_next = vectorR_next + beta * vectorD_current
        # Update
        vectorX_current = vectorX_next
        vectorR_current = vectorR_next
        vectorD_current = vectorD_next
    end

    # Output variables
    if informationPrint == true
        println("For $(maximumIteration) iterations")
        println("the method error is: $(methodError)")
        return methodError
    end
    return nothing
end
# =================================================================
# Biconjugate Gradient Stabilized Method
# =================================================================
function biconjugate_gradient_stabilized_method(
    systemDimension::Integer,
    matrixA::AbstractMatrix,
    vectorB::AbstractVector,
    vectorX_exactSolution::AbstractVector,
    vectorX_initial::AbstractVector,
    tolerance::AbstractFloat)

    maximumIteration = systemDimension
    methodError = 0.0
    iterationNumber = 0
    vectorX_current = vectorX_initial
    vectorR_current = vectorB - matrixA * vectorX_current
    vectorD_current = vectorR_current
    vectorR_ast = vectorR_current

    for k = 1:maximumIteration
        vectorAD = matrixA * vectorD_current
        alpha = dot(vectorR_current, vectorR_ast) / dot(vectorAD, vectorR_ast)
        vectorS = vectorR_current - alpha * vectorAD
        vectorAS = matrixA * vectorS
        omega = dot(vectorAS, vectorS) / dot(vectorAS, vectorAS)
        vectorX_next = vectorX_current + alpha * vectorD_current + omega * vectorS
        # Stop Condition
        methodError = norm(vectorX_next - vectorX_exactSolution) / norm(vectorX_next)
        if methodError <= tolerance
            iterationNumber = k
            break
        end
        vectorR_next = vectorS - omega * vectorAS
        beta = (dot(vectorR_next, vectorR_ast) / dot(vectorR_current, vectorR_ast)) * (alpha / omega)
        vectorD_next = vectorR_next + beta * (vectorD_current - omega * vectorAD)
        # Update
        vectorX_current = vectorX_next
        vectorR_current = vectorR_next
        vectorD_current = vectorD_next
    end

    # Output variables
    println("The iteration number is: $(iterationNumber).")
    println("The method error is: $(methodError).")
    return nothing
end
# =================================================================
# Biconjugate Gradient Stabilized Method (for numerical experiments)
# =================================================================
function biconjugate_gradient_stabilized_method!(
    matrixA::AbstractMatrix,
    vectorB::AbstractVector,
    vectorX_exactSolution::AbstractVector,
    vectorX_initial::AbstractVector,
    maximumIteration::Integer,
    informationPrint::Bool)

    methodError = 0.0
    vectorX_current = vectorX_initial
    vectorR_current = vectorB - matrixA * vectorX_current
    vectorD_current = vectorR_current
    vectorR_ast = vectorR_current

    for k = 1:maximumIteration
        vectorAD = matrixA * vectorD_current
        alpha = dot(vectorR_current, vectorR_ast) / dot(vectorAD, vectorR_ast)
        vectorS = vectorR_current - alpha * vectorAD
        vectorAS = matrixA * vectorS
        omega = dot(vectorAS, vectorS) / dot(vectorAS, vectorAS)
        vectorX_next = vectorX_current + alpha * vectorD_current + omega * vectorS
        # Stop Condition
        if k == maximumIteration
            methodError = norm(vectorX_next - vectorX_exactSolution) / norm(vectorX_next)
            break
        end
        vectorR_next = vectorS - omega * vectorAS
        beta = (dot(vectorR_next, vectorR_ast) / dot(vectorR_current, vectorR_ast)) * (alpha / omega)
        vectorD_next = vectorR_next + beta * (vectorD_current - omega * vectorAD)
        # Update
        vectorX_current = vectorX_next
        vectorR_current = vectorR_next
        vectorD_current = vectorD_next
    end

    # Output variables
    if informationPrint == true
        println("For $(maximumIteration) iterations")
        println("the method error is: $(methodError)")
        return methodError
    end
    return nothing
end
# =================================================================
# Restarted Generalized Minimal Residual Method
# =================================================================
function restarted_generalized_minimal_residual_method(
    systemDimension::Integer,
    matrixA::AbstractMatrix,
    vectorB::AbstractVector,
    vectorX_initial::AbstractVector,
    vectorX_exactSolution::AbstractVector,
    restartParameter::Integer,
    tolerance::AbstractFloat)

    matrixH = Matrix{Float64}(undef, restartParameter + 1, restartParameter)
    matrixQ = Matrix{Float64}(undef, systemDimension, restartParameter + 1)
    vectorX_current = vectorX_initial
    methodError = 0.0
    flag = true
    matrixDimension = restartParameter

    k = 0
    while flag == true
        # println("Reinicio:",k)
        vectorR_start = vectorB - matrixA * vectorX_current
        beta = norm(vectorR_start)
        matrixQ[:, 1] = vectorR_start / beta
        vectorE = zeros(restartParameter + 1)
        vectorE[1] = 1
        vectorBetaE = beta * vectorE
        vectorCoseno = Vector{Float64}(undef, restartParameter)
        vectorSeno = Vector{Float64}(undef, restartParameter)
        kappa = 0.38
        for j = 1:restartParameter
            vectorVJ = matrixA * matrixQ[:, j]
            normVJ = norm(vectorVJ)
            for i = 1:j
                matrixH[i, j] = dot(vectorVJ, matrixQ[:, i])
                vectorVJ = vectorVJ - matrixH[i, j] * matrixQ[:, i]
            end
            if norm(vectorVJ) / normVJ <= kappa
                for i = 1:j
                    vectorP = dot(matrixQ[:, i], vectorVJ)
                    vectorVJ = vectorVJ - vectorP * matrixQ[:, i]
                    matrixH[i, j] = matrixH[i, j] + vectorP
                end
            end
            matrixH[j+1, j] = norm(vectorVJ)
            if abs(matrixH[j+1, j]) < tolerance
                println("fin")
                matrixDimension = j
                println("$(matrixDimension)")
                break
            end
            matrixQ[:, j+1] = vectorVJ / matrixH[j+1, j]
            for i = 1:j-1
                auxiliarValue = matrixH[i, j]
                matrixH[i, j] = vectorCoseno[i] * auxiliarValue + vectorSeno[i] * matrixH[i+1, j]
                matrixH[i+1, j] = -vectorSeno[i] * auxiliarValue + vectorCoseno[i] * matrixH[i+1, j]
            end
            if (abs(matrixH[j, j]) > abs(matrixH[j+1, j]))
                auxiliarValue1 = matrixH[j+1, j] / matrixH[j, j]
                auxiliarValue2 = sign(matrixH[j, j]) * sqrt(1 + auxiliarValue1^2)
                vectorCoseno[j] = 1 / auxiliarValue2
                vectorSeno[j] = auxiliarValue1 * vectorCoseno[j]
            else
                auxiliarValue1 = matrixH[j, j] / matrixH[j+1, j]
                auxiliarValue2 = sign(matrixH[j+1, j]) * sqrt(1 + auxiliarValue1^2)
                vectorSeno[j] = 1 / auxiliarValue2
                vectorCoseno[j] = auxiliarValue1 * vectorSeno[j]
            end
            valueHjj = matrixH[j, j]
            matrixH[j, j] = vectorCoseno[j] * valueHjj + vectorSeno[j] * matrixH[j+1, j]
            matrixH[j+1, j] = -vectorSeno[j] * valueHjj + vectorCoseno[j] * matrixH[j+1, j]
            vectorBetaE[j+1] = -vectorSeno[j] * vectorBetaE[j]
            vectorBetaE[j] = vectorCoseno[j] * vectorBetaE[j]
        end
        for i = 1:matrixDimension-1
            auxiliarValue = matrixH[i, matrixDimension]
            matrixH[i, matrixDimension] = vectorCoseno[i] * auxiliarValue + vectorSeno[i] * matrixH[i+1, matrixDimension]
            matrixH[i+1, matrixDimension] = -vectorSeno[i] * auxiliarValue + vectorCoseno[i] * matrixH[i+1, matrixDimension]
        end
        vectorY = backward_substitution(matrixDimension, matrixH, vectorBetaE)
        vectorX_aproximateSolution = vectorX_current + matrixQ[:, 1:matrixDimension] * vectorY
        methodError = norm(vectorX_aproximateSolution - vectorX_exactSolution) / (norm(vectorX_aproximateSolution))
        if methodError < tolerance
            break
        end
        vectorX_current = vectorX_aproximateSolution
        k += 1
    end

    println("El número de reinicios es de $(k).")
    println("El error del método es: $(methodError).")
    return nothing
end
# =================================================================
# Restarted Generalized Minimal Residual Method (for numerical experiments)
# =================================================================
function restarted_generalized_minimal_residual_method!(
    systemDimension::Integer,
    matrixA::AbstractMatrix,
    vectorB::AbstractVector,
    vectorX_initial::AbstractVector,
    vectorX_exactSolution::AbstractVector,
    restartParameter::Integer,
    maximumRestart::Integer,
    tolerance::AbstractFloat,
    informationPrint::Bool)

    matrixH = Matrix{Float64}(undef, restartParameter + 1, restartParameter)
    matrixQ = Matrix{Float64}(undef, systemDimension, restartParameter + 1)
    vectorX_current = vectorX_initial
    methodError = 0.0
    flag = true
    matrixDimension = restartParameter

    k = 0
    while flag == true
        # println("Reinicio:",k)
        vectorR_start = vectorB - matrixA * vectorX_current
        Beta = norm(vectorR_start)
        matrixQ[:, 1] = vectorR_start / Beta
        vectorE = zeros(restartParameter + 1)
        vectorE[1] = 1
        vectorBetaE = Beta * vectorE
        vectorCoseno = Vector{Float64}(undef, restartParameter)
        vectorSeno = Vector{Float64}(undef, restartParameter)
        kappa = 0.38
        for j = 1:restartParameter
            vectorVJ = matrixA * matrixQ[:, j]
            normVJ = norm(vectorVJ)
            for i = 1:j
                matrixH[i, j] = dot(vectorVJ, matrixQ[:, i])
                vectorVJ = vectorVJ - matrixH[i, j] * matrixQ[:, i]
            end
            if norm(vectorVJ) / normVJ <= kappa
                for i = 1:j
                    vectorP = dot(matrixQ[:, i], vectorVJ)
                    vectorVJ = vectorVJ - vectorP * matrixQ[:, i]
                    matrixH[i, j] = matrixH[i, j] + vectorP
                end
            end
            matrixH[j+1, j] = norm(vectorVJ)
            if abs(matrixH[j+1, j]) < tolerance
                # println("fin")
                matrixDimension = j
                # println("$(matrixDimension)")
                break
            end
            matrixQ[:, j+1] = vectorVJ / matrixH[j+1, j]
            for i = 1:j-1
                auxiliarValue = matrixH[i, j]
                matrixH[i, j] = vectorCoseno[i] * auxiliarValue + vectorSeno[i] * matrixH[i+1, j]
                matrixH[i+1, j] = -vectorSeno[i] * auxiliarValue + vectorCoseno[i] * matrixH[i+1, j]
            end
            if (abs(matrixH[j, j]) > abs(matrixH[j+1, j]))
                auxiliarValue1 = matrixH[j+1, j] / matrixH[j, j]
                auxiliarValue2 = sign(matrixH[j, j]) * sqrt(1 + auxiliarValue1^2)
                vectorCoseno[j] = 1 / auxiliarValue2
                vectorSeno[j] = auxiliarValue1 * vectorCoseno[j]
            else
                auxiliarValue1 = matrixH[j, j] / matrixH[j+1, j]
                auxiliarValue2 = sign(matrixH[j+1, j]) * sqrt(1 + auxiliarValue1^2)
                vectorSeno[j] = 1 / auxiliarValue2
                vectorCoseno[j] = auxiliarValue1 * vectorSeno[j]
            end
            valueHjj = matrixH[j, j]
            matrixH[j, j] = vectorCoseno[j] * valueHjj + vectorSeno[j] * matrixH[j+1, j]
            matrixH[j+1, j] = -vectorSeno[j] * valueHjj + vectorCoseno[j] * matrixH[j+1, j]
            vectorBetaE[j+1] = -vectorSeno[j] * vectorBetaE[j]
            vectorBetaE[j] = vectorCoseno[j] * vectorBetaE[j]
        end
        for i = 1:matrixDimension-1
            auxiliarValue = matrixH[i, matrixDimension]
            matrixH[i, matrixDimension] = vectorCoseno[i] * auxiliarValue + vectorSeno[i] * matrixH[i+1, matrixDimension]
            matrixH[i+1, matrixDimension] = -vectorSeno[i] * auxiliarValue + vectorCoseno[i] * matrixH[i+1, matrixDimension]
        end
        vectorY = backward_substitution(matrixDimension, matrixH, vectorBetaE)
        vectorX_aproximateSolution = vectorX_current + matrixQ[:, 1:matrixDimension] * vectorY
        if k == maximumRestart
            methodError = norm(vectorX_aproximateSolution - vectorX_exactSolution) / (norm(vectorX_aproximateSolution))
            break
        end
        vectorX_current = vectorX_aproximateSolution
        k += 1
    end

    # Output variables
    if informationPrint == true
        println("For $(maximumRestart) restart with a restart parameter of $(restartParameter)")
        println("the method error is: $(methodError)")
        return methodError
    end
    return nothing
end
#######################################################################################################################
# Auxiliary Functions
#######################################################################################################################
function backward_substitution(systemDimension::Integer, matrixU::AbstractMatrix, vectorB::AbstractVector)
    vectorX = Vector{Float64}(undef, systemDimension)
    vectorX[systemDimension] = vectorB[systemDimension] / matrixU[systemDimension, systemDimension]
    for i = systemDimension-1:-1:1
        accumulator = dot(matrixU[i, i+1:systemDimension], vectorX[i+1:systemDimension])
        vectorX[i] = (vectorB[i] - accumulator) / matrixU[i, i]
    end
    return vectorX
end