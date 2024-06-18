using BenchmarkTools
using LinearAlgebra: Diagonal, tril, triu
using DataFrames, CSV
include("src/IterativeSolversMethods.jl")
BenchmarkTools.DEFAULT_PARAMETERS.samples = 10
# ==============================================================================
function run_jacobi_method(systemDimension::Integer)
    println("===Jacobi Method===")

    matrixA = Matrix{Float64}(undef, systemDimension, systemDimension)
    system_matrix_sparse(systemDimension, matrixA)

    vectorB = Vector{Float64}(undef, systemDimension)
    vector_independent_term(systemDimension, vectorB)

    vectorX_exactSolution = ones(systemDimension)
    vectorX_initial = zeros(systemDimension)

    # Split Matrix A = D + L + U
    matrixD = convert(Matrix, Diagonal(matrixA))
    matrixL = tril(matrixA, -1)
    matrixU = triu(matrixA, 1)

    # Set for Jacobi Method
    inverseMatrixD = inv(matrixD)
    matrixT = matrixL + matrixU

    # DataFrame for save method information
    numericalResults = DataFrame()
    numericalResults.Iterations = 25:25:200
    numericalResults.methodError = zeros(8)
    numericalResults.RunTime = zeros(8)

    for i = 1:8
        maximumIteration = i * 25
        numericalResults[i, 2] = jacobi_method!(inverseMatrixD, matrixT, vectorB, vectorX_exactSolution, vectorX_initial, maximumIteration, true)
        methodTime = @belapsed jacobi_method!($inverseMatrixD, $matrixT, $vectorB, $vectorX_exactSolution, $vectorX_initial, $maximumIteration, false)
        numericalResults[i, 3] = methodTime
    end
    CSV.write("Numerical_Experiments/jacobiNumericalResults_$(string(systemDimension)).csv", numericalResults)
    return nothing
end
# ==============================================================================
function run_gauss_seidel_method(systemDimension::Integer)
    println("===Gauss-Seidel Method===")

    matrixA = Matrix{Float64}(undef, systemDimension, systemDimension)
    system_matrix_sparse(systemDimension, matrixA)

    vectorB = Vector{Float64}(undef, systemDimension)
    vector_independent_term(systemDimension, vectorB)

    vectorX_exactSolution = ones(systemDimension)
    vectorX_initial = zeros(systemDimension)

    # Split Matrix A = D + L + U
    matrixD = convert(Matrix, Diagonal(matrixA))
    matrixL = tril(matrixA, -1)
    matrixU = triu(matrixA, 1)

    # Set for Gauss-Seidel Method
    inverseMatrixT = inv(matrixD + matrixL)

    # DataFrame for save method information
    numericalResults = DataFrame()
    numericalResults.Iterations = 25:25:200
    numericalResults.methodError = zeros(8)
    numericalResults.RunTime = zeros(8)

    for i = 1:8
        maximumIteration = i * 25
        numericalResults[i, 2] = gauss_seidel_method!(inverseMatrixT, matrixU, vectorB, vectorX_exactSolution, vectorX_initial, maximumIteration, true)
        methodTime = @belapsed gauss_seidel_method!($inverseMatrixT, $matrixU, $vectorB, $vectorX_exactSolution, $vectorX_initial, $maximumIteration, false)
        numericalResults[i, 3] = methodTime
    end
    CSV.write("Numerical_Experiments/gssNumericalResults_$(string(systemDimension)).csv", numericalResults)
    return nothing
end
# ==============================================================================
function run_sor_method(systemDimension::Integer, relaxationParameter::AbstractFloat)
    println("===Successive Over-Relaxation Method===")

    matrixA = Matrix{Float64}(undef, systemDimension, systemDimension)
    system_matrix_sparse(systemDimension, matrixA)
    
    vectorB = Vector{Float64}(undef, systemDimension)
    vector_independent_term(systemDimension, vectorB)

    vectorX_exactSolution = ones(systemDimension)
    vectorX_initial = zeros(systemDimension)

    # Split Matrix A = D + L + U
    matrixD = convert(Matrix, Diagonal(matrixA))
    matrixL = tril(matrixA, -1)
    matrixU = triu(matrixA, 1)

    # Set for Gauss-Seidel Method
    inverseMatrixT = inv(matrixD + relaxationParameter * matrixL)
    matrixOmegaD = (1 - relaxationParameter) * matrixD
    matrixOmegaU = relaxationParameter * matrixU
    vectorF = relaxationParameter * inverseMatrixT * vectorB

    # DataFrame for save method information
    numericalResults = DataFrame()
    numericalResults.Iterations = 25:25:200
    numericalResults.methodError = zeros(8)
    numericalResults.RunTime = zeros(8)
    #sor_method(matrixD, matrixL, matrixU, vectorB, vectorX_exactSolution, vectorX_initial, maximumIteration, relaxationParameter, tolerance)
    for i = 1:8
        maximumIteration = i * 25
        numericalResults[i, 2] = sor_method!(inverseMatrixT, matrixOmegaD, matrixOmegaU, vectorF, vectorX_exactSolution, vectorX_initial, maximumIteration, relaxationParameter, true)
        methodTime = @belapsed sor_method!($inverseMatrixT, $matrixOmegaD, $matrixOmegaU, $vectorF, $vectorX_exactSolution, $vectorX_initial, $maximumIteration, $relaxationParameter, false)
        numericalResults[i, 3] = methodTime
    end
    CSV.write("Numerical_Experiments/sorNumericalResults_$(string(systemDimension))_$(string(relaxationParameter)).csv", numericalResults)
    return nothing
end
# ==============================================================================
function run_conjugate_gradiente_method(systemDimension::Integer)
    println("===Conjugate Gradient Method===")

    matrixA = Matrix{Float64}(undef, systemDimension, systemDimension)
    system_matrix_sparse(systemDimension, matrixA)

    vectorB = Vector{Float64}(undef, systemDimension)
    vector_independent_term(systemDimension, vectorB)

    vectorX_exactSolution = ones(systemDimension)
    vectorX_initial = zeros(systemDimension)

    # DataFrame for save method information
    numericalResults = DataFrame()
    numericalResults.Iterations = 5:5:40
    numericalResults.methodError = zeros(8)
    numericalResults.RunTime = zeros(8)

    for i = 1:8
        maximumIteration = i * 5
        numericalResults[i, 2] = conjugate_gradient_method!(matrixA, vectorB, vectorX_exactSolution, vectorX_initial, maximumIteration, true)
        methodTime = @belapsed conjugate_gradient_method!($matrixA, $vectorB, $vectorX_exactSolution, $vectorX_initial, $maximumIteration, false)
        numericalResults[i, 3] = methodTime
    end
    CSV.write("Numerical_Experiments/cgNumericalResults_$(string(systemDimension)).csv", numericalResults)
    return nothing
end
# ==============================================================================
function run_biconjugate_gradiente_stabilized_method(systemDimension::Integer)
    println("===Biconjugate Gradient Stabilized Method===")
    
    matrixA = Matrix{Float64}(undef, systemDimension, systemDimension)
    system_matrix_sparse(systemDimension, matrixA)

    vectorB = Vector{Float64}(undef, systemDimension)
    vector_independent_term(systemDimension, vectorB)
    
    vectorX_exactSolution = ones(systemDimension)
    vectorX_initial = zeros(systemDimension)

    # DataFrame for save method information
    numericalResults = DataFrame()
    numericalResults.Iterations = 5:5:40
    numericalResults.methodError = zeros(8)
    numericalResults.RunTime = zeros(8)

    for i = 1:8
        maximumIteration = i * 5
        numericalResults[i, 2] = biconjugate_gradient_stabilized_method!(matrixA, vectorB, vectorX_exactSolution, vectorX_initial, maximumIteration, true)
        methodTime = @belapsed biconjugate_gradient_stabilized_method!($matrixA, $vectorB, $vectorX_exactSolution, $vectorX_initial, $maximumIteration, false)
        numericalResults[i, 3] = methodTime
    end
    CSV.write("Numerical_Experiments/bicgstabNumericalResults_$(string(systemDimension)).csv", numericalResults)
    return nothing
end
# ==============================================================================
function run_restarted_generalized_minimal_residual_method(systemDimension::Integer, restartParameter::Integer)
    println("===Generalized Minimal Residual Method===")
    matrixA = Matrix{Float64}(undef, systemDimension, systemDimension)
    system_matrix_sparse(systemDimension, matrixA)

    vectorB = Vector{Float64}(undef, systemDimension)
    vector_independent_term(systemDimension, vectorB)
    
    vectorX_exactSolution = ones(systemDimension)
    vectorX_initial = zeros(systemDimension)
    tolerance = 1.0e-10
    
    # DataFrame for save method information
    numericalResults = DataFrame()
    numericalResults.Iterations = 5:5:40
    numericalResults.methodError = zeros(8)
    numericalResults.RunTime = zeros(8)
    
    for i = 1:8
        maximumIteration = i * 5
        numericalResults[i, 2] = restarted_generalized_minimal_residual_method!(systemDimension, matrixA, vectorB, vectorX_initial, vectorX_exactSolution, restartParameter, maximumIteration, tolerance, true)
        methodTime = @belapsed restarted_generalized_minimal_residual_method!($systemDimension, $matrixA, $vectorB, $vectorX_initial, $vectorX_exactSolution, $restartParameter, $maximumIteration, $tolerance, false)
        numericalResults[i, 3] = methodTime
    end
    CSV.write("Numerical_Experiments/restartedGmresNumericalResults_$(string(systemDimension))_$(string(restartParameter)).csv", numericalResults)
    return nothing
end