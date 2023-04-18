using BenchmarkTools
using LinearAlgebra: Diagonal, tril, triu
using SparseArrays: sparse, SparseMatrixCSC
using DataFrames, CSV
include("src/IterativesSolversMethods.jl")
BenchmarkTools.DEFAULT_PARAMETERS.samples = 10
function run_jacobi_method(systemdimension::Integer)
    matrixA = Matrix{Float64}(undef, systemdimension, systemdimension)
    system_matrix_sparse(systemdimension, matrixA)
    vectorb = Vector{Float64}(undef, systemdimension)
    vector_independent_term(systemdimension, vectorb)
    vectorx_exactsolution = ones(systemdimension)
    vectorx_initial = zeros(systemdimension)
    #tolerance = 1.0e-10
    # Matrix Split
    matrixD =  convert(Matrix, Diagonal(matrixA))
    matrixL = tril(matrixA,-1)
    matrixU = triu(matrixA,1)
    println("===Jacobi Method===")
    # DataFrame for save method information
    sizeSEL = string(systemdimension)
    numericalresults = DataFrame()
    numericalresults.Iterations = 25:25:200
    numericalresults.MethodError = zeros(8)
    numericalresults.RunTime = zeros(8)
    #jacobi_method(matrixD, matrixL, matrixU, vectorb, vectorx_exactsolution, vectorx_initial, maximunIteration, tolerance)
    for indexi = 1:8
        maximumiteration = indexi*25
        numericalresults[indexi, 2] = jacobi_method!(matrixD, matrixL, matrixU, vectorb, vectorx_exactsolution, vectorx_initial, maximumiteration, true)
        methodtime = @belapsed jacobi_method!($matrixD, $matrixL, $matrixU, $vectorb, $vectorx_exactsolution, $vectorx_initial, $maximumiteration, false)
        numericalresults[indexi, 3] = methodtime
    end
    CSV.write("Numerical_Experiments/jacobiNumericalResults_$(sizeSEL).csv", numericalresults)
    return nothing
end
function run_gauss_seidel_method(systemdimension::Integer)
    matrixA = Matrix{Float64}(undef, systemdimension, systemdimension)
    system_matrix_sparse(systemdimension, matrixA)
    vectorb = Vector{Float64}(undef, systemdimension)
    vector_independent_term(systemdimension, vectorb)
    vectorx_exactsolution = ones(systemdimension)
    vectorx_initial = zeros(systemdimension)
    #tolerance = 1.0e-10
    # Matrix Split
    matrixD =  convert(Matrix, Diagonal(matrixA))
    matrixL = tril(matrixA,-1)
    matrixU = triu(matrixA,1)
    println("===Gauss-Seidel Method===")
    # DataFrame for save method information
    sizeSEL = string(systemdimension)
    numericalresults = DataFrame()
    numericalresults.Iterations = 25:25:200
    numericalresults.MethodError = zeros(8)
    numericalresults.RunTime = zeros(8)
    #gauss_seidel_method(matrixD, matrixL, matrixU, vectorb, vectorx_exactsolution, vectorx_initial, maximunIteration, tolerance)
    for indexi = 1:8
        maximumiteration = indexi*25
        numericalresults[indexi, 2] = gauss_seidel_method!(matrixD, matrixL, matrixU, vectorb, vectorx_exactsolution, vectorx_initial, maximumiteration, true)
        methodtime = @belapsed gauss_seidel_method!($matrixD, $matrixL, $matrixU, $vectorb, $vectorx_exactsolution, $vectorx_initial, $maximumiteration, false)
        numericalresults[indexi, 3] = methodtime
    end
    CSV.write("Numerical_Experiments/gssNumericalResults_$(sizeSEL).csv", numericalresults)
    return nothing
end
function run_sor_method(systemdimension::Integer, relaxationParameter::AbstractFloat)
    matrixA = Matrix{Float64}(undef, systemdimension, systemdimension)
    system_matrix_sparse(systemdimension, matrixA)
    vectorb = Vector{Float64}(undef, systemdimension)
    vector_independent_term(systemdimension, vectorb)
    vectorx_exactsolution = ones(systemdimension)
    vectorx_initial = zeros(systemdimension)
    #tolerance = 1.0e-10
    # Matrix Split
    matrixD =  convert(Matrix, Diagonal(matrixA))
    matrixL = tril(matrixA,-1)
    matrixU = triu(matrixA,1)
    println("===Sucessive Over-Relaxation Method===")
    # DataFrame for save method information
    sizeSEL = string(systemdimension)
    omegaparameter = string(relaxationParameter)
    numericalresults = DataFrame()
    numericalresults.Iterations = 25:25:200
    numericalresults.MethodError = zeros(8)
    numericalresults.RunTime = zeros(8)
    #sor_method(matrixD, matrixL, matrixU, vectorb, vectorx_exactsolution, vectorx_initial, maximunIteration, relaxationParameter, tolerance)
    for indexi = 1:8
        maximumiteration = indexi*25
        numericalresults[indexi, 2] = sor_method!(matrixD, matrixL, matrixU, vectorb, vectorx_exactsolution, vectorx_initial, maximumiteration, relaxationParameter, true)
        methodtime = @belapsed sor_method!($matrixD, $matrixL, $matrixU, $vectorb, $vectorx_exactsolution, $vectorx_initial, $maximumiteration, $relaxationParameter, false)
        numericalresults[indexi, 3] = methodtime
    end
    CSV.write("Numerical_Experiments/sorNumericalResults_$(sizeSEL)_$(omegaparameter).csv", numericalresults)
    return nothing
end
function run_conjuate_gradiente_method(systemdimension::Integer)
    matrixA = Matrix{Float64}(undef, systemdimension, systemdimension)
    system_matrix_sparse(systemdimension, matrixA)
    matrixsparseA = sparse(matrixA)
    vectorb = Vector{Float64}(undef, systemdimension)
    vector_independent_term(systemdimension, vectorb)
    vectorx_exactsolution = ones(systemdimension)
    vectorx_initial = zeros(systemdimension)
    #tolerance = 1.0e-10
    println("===Conjugate Gradient Method===")
    # DataFrame for save method information
    sizeSEL = string(systemdimension)
    numericalresults = DataFrame()
    numericalresults.Iterations = 5:5:40
    numericalresults.MethodError = zeros(8)
    numericalresults.RunTime = zeros(8)
    #conjugate_gradient_method(systemdimension, matrixsparseA, vectorb, vectorx_exactsolution, vectorx_initial, tolerance)
    for indexi = 1:8
        maximumiteration = indexi*5
        numericalresults[indexi, 2] = conjugate_gradient_method!(matrixsparseA, vectorb, vectorx_exactsolution, vectorx_initial, maximumiteration, true)
        methodtime = @belapsed conjugate_gradient_method!($matrixsparseA, $vectorb, $vectorx_exactsolution, $vectorx_initial, $maximumiteration, false)
        numericalresults[indexi, 3] = methodtime
    end
    CSV.write("Numerical_Experiments/cgNumericalResults_$(sizeSEL).csv", numericalresults)
    return nothing
end
function run_biconjuate_gradiente_stabilized_method(systemdimension::Integer)
    matrixA = Matrix{Float64}(undef, systemdimension, systemdimension)
    system_matrix_sparse(systemdimension, matrixA)
    matrixsparseA = sparse(matrixA)
    vectorb = Vector{Float64}(undef, systemdimension)
    vector_independent_term(systemdimension, vectorb)
    vectorx_exactsolution = ones(systemdimension)
    vectorx_initial = zeros(systemdimension)
    #tolerance = 1.0e-10
    println("===Biconjugate Gradient Stabilized Method===")
    # DataFrame for save method information
    sizeSEL = string(systemdimension)
    numericalresults = DataFrame()
    numericalresults.Iterations = 5:5:40
    numericalresults.MethodError = zeros(8)
    numericalresults.RunTime = zeros(8)
    #biconjugate_gradient_stabilized_method(systemdimension, matrixsparseA, vectorb, vectorx_exactsolution, vectorx_initial, tolerance)
    for indexi = 1:8
        maximumiteration = indexi*5
        numericalresults[indexi, 2] = biconjugate_gradient_stabilized_method!(matrixsparseA, vectorb, vectorx_exactsolution, vectorx_initial, maximumiteration, true)
        methodtime = @belapsed biconjugate_gradient_stabilized_method!($matrixsparseA, $vectorb, $vectorx_exactsolution, $vectorx_initial, $maximumiteration, false)
        numericalresults[indexi, 3] = methodtime
    end
    CSV.write("Numerical_Experiments/bicgstabNumericalResults_$(sizeSEL).csv", numericalresults)
    return nothing
end
function run_restarted_generalized_minimal_residual_method(systemdimension::Integer, restartparameter::Integer)
    matrixA = Matrix{Float64}(undef, systemdimension, systemdimension)
    system_matrix_sparse(systemdimension, matrixA)
    matrixsparseA = sparse(matrixA)
    vectorb = Vector{Float64}(undef, systemdimension)
    vector_independent_term(systemdimension, vectorb)
    vectorx_exactsolution = ones(systemdimension)
    vectorx_initial = zeros(systemdimension)
    tolerance = 1.0e-10
    println("===Generalized Minimal Residual Method===")
    # DataFrame for save method information
    sizeSEL = string(systemdimension)
    usedparameter = string(restartparameter)
    numericalresults = DataFrame()
    numericalresults.Iterations = 5:5:40
    numericalresults.MethodError = zeros(8)
    numericalresults.RunTime = zeros(8)
    #restarted_generalized_minimal_residual_method(systemdimension, matrixsparseA, vectorb, vectorx_initial, vectorx_exactsolution, restartparameter, tolerance)
    for indexi = 1:8
        maximumiteration = indexi*5
        numericalresults[indexi, 2] = restarted_generalized_minimal_residual_method!(systemdimension, matrixsparseA, vectorb, vectorx_initial, vectorx_exactsolution, restartparameter, maximumiteration, tolerance, true)
        methodtime = @belapsed restarted_generalized_minimal_residual_method!($systemdimension, $matrixsparseA, $vectorb, $vectorx_initial, $vectorx_exactsolution, $restartparameter, $maximumiteration, $tolerance, false)
        numericalresults[indexi, 3] = methodtime
    end
    CSV.write("Numerical_Experiments/restartedgmresNumericalResults_$(sizeSEL)_$(usedparameter).csv", numericalresults)
    return nothing
end