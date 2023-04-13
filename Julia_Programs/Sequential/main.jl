using BenchmarkTools
using LinearAlgebra: Diagonal, tril, triu
using SparseArrays: sparse, SparseMatrixCSC
include("src/IterativesSolversMethods.jl")
function run_jacobi_method(systemdimension::Integer, maximunIteration::Integer)
    matrixA = system_matrix_sparse(systemdimension)
    vectorb = Vector{Float64}(undef, systemdimension)
    vector_independent_term(systemdimension, vectorb)
    vectorx_exactsolution = ones(systemdimension)
    vectorx_initial = zeros(systemdimension)
    tolerance = 1.0e-10
    # Matrix Split
    matrixD =  convert(Matrix, Diagonal(matrixA))
    matrixL = tril(matrixA,-1)
    matrixU = triu(matrixA,1)
    #println("Jacobi Method")
    #jacobi_method(matrixD, matrixL, matrixU, vectorb, vectorx_exactsolution, vectorx_initial, maximunIteration, tolerance)
    jacobi_method!(matrixD, matrixL, matrixU, vectorb, vectorx_exactsolution, vectorx_initial, maximunIteration)
    return nothing
end
function run_gauss_seidel_method(systemdimension::Integer, maximunIteration::Integer)
    matrixA = system_matrix_sparse(systemdimension)
    vectorb = Vector{Float64}(undef, systemdimension)
    vector_independent_term(systemdimension, vectorb)
    vectorx_exactsolution = ones(systemdimension)
    vectorx_initial = zeros(systemdimension)
    tolerance = 1.0e-10
    # Matrix Split
    matrixD =  convert(Matrix, Diagonal(matrixA))
    matrixL = tril(matrixA,-1)
    matrixU = triu(matrixA,1)
    #println("Gauss-Seidel Method")
    #gauss_seidel_method(matrixD, matrixL, matrixU, vectorb, vectorx_exactsolution, vectorx_initial, maximunIteration, tolerance)
    gauss_seidel_method!(matrixD, matrixL, matrixU, vectorb, vectorx_exactsolution, vectorx_initial, maximunIteration)
    return nothing
end
function run_sor_method(systemdimension::Integer, maximunIteration::Integer, relaxationParameter::Real)
    matrixA = system_matrix_sparse(systemdimension)
    vectorb = Vector{Float64}(undef, systemdimension)
    vector_independent_term(systemdimension, vectorb)
    vectorx_exactsolution = ones(systemdimension)
    vectorx_initial = zeros(systemdimension)
    tolerance = 1.0e-10
    # Matrix Split
    matrixD =  convert(Matrix, Diagonal(matrixA))
    matrixL = tril(matrixA,-1)
    matrixU = triu(matrixA,1)
    #println("Sucessive Over Relaxation Method")
    #sor_method(matrixD, matrixL, matrixU, vectorb, vectorx_exactsolution, vectorx_initial, maximunIteration, relaxationParameter, tolerance)
    sor_method!(matrixD, matrixL, matrixU, vectorb, vectorx_exactsolution, vectorx_initial, maximunIteration, relaxationParameter)
    return nothing
end
function run_conjuate_gradiente_method(systemdimension::Integer, maximumiteration::Integer)
    matrixA = system_matrix_sparse(systemdimension)
    matrixsparseA = sparse(matrixA)
    vectorb = Vector{Float64}(undef, systemdimension)
    vector_independent_term(systemdimension, vectorb)
    vectorx_exactsolution = ones(systemdimension)
    vectorx_initial = zeros(systemdimension)
    tolerance = 1.0e-10
    #println("Conjugate Gradient Method")
    #conjugate_gradient_method(systemdimension, matrixsparseA, vectorb, vectorx_exactsolution, vectorx_initial, tolerance)
    conjugate_gradient_method!(systemdimension, matrixsparseA, vectorb, vectorx_exactsolution, vectorx_initial, maximumiteration)
    return nothing
end
function run_biconjuate_gradiente_stabilized_method(systemdimension::Integer, maximumiteration::Integer)
    matrixA = system_matrix_sparse(systemdimension)
    matrixsparseA = sparse(matrixA)
    vectorb = Vector{Float64}(undef, systemdimension)
    vector_independent_term(systemdimension, vectorb)
    vectorx_exactsolution = ones(systemdimension)
    vectorx_initial = zeros(systemdimension)
    tolerance = 1.0e-10

    #println("Biconjugate Gradient Stabilized Method")
    #biconjugate_gradient_stabilized_method(systemdimension, matrixsparseA, vectorb, vectorx_exactsolution, vectorx_initial, tolerance)
    biconjugate_gradient_stabilized_method!(systemdimension, matrixsparseA, vectorb, vectorx_exactsolution, vectorx_initial, maximumiteration)

    return nothing
end
function run_restarted_generalized_minimal_residual_method(systemdimension::Integer, restartparameter::Integer, maximumrestart::Integer)
    matrixA = system_matrix_sparse(systemdimension)
    matrixsparseA = sparse(matrixA)
    vectorb = Vector{Float64}(undef, systemdimension)
    vector_independent_term(systemdimension, vectorb)
    vectorx_exactsolution = ones(systemdimension)
    vectorx_initial = zeros(systemdimension)
    tolerance = 1.0e-10

    #println("Generalized Minimal Residual Method")
    #restarted_generalized_minimal_residual_method(systemdimension, matrixsparseA, vectorb, vectorx_initial, vectorx_exactsolution, restartparameter, tolerance)
    restarted_generalized_minimal_residual_method!(systemdimension, matrixsparseA, vectorb, vectorx_initial, vectorx_exactsolution, restartparameter, maximumrestart, tolerance)

    return nothing
end