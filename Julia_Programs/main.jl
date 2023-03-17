using BenchmarkTools
include("functions_library.jl")

function run_conjuate_gradiente_method(systemDimension::Int64)
    mtxA = system_matrix(systemDimension)
    vtrB = vector_independant_terms(systemDimension)
    vtrExactSol = ones(systemDimension)
    vtrX_initial = zeros(systemDimension)
    tolerance = 1.0e-10

    println("Conjugate Gradient Method")
    methodError, iterationNumber = conjugate_gradient_method(systemDimension, mtxA, vtrB, vtrExactSol, vtrX_initial, tolerance)
    println("El número de iteraciones es de $(iterationNumber).")
    println("El error del método es: $(methodError).")
    @btime conjugate_gradient_method($systemDimension, $mtxA, $vtrB, $vtrExactSol, $vtrX_initial, $tolerance)
end