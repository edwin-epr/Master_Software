using LinearAlgebra
using Plots
using SparseArrays
using CUDA, CUDA.CUBLAS, CUDA.CUSOLVER, CUDA.CUSPARSE
using BenchmarkTools

# Functions 
# System Equation Linear Sparse
function system_matrix_sparse(systemDimension::Integer, matrixA::AbstractMatrix)
    vectorDiagonalMain = 3 * ones(systemDimension)
    vectorDiagonalUpperLower = -1 * ones(systemDimension - 1)
    matrixA .= convert(Matrix, Tridiagonal(vectorDiagonalUpperLower, vectorDiagonalMain, vectorDiagonalUpperLower))
    for i = 1:systemDimension
        if matrixA[i, systemDimension-i+1] == 0.0
            matrixA[i, systemDimension-i+1] = 0.5
        end
    end
    return nothing
end
# 
function vector_independent_term(systemDimension::Integer, vectorB::AbstractVector)
    vectorB[1] = 2.5
    vectorB[systemDimension] = 2.5
    position::Integer = floor(systemDimension / 2)
    for i = 2:systemDimension-1
        if i == position || i == position + 1
            vectorB[i] = 1.0
        else
            vectorB[i] = 1.5
        end
    end
    return nothing
end

N = 128
tA = Matrix{Float64}(undef, N, N)
b = Vector{Float64}(undef, N)
x_0 = zeros(N);
system_matrix_sparse(N, tA)
vector_independent_term(N, b);
A = sparse(tA)
modulo = 10
precondicionador = "Jacobi"

# ****************************************************************************************************************************

function leastsquares(H, r)
    r′ = zeros(size(H)[1])
    r′[1] = norm(r)
    x = H \ r′
end

function reiniciarvariables(x, A, b)
    k = 1
    x₀ = x
    r = b - A*x₀
    H = zeros(2,1)
    q = [r / norm(r)]
    return x₀, r, q, k, H
end

function precondition(name, A, b)
    if name == "Jacobi"
        M_jacobi = Diagonal(A)
        return [M_jacobi]

    elseif name == "Gauss-Seidel"
        D = Diagonal(A)
        U = triu(A,1) 
        L = tril(A,-1)
        M_gauss_seidel_1 = I+(L*inv(D))
        M_gauss_seidel_2 = D+U
        return [M_gauss_seidel_1, M_gauss_seidel_2]

    elseif name == "SOR"
        α = 1.8
        D = Diagonal(A)
        U = triu(A,1) 
        L = tril(A,-1)
        M_sor_1 = I+(α*L*inv(D))
        M_sor_2 = D+α*U
        return [M_sor_1, M_sor_2]
    end
end

function reiniciarvariablesprecondicionado(x::Vector, A::AbstractMatrix, b::Vector{Float64}, Ms::Vector)
    k = 1
    x₀ = x
    H = zeros(2,1) 
    residual₀ = b - A*x₀
    [residual₀ = M\residual₀ for M ∈ Ms]
    q = [residual₀ / norm(residual₀)]
    return x₀, residual₀, q, k, H
end

########################
function parallelgmresreiniciado(A::SparseMatrixCSC{Float64, Int64}, b::Vector{Float64}, ϵ = 1e-5)
    x₀ = zeros(length(b))
    residual₀ = b - A*x₀
    q = [residual₀ / norm(residual₀)]
    normres_gmresrei = []
    
    k = 1
    x = x₀
    H = zeros(2,1)
    residual = residual₀
    counter = 0
    
    d_A = CuSparseMatrixCSR{Float64}(A)
    d_b = CuArray{Float64}(b)
    d_x₀ = CuArray(x₀)
    d_x = CuArray(x)    

    while norm(residual) > ϵ
        y = A*q[k]
        for j ∈ 1:k
            H[j,k] = q[j]' * y
            y -= H[j,k]*q[j]
        end
        H[k+1,k] = norm(y)
        push!(q, y/H[k+1,k])
        H = vcat(H, zeros(1, size(H)[2]))
        H = hcat(H, zeros(size(H)[1], 1))
        k += 1
       if k % modulo == 0 
            c = leastsquares(H, residual₀)
            Q = hcat(q...)

            dQ = CuArray(Q)
            dc = CuArray(c)
            
            d_x = dQ*dc + d_x₀
            residual = Array(d_A*d_x - d_b)
            x = Array(d_x)
            x₀, residual₀, q, k, H = reiniciarvariables(x, A, b)
            d_x₀ = CuArray(x₀)
            counter+=1
        end
        normres_gmresrei = [normres_gmresrei;norm(residual₀)]
    end
    return normres_gmresrei 
end
########################
function parallelgmresprecondicionadoreiniciado(A::AbstractMatrix, b::Vector{Float64}, precondition_name::String, ϵ = 1e-5)
    x₀ = zeros(length(b))
    Ms = precondition(precondition_name, A, b)
    residual₀ = b - A*x₀
    [residual₀ = M\residual₀ for M ∈ Ms]
    q = [residual₀ / norm(residual₀)]
    normres_gmresreipre = []

    k = 1
    x = x₀
    H = zeros(2,1)
    residual = residual₀
    counter = 0

    d_A = CuSparseMatrixCSR{Float64}(A)
    d_b = CuArray{Float64}(b)
    d_x₀ = CuArray(x₀)
    d_x = CuArray(x)  

    while norm(residual) > ϵ
        ω = A*q[k]
        [ω = M\ω for M ∈ Ms]
        for j ∈ 1:k
            H[j,k] = q[j]' * ω
            ω -= H[j,k]*q[j]
        end
        H[k+1,k] = norm(ω)
        push!(q, ω/H[k+1,k])
        H = vcat(H, zeros(1, size(H)[2]))
        H = hcat(H, zeros(size(H)[1], 1))
        k += 1
        if k % modulo == 0
            c = leastsquares(H, residual₀)
            Q = hcat(q...)

            dQ = CuArray(Q)
            dc = CuArray(c)
            
            d_x = dQ*dc + d_x₀
            residual = Array(d_A*d_x - d_b)
            [residual = M\residual for M ∈ Ms]
            x = Array(d_x)
            x₀, residual₀, q, k, H = reiniciarvariablesprecondicionado(x, A, b, Ms)
            d_x₀ = CuArray(x₀)
        end
        normres_gmresreipre = [normres_gmresreipre;norm(residual₀)]
        counter+=1
    end
    return normres_gmresreipre
end
########################

BenchmarkTools.DEFAULT_PARAMETERS.samples = 20
println()
println("N = ", N, " precondicionador: ", precondicionador)
println()
BgmresRei = @benchmark parallelgmresreiniciado(A, b)
BgmresPreRei = @benchmark parallelgmresprecondicionadoreiniciado(A, b, precondicionador);

display(BgmresRei)
display(BgmresPreRei)