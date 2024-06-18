using LinearAlgebra
using Plots
using CUDA, CUDA.CUBLAS, CUDA.CUSOLVER
using BenchmarkTools

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

N = 512
A = Matrix{Float64}(undef, N, N)
b = Vector{Float64}(undef, N)
x_0 = zeros(N);
system_matrix_sparse(N, A)
vector_independent_term(N, b);

function paralleljacobi(A, b, ϵ=1e-5)
    d = diag(A)
    d_d = CuArray{Float64}(d)
    N = -triu(A, 1) - tril(A, -1)
    d_N = CuSparseMatrixCSC{Float64}(N)

    d_A = CuSparseMatrixCSC{Float64}(A)
    d_b = CuArray{Float64}(b)
    d_x = CUDA.zeros(Float64, size(b))
    # counter = 0
    # while norm(d_A*d_x - d_b) > ϵ
    #     d_x = (d_N*d_x + d_b) ./ d_d
    #     counter+=1
    # end
    # display(d_x)
    normres = []
    for i = 1:40
        d_x = (d_N * d_x + d_b) ./ d_d
        normres = [normres; CUBLAS.norm(d_A * d_x - d_b)]
    end
    return normres
end

function parallelgaussseidel(A, b, ϵ=1e-5)
    U = -triu(A, 1)
    L = tril(A, 0)
    d_U = CuSparseMatrixCSC{Float64}(U)

    d_A = CuSparseMatrixCSC{Float64}(A)
    d_b = CuArray{Float64}(b)
    d_x = CUDA.zeros(Float64, size(b))

    # counter = 0
    # while CUBLAS.norm(d_A*d_x - d_b) > ϵ
    #     d_F = d_U*d_x + d_b
    #     d_x = d_L \ d_F
    #     counter+=1
    # end
    # return counter

    normres = []
    for i = 1:40
        f = Array(d_U * d_x + d_b)
        d_x = CuArray{Float64}(L \ f)
        normres = [normres; CUBLAS.norm(d_A * d_x - d_b)]
    end
    return normres
end

function parallelsor(A, b, ω=1.2, ϵ=1e-5)
    D = Diagonal(A)
    U = -triu(A, 1)
    L = -tril(A, -1)

    # L′ = D-ω*L
    # U′ = ω*U+(1.0-ω)*D
    # b′ = ω*b
    DL = D - ω * L
    UD = ω * U + (1.0 - ω) * D
    f = ω * b

    d_UD = CuSparseMatrixCSC{Float64}(UD)
    d_f = CuArray{Float64}(f)

    d_A = CuSparseMatrixCSC{Float64}(A)
    d_b = CuArray{Float64}(b)
    d_x = CUDA.zeros(Float64, size(b))

    # counter = 0
    # while CUBLAS.norm(d_A*d_x - d_B) > ϵ
    #     d_G = d_UD*d_x + d_f
    #     d_x = d_DL \ d_G
    #     counter+=1
    # end
    # return counter
    normres = []
    for i = 1:40
        g = Array(d_UD * d_x + d_f)
        d_x = CuArray{Float64}(DL \ g)
        normres = [normres; CUBLAS.norm(d_A * d_x - d_b)]
    end
    return normres
end

BenchmarkTools.DEFAULT_PARAMETERS.samples = 20
bJap = @benchmark paralleljacobi(A, b)
bGSSSp = @benchmark parallelgaussseidel(A, b)
bSORp = @benchmark parallelsor(A, b);

display(bJap)
display(bGSSSp)
display(bSORp)

display("CGM")

function parallel_cgm(A, b, tolerance=1e-5)

    d_A = CuSparseMatrixCSC{Float64}(A)
    d_B = CuArray(b)

    x_c = zeros(Float64, size(b))
    d_X_c = CuArray(x_c)
    # r_c = b - A * x_c
    r_c = Array(d_B - d_A * d_X_c)
    d_c = r_c
    d_D_c = CuArray(d_c)
    normres_cgm = []

    for k = 1:40
        Ad_c = Array(d_A * d_D_c)
        alpha = dot(r_c, r_c) / dot(Ad_c, d_c)
        x_n = x_c + alpha * d_c
        # normres_cgm = [normres_cgm;norm(b-A*x_n)]
        # Stop Condition
        # methodError = norm(x_n - vectorX_exactSolution) / norm(x_n)
        # if methodError <= tolerance
        #     iterationNumber = k
        #     break
        # end
        r_n = r_c - alpha * Ad_c
        normres_cgm = [normres_cgm; norm(r_n)]
        beta = dot(r_n, r_n) / dot(r_c, r_c)
        d_n = r_n + beta * d_c
        # Update
        x_c = x_n
        r_c = r_n
        d_c = d_n
        d_D_c = CuArray(d_c)
    end

    return normres_cgm
end

function parallel_cgm_precondicionado(A::AbstractMatrix, b::AbstractVector, tolerance=1e-5)

    d_A = CuSparseMatrixCSC{Float64}(A)
    d_b = CuArray(b)

    # M = Diagonal(A)
    D = Diagonal(A)
    U = triu(A, 1)
    L = tril(A, -1)
    α = 1.13
    M_sor_1 = I + (α * L * inv(D))
    M_sor_2 = D + α * U

    d_M_sor1 = CuArray(M_sor_1)
    d_M_sor2 = CuArray(M_sor_2)
    Ms = [d_M_sor1, d_M_sor2]

    x_c = zeros(length(b))
    d_x_c = CuArray(x_c)

    d_r_c = d_b - d_A * d_x_c
    r_c = Array(d_r_c)

    d_z = d_r_c
    [d_z = M \ d_z for M ∈ Ms]
    z = Array(d_z)

    p_c = z
    d_p_c = CuArray(p_c)

    normres_cgm = []

    for k = 1:40
        Adc = Array(d_A * d_p_c)
        alpha = dot(r_c, z) / dot(Adc, p_c)
        x_n = x_c + alpha * p_c
        # normres_cgm = [normres_cgm;norm(b-A*x_n)]
        # Stop Conditioj
        # methodError = norm(x_n - vectorX_exactSolution) / norm(x_n)
        # if methodError <= tolerance
        #     iterationNumber = k
        #     break
        # end
        r_n = r_c - alpha * Adc
        normres_cgm = [normres_cgm; norm(r_n)]
        d_z_n = CuArray(r_n)
        [d_z_n = M \ d_z_n for M ∈ Ms]
        z_n = Array(d_z_n)
        beta = dot(r_n, z_n) / dot(r_c, z)
        p_n = z_n + beta * p_c
        # Update
        x_c = x_n
        r_c = r_n
        p_c = p_n
        d_p_c = CuArray(p_c)
        z = z_n
    end

    return normres_cgm
end

BenchmarkTools.DEFAULT_PARAMETERS.samples = 20
bcgmpsp = @benchmark parallel_cgm(A, b)
bcgmpp = @benchmark parallel_cgm_precondicionado(A, b);

display(bcgmpsp)
display(bcgmpp)