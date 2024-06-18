using LinearAlgebra
using Plots
using SparseArrays
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

N = 512
tA = Matrix{Float64}(undef, N, N)
b = Vector{Float64}(undef, N)
x_0 = zeros(N);
system_matrix_sparse(N, tA)
vector_independent_term(N, b);
A = sparse(tA)

function jacobi_sparse(A::SparseMatrixCSC{Float64,Int64}, b::Vector{Float64}, ϵ=1e-5)
    D = Diagonal(A)
    N = -triu(A, 1) - tril(A, -1)
    x = zeros(length(b))
    normres = []
    for i = 1:40
        x = D \ (N * x + b)
        normres = [normres; norm(A * x - b)]
    end
    return normres
end

function gss_sparse(A::SparseMatrixCSC{Float64,Int64}, b::Vector{Float64}, ϵ=1e-5)
    U = -triu(A, 1)
    L = tril(A, 0)
    x = zeros(length(b))
    normres = []
    for i = 1:40
        # x = inv(L₀)*(U*x + b)
        x = L \ (U * x + b)
        normres = [normres; norm(A * x - b)]
    end
    return normres
end

function sor_sparse(A::SparseMatrixCSC{Float64,Int64}, b::Vector{Float64}, ω=1.2::Float64, ϵ=1e-5)

    D = Diagonal(A)
    U = -triu(A, 1)
    L = -tril(A, -1)
    x = zeros(length(b))

    DL = D - ω * L
    UD = ω * U + (1.0 - ω) * D
    f = ω * b
    normres = []
    for i = 1:40
        # x = inv(L′) * (U′*x + b′)
        x = DL \ (UD * x + f)
        normres = [normres; norm(A * x - b)]
    end
    return normres
end

function omega_sor()
    color = ["#fde7f7", "#f471d1", "#180212", "#8e0b6b", "#d510a1"]
    graf1 = plot()
    ω = 1.0
    for i = 1:5
        normres_srss = []
        normres_srss = sor_sparse(A, b, ω)
        graf1 = scatter!(normres_srss, markersize=4, label="\$\\omega=$ω\$", c=color[i])
        graf1 = plot!(xaxis=("iterations"), yaxis=("residuals", :log))
        ω = ω + 0.05
    end
    savefig(graf1, "OmegaSor")
end

# omega_sor()

# normres_jacobi = jacobi_sparse(A, b)
# normres_gauss_seidel = gss_sparse(A, b)
# normres_sor = sor_sparse(A, b)
# graf2 = scatter(normres_jacobi, markersize=4, label="Jacobi", c="red")
# scatter!(normres_gauss_seidel, markersize=4, label="Gauss-Seidel", c="green")
# scatter!(normres_sor, markersize=4, label="SOR", c="blue")
# plot!(xaxis=("iteraciones"), yaxis=("residuales", :log))
# plot!(title="Convergencia de Jacobi, Gauss-Seidel y SOR")
# savefig(graf2, "Convergencia")

BenchmarkTools.DEFAULT_PARAMETERS.samples = 20
bJa = @benchmark jacobi_sparse(A, b)
bGSSS = @benchmark gss_sparse(A, b)
bSOR = @benchmark sor_sparse(A, b);

display(bJa)
display(bGSSS)
display(bSOR)


function cgm_sparse(
    A::SparseMatrixCSC{Float64,Int64},
    b::AbstractVector,
    ϵ=1e-5)

    x_c = zeros(length(b))
    r_c = b - A * x_c
    d_c = r_c
    normres_cgm = []

    for k = 1:40
        vectorAD = A * d_c
        alpha = dot(r_c, r_c) / dot(vectorAD, d_c)
        x_n = x_c + alpha * d_c
        # normres_cgm = [normres_cgm;norm(b-A*x_n)]
        # Stop Condition
        # methodError = norm(x_n - vectorX_exactSolution) / norm(x_n)
        # if methodError <= tolerance
        #     iterationNumber = k
        #     break
        # end
        r_n = r_c - alpha * vectorAD
        normres_cgm = [normres_cgm; norm(r_n)]
        beta = dot(r_n, r_n) / dot(r_c, r_c)
        d_n = r_n + beta * d_c
        # Update
        x_c = x_n
        r_c = r_n
        d_c = d_n
    end

    return normres_cgm
end

function cgm_precondicionado_sparse(
    A::SparseMatrixCSC{Float64,Int64},
    b::AbstractVector,
    precondicionador::String,
    ϵ=1e-5)

    D = Diagonal(A)
    U = triu(A, 1)
    L = tril(A, -1)
    α = 1.2

    if precondicionador == "Jacobi"
        Ms = [D]
    end
    if precondicionador == "SOR"
        M_sor_1 = I + (α * L * inv(D))
        M_sor_2 = D + α * U
        Ms = [M_sor_1, M_sor_2]
    end

    x_c = zeros(length(b))
    r_c = b - A * x_c
    z = r_c
    [z = M \ z for M ∈ Ms]
    d_c = z
    normres_cgm = []

    for k = 1:40
        Ad = A * d_c
        alpha = dot(r_c, z) / dot(Ad, d_c)
        x_n = x_c + alpha * d_c
        r_n = r_c - alpha * Ad
        normres_cgm = [normres_cgm; norm(r_n)]
        z_next = r_n
        [z_next = M \ z_next for M ∈ Ms]
        beta = dot(r_n, z) / dot(r_c, z)
        d_n = z_next + beta * d_c
        # Update
        x_c = x_n
        r_c = r_n
        d_c = d_n
        z = z_next
    end

    return normres_cgm
end

# normres_cgm = cgm_sparse(A, b);
# normres_cgm_pJ = cgm_precondicionado_sparse(A, b, "Jacobi");
# normres_cgm_pS = cgm_precondicionado_sparse(A, b, "SOR");

# f2 = scatter(normres_cgm, markersize=4, label="Sin Precondicionador", c="#6b0851")
# scatter!(normres_cgm_pJ, markersize=4, label="Precondicionador Jacobi", c="blue")
# scatter!(normres_cgm_pS, markersize=4, label="Precondicionador SOR", c="green")
# plot!(xaxis=("iteraciones"), yaxis=("residuales", :log))
# plot!(title="Convergencia del Método del gradiente conjugado")
# savefig(f2, "convergencia2")

bcg = @benchmark cgm_sparse(A, b);
bcgpS = @benchmark cgm_precondicionado_sparse(A, b, "SOR");

display(bcg)
display(bcgpS)