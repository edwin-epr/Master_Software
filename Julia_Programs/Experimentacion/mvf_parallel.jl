function parallelsor(A, b, ω=1.2, ϵ=1e-5)
    D = Diagonal(A)
    U = -triu(A, 1)
    L = -tril(A, -1)

    DL = D - ω * L
    UD = ω * U + (1.0 - ω) * D
    f = ω * b

    d_UD = CuSparseMatrixCSC{Float64}(UD)
    d_f = CuArray{Float64}(f)

    d_A = CuSparseMatrixCSC{Float64}(A)
    d_b = CuArray{Float64}(b)
    d_x = CUDA.zeros(Float64, size(b))

    normres = []
    norma = CUBLAS.norm(d_A * d_x - d_b)
    while norma > ϵ
        g = Array(d_UD * d_x + d_f)
        d_x = CuArray{Float64}(DL \ g)
        norma = CUBLAS.norm(d_A * d_x - d_b)
        normres = [normres; norma]
    end
    return normres
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
        r_n = r_c - alpha * Adc
        norma = norm(r_n)
        normres_cgm = [normres_cgm; norma]
        # Stop Condition
        if norma < ϵ
            break
        end
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
        α = 1.1
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