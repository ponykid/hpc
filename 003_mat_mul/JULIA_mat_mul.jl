using CUDA

# Dimesions of the matrices
m = 6
k = 8
n = 10

# Initialise Matrices 
A = rand(m,k)
B = rand(k,n)

function matMul(A, B)
    # Retrieve dimensions
    (m, k) = size(A)
    (k, n) = size(B)

    # Allocation and transfering matrices to GPU memory
    d_A = CuArray(A)
    d_B = CuArray(B)
    d_C = CUDA.zeros(m,n)

    # Kernel Invocation
    dimBlock = (2, 2)
    dimGrid = (Int(ceil(n/dimBlock[1])), Int(ceil(m/dimBlock[2])))

    @cuda blocks=dimGrid threads=dimBlock matMulKernel!(d_A, d_B, d_C, m, k, n)

    C = Array(d_C)

    return C
end

function matMulKernel!(d_A, d_B, d_C, m, k, n)
    rowIdx = blockDim().y * (blockIdx().y-1) + threadIdx().y
    colIdx = blockDim().x * (blockIdx().x-1) + threadIdx().x

    if (rowIdx ≤ m) && (colIdx ≤ n)
        sum = 0
        for idx = 1:k
            sum += d_A[rowIdx, idx] * d_B[idx, colIdx]
        end
        d_C[rowIdx,colIdx] = sum
    end

    return
end

C = matMul(A, B)

@assert C ≈ A*B