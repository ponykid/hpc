using CUDA

# Defining Matrix dimensions
row = 9
col = 8

# Declaring Matrix
pic = Array{Float32}(rand(row,col))

function scalePic(pic::Matrix{Float32}, row::Int, col::Int)
    # Allocate and transfer matrix to GPU Memory
    d_pic = CuArray(pic)

    # Kernel Invocation
    dimBlock = (2, 2) # Threads in a block
    dimGrid = (Int(ceil(col/dimBlock[1])), Int(ceil(row/dimBlock[2]))) # Blocks in a grid

    @cuda threads=dimBlock blocks=dimGrid scalePicKernel!(d_pic, row, col)
    
    # Transfer results back to CPU
    pic = Array(d_pic)

    return pic
end

function scalePicKernel!(pic, row, col)
    rowIdx = blockDim().y * (blockIdx().y-1) + threadIdx().y
    colIdx = blockDim().x * (blockIdx().x-1) + threadIdx().x 
    
    if (rowIdx ≤ row) && (colIdx ≤ col)
        pic[rowIdx,colIdx] *= 2
    end

    return nothing
end

# Calling functions
pic_scaled = scalePic(pic, row, col)

@assert pic*2 == pic_scaled