%%cu
#include <cuda_runtime.h>
#include <iostream>

__global__ void transposeKernel(float* A, float* B, int m, int n) {
    int row = blockIdx.y*blockDim.y + threadIdx.y;
    int col = blockIdx.x*blockDim.x + threadIdx.x;
    if (row < n && col < m) {
        B[row*m + col] = A[col*n + row];
    }
}

int main() {
    int m = 3, n = 3;
    int size = m*n*sizeof(float);
    float* A, * B;
    cudaMalloc(&A, size);
    cudaMalloc(&B, size);
    // initialize A with sample values
    float A_host[9] = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    cudaMemcpy(A, A_host, size, cudaMemcpyHostToDevice);
    dim3 block(3, 3);
    dim3 grid((m + block.x - 1) / block.x, (n + block.y - 1) / block.y);
    transposeKernel<<<grid, block>>>(A, B, m, n);
    cudaDeviceSynchronize();
    // print the input matrix A
    std::cout << "Matrix A:" << std::endl;
    for (int i = 0; i < 9; i++) {
        std::cout << A_host[i] << " ";
        if ((i + 1) % 3 == 0) std::cout << std::endl;
    }
    // print the transposed matrix B
    float B_host[9];
    cudaMemcpy(B_host, B, size, cudaMemcpyDeviceToHost);
    std::cout << "Matrix A transposed (A^T):" << std::endl;
    for (int i = 0; i < 9; i++) {
        std::cout << B_host[i] << " ";
        if ((i + 1) % 3 == 0) std::cout << std::endl;
    }
    cudaFree(A);
    cudaFree(B);
    return 0;
}