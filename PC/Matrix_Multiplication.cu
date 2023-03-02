#include <cuda_runtime.h>
#include <iostream>

__global__ void matmulKernel(float* A, float* B, float* C) {
    int row = blockIdx.y*blockDim.y + threadIdx.y;
    int col = blockIdx.x*blockDim.x + threadIdx.x;
    if(row < 3 && col < 3) {
        float value = 0;
        for (int i = 0; i < 3; i++) {
            value += A[row*3 + i] * B[i*3 + col];
        }
        C[row*3 + col] = value;
    }
}

int main() {
    float* A, * B, * C;
    cudaMalloc(&A, 9 * sizeof(float));
    cudaMalloc(&B, 9 * sizeof(float));
    cudaMalloc(&C, 9 * sizeof(float));
    // initialize A and B with sample values
    float A_host[9] = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    float B_host[9] = {9, 8, 7, 6, 5, 4, 3, 2, 1};
    cudaMemcpy(A, A_host, 9 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(B, B_host, 9 * sizeof(float), cudaMemcpyHostToDevice);
    dim3 block(3, 3);
    dim3 grid((3 + block.x - 1) / block.x, (3 + block.y - 1) / block.y);
    matmulKernel<<<grid, block>>>(A, B, C);
    cudaDeviceSynchronize();
    std::cout<<"\n **************** CUDA Program for Matrix Multiplicaion *********************** \n";
    // print the input matrices A and B
    std::cout << "Matrix A:" << std::endl;
    for (int i = 0; i < 9; i++) {
        std::cout << A_host[i] << " ";
        if ((i + 1) % 3 == 0) std::cout << std::endl;
    }
    std::cout << "Matrix B:" << std::endl;
    for (int i = 0; i < 9; i++) {
        std::cout << B_host[i] << " ";
        if ((i + 1) % 3 == 0) std::cout << std::endl;
    }
    // print the result matrix C
    float C_host[9];
    cudaMemcpy(C_host, C, 9 * sizeof(float), cudaMemcpyDeviceToHost);
    std::cout << "Matrix C (A*B):" << std::endl;
    for (int i = 0; i < 9; i++) {
        std::cout << C_host[i] << " ";
        if ((i + 1) % 3 == 0) std::cout << std::endl;
    }
    cudaFree(A);
    cudaFree(B);
    cudaFree(C);
    return 0;
}
