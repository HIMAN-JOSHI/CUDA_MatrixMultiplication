// This program demonstrates Matrix Multiplication using CUDA.

#include<stdio.h>

// cuda headers
#include<cuda.h>

//macros
#define BLOCK_WIDTH 32

// global variables
int *hostA = NULL;
int *hostB = NULL;
int *hostC = NULL;
int *gold = NULL;

int *deviceA = NULL;
int *deviceB = NULL;
int *deviceC = NULL;

// cuda kernel function.
__global__ void matrixMultiplicationGPU(int *A, int *B, int *C, int numARows, int numACols, int numBCols, int numCCols){

    // variable declarations
    int row = blockIdx.y * blockDim.y + threadIdx.y; // replaces for(int i = 0 ; i < numARows; ++i)

    int column = blockIdx.x * blockDim.x + threadIdx.x; // replaces for(int j = 0; j < numBColsl; ++j)

    // code
    if((row < numARows) && (column < numBCols)){
        int value = 1;
        for(int k=0; k < numACols; k++){
            int a = A[row * numACols + k];
            int b = B[k* numBCols + column];
            value += a * b;
        }
        C[row * numCCols + column] = value;
    }


}

int main(int argc, char*argv[]){

    // function declarations
    void initA(int* data, int, int);
    void initB(int* data, int, int);
    void matrixMultiplicationCPU(int*, int*, int*, int, int,int,int);
    void cleanup();

    // variable declarations
    int numARows=BLOCK_WIDTH;
    int numACols=BLOCK_WIDTH;
    int numBRows=BLOCK_WIDTH;
    int numBCols=BLOCK_WIDTH;
    int numCRows=BLOCK_WIDTH;
    int numCCols=BLOCK_WIDTH;

    int numGoldRows=BLOCK_WIDTH;
    int numGoldCols=BLOCK_WIDTH;

    int sizeA = numARows * numACols * sizeof(int); // 2-D array (matrix) is represented as 1-D array in memory.
    int sizeB = numARows * numBCols * sizeof(int); // 2-D array (matrix) is represented as 1-D array in memory.
    int sizeC = numCRows * numCCols * sizeof(int); // 2-D array (matrix) is represented as 1-D array in memory.
    int sizeGold = numGoldRows * numGoldCols * sizeof(int); // 2-D array (matrix) is represented as 1-D array in memory.

    cudaError_t result = cudaSuccess;

    // code
    // host memory allocation
    hostA = (int *) malloc(sizeA);
    if(hostA==NULL){
        printf("Host memory allocation is failed for hostA matrix.\n");
        cleanup();
        exit(EXIT_FAILURE);
    }

    hostB = (int*) malloc(sizeB);
    if(hostB==NULL){
        
        printf("Host memory allocation is failed for hostB matrix.\n");
        cleanup();
        exit(EXIT_FAILURE);
    }

    hostC = (int*) malloc(sizeC);
    if(hostC==NULL){
        
        printf("Host memory allocation is failed for hostC matrix.\n");
        cleanup();
        exit(EXIT_FAILURE);
    }

    gold = (int*)malloc(sizeGold);
    if(gold==NULL){
        
        printf("Host memory allocation is failed for gold matrix.\n");
        cleanup();
        exit(EXIT_FAILURE);
    }

    // printing matrix dimensions and sizes
    printf("The Dimensions of Matrix 'hostA' are : %d x %d \n", numARows, numACols);
    printf("The Dimensions of Matrix 'hostB' are : %d x %d \n", numBRows, numBCols);
    printf("The Dimensions of Matrix 'hostC' are : %d x %d \n", numCRows, numCCols);
    printf("The Dimensions of Matrix 'Gold' are : %d x %d \n", numGoldRows, numGoldCols);

    // fill source matrices
    initA(hostA, numARows, numACols);
    initB(hostB, numBRows, numBCols);

    // device memory allocation
    result = cudaMalloc((void**) &deviceA, sizeA);
    if(result!=cudaSuccess){
        printf("Device memory allocation is failed for deviceA matrix.\n");
        cleanup();
        exit(EXIT_FAILURE);
    }

    result = cudaMalloc((void**)&deviceB, sizeB);
    if(result!=cudaSuccess){
        printf("Device memory allocation is failed for deviceB matrix.\n");
        cleanup();
        exit(EXIT_FAILURE);
    }

    result = cudaMalloc((void**)&deviceC, sizeC);
    if(result!=cudaSuccess){
        printf("Device memory allocation is failed for deviceC matrix.\n");
        cleanup();
        exit(EXIT_FAILURE);
    }

    // copy data from host matrices into device matrices
    result = cudaMemcpy(deviceA, hostA, sizeA, cudaMemcpyHostToDevice);
    if(result!=cudaSuccess){
        printf("Host to device data copy is failed for deviceA matrix.\n");
        cleanup();
        exit(EXIT_FAILURE);
    }

    result = cudaMemcpy(deviceB, hostB, sizeB, cudaMemcpyHostToDevice);
    if(result!=cudaSuccess){
        printf("Host to device data copy is failed for deviceB matrix.\n");
        cleanup();
        exit(EXIT_FAILURE);
    }

    // CUDA kernel configuration
    dim3 dimGrid = (ceil((int)numBCols / (int)BLOCK_WIDTH), ceil((int)numARows/(int)BLOCK_WIDTH), 1);
    dim3 dimBlock = dim3(BLOCK_WIDTH, BLOCK_WIDTH, 1);

    // CUDA kernel for matrix multiplication
    
    matrixMultiplicationGPU <<<dimGrid, dimBlock>>>(deviceA, deviceB, deviceC, numARows, numACols, numBCols, numCCols);

    

    // copy data from device matrix into host matrix
    result = cudaMemcpy(hostC, deviceC, sizeC, cudaMemcpyDeviceToHost);
    if(result != cudaSuccess){
        printf("Device to Host data copy is failed for hostC matrix.\n");
        cleanup();
        exit(EXIT_FAILURE);
    }

    // matrix multiplication on host
    matrixMultiplicationCPU(hostA, hostB, gold, numARows, numACols, numBCols, numCCols);

    // comparison
    int breakValue = -1;
    bool bAccuracy = true;
    for(int i = 0; i < numCRows; i++){
        
        int val1 = gold[i];
        int val2 = hostC[i];
        if(val1 != val2){
            bAccuracy = false;
            breakValue = i;
            break;
        }
    }

    char str[128];
    if(bAccuracy == false){
        sprintf(str, "Comparison of CPU and GPU Matrix Multiplication is not accurate at array index %d", breakValue);
    }else{
        sprintf(str, "Comparison of CPU and GPU Matrix Multiplication is accurate.");
    }

    
    printf("%s\n", str);

    // cleanup
    cleanup();

    return(0);
}

void initA(int *data, int row, int col){

    int num=1;

    // code
    for(int i=0; i<row; i++){
        for(int j=0; j<col; j++){
            *(data + i * col + j) = num;
            num++;
        }
    }
}

void initB(int *data, int row, int col){

    int num = BLOCK_WIDTH;

    // code
    for(int i = 0; i < row; i++){
        for(int j = 0; j < col; j++){
            *(data + i * col + j) = num;
            num--;
        }
    }

}



void matrixMultiplicationCPU(int *A, int *B, int *C, int numARows,int numACols ,int numBCols, int numCCols){

    // code
    

    for(int i = 0 ; i < numARows; ++i){

        for(int j = 0; j < numBCols; ++j){

            int value = 1;
            for(int k = 0; k < numACols; ++k){

                int a = A[i * numACols + k];
                int b = B[k * numBCols + j];
                value += a * b;
            }
            C[i * numCCols + j] = value;
        }
    }
        
}

void cleanup(void){

    // code
    if(deviceC){
        cudaFree(deviceC);
        deviceC = NULL;
    }

    if(deviceB){
        cudaFree(deviceB);
        deviceB = NULL;

    }

    if(deviceA){
        cudaFree(deviceA);
        deviceA = NULL;
    }

    if(gold){
        free(gold);
        gold = NULL;
    }

    if(hostC){
        free(hostC);
        hostC = NULL;
    }

    if(hostB){
        free(hostB);
        hostB = NULL;
    }

    if(hostA){
        free(hostA);
        hostA = NULL;
    }
}

