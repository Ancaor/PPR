#include <iostream>
#include <fstream>
#include <string.h>
#include <sys/time.h>

using namespace std;

const int N=2000;

const int THREADS_PER_BLOCK_1D = 1024;

__global__ void MatAdd (float *A, float *B, float * C, int N)
 {
 int j = blockIdx.x * blockDim.x + threadIdx.x;
 int i = blockIdx.y * blockDim.y + threadIdx.y;
 int index=i*N+j;
 if (i < N && j < N)
 C[index] = A[index] + B[index];
 }

__global__ void MatAdd_Filas( float *A, float *B, float *C, int N)
{
int i = blockIdx.x * blockDim.x + threadIdx.x;  // Compute column index
if(i < N){
  int row=i*N; // Compute global 1D index
  for(int index = row; index < row+N; index++){
	  C[index] = A[index] + B[index]; // Compute C element
  }
}
}

__global__ void MatAdd_Columnas( float *A, float *B, float *C, int N)
{
int j = blockIdx.x * blockDim.x + threadIdx.x;  // Indica la columna que computará la hebra
if(j < N){
  for(int i=0; i < N; i++){  // para cada fila
    int index = i*N + j;
    
    C[index] = A[index] + B[index]; // Compute C element
  }
}
}


//**************************************************************************
double cpuSecond()
{
	struct timeval tp;
	gettimeofday(&tp, NULL);
	return((double)tp.tv_sec + (double)tp.tv_usec*1e-6);
}

//**************************************************************************

int main()
{
int i;
const int NN=N*N;
/* pointers to host memory */
/* Allocate arrays A, B and C on host*/
float * A = (float*) malloc(NN*sizeof(float));
float * B = (float*) malloc(NN*sizeof(float));
float * C_original = (float*) malloc(NN*sizeof(float));
float * C_row = (float*) malloc(NN*sizeof(float));
float * C_column = (float*) malloc(NN*sizeof(float));


/* pointers to device memory */
float *A_d, *B_d, *C_d_original, *C_d_row, *C_d_column;
/* Allocate arrays a_d, b_d and c_d on device*/
cudaMalloc ((void **) &A_d, sizeof(float)*NN);
cudaMalloc ((void **) &B_d, sizeof(float)*NN);
cudaMalloc ((void **) &C_d_original, sizeof(float)*NN);
cudaMalloc ((void **) &C_d_row, sizeof(float)*NN);
cudaMalloc ((void **) &C_d_column, sizeof(float)*NN);

/* Initialize arrays a and b */
for (i=0; i<NN;i++)
{
  A[i]= (float) i;
  B[i]= (float) i;
}

/* Copy data from host memory to device memory */
cudaMemcpy(A_d, A, sizeof(float)*NN, cudaMemcpyHostToDevice);
cudaMemcpy(B_d, B, sizeof(float)*NN, cudaMemcpyHostToDevice);

/* Compute the execution configuration */
dim3 threadsPerBlock (sqrt(THREADS_PER_BLOCK_1D), sqrt(THREADS_PER_BLOCK_1D));
dim3 numBlocks( ceil ((float)(N)/threadsPerBlock.x), ceil ((float)(N)/threadsPerBlock.y) );

//*********************************Original Kernel****************************

double t1 = cpuSecond();

MatAdd <<<numBlocks, threadsPerBlock>>> (A_d, B_d, C_d_original, N);

double tKernelOriginal = cpuSecond() - t1; 
/* Copy data from deveice memory to host memory */
cudaMemcpy(C_original, C_d_original, sizeof(float)*NN, cudaMemcpyDeviceToHost);


//*********************************Row Kernel****************************

t1 = cpuSecond();

/* Compute the execution configuration */

int numBlocks_1D =  ceil ((float)(N)/THREADS_PER_BLOCK_1D);

MatAdd_Filas <<<numBlocks_1D, THREADS_PER_BLOCK_1D>>> (A_d, B_d, C_d_row, N);

double tKernelRow = cpuSecond() - t1; 
/* Copy data from deveice memory to host memory */
cudaMemcpy(C_row, C_d_row, sizeof(float)*NN, cudaMemcpyDeviceToHost);


//*********************************Column Kernel****************************

t1 = cpuSecond();

MatAdd_Columnas <<<numBlocks_1D, THREADS_PER_BLOCK_1D>>> (A_d, B_d, C_d_column, N);

double tKernelColumn = cpuSecond() - t1; 

/* Copy data from deveice memory to host memory */
cudaMemcpy(C_column, C_d_column, sizeof(float)*NN, cudaMemcpyDeviceToHost);

//*********************************** Print Results ******************************************

cout << endl;

cout << "N : " << N << endl;
int tamaBloque = threadsPerBlock.x * threadsPerBlock.y;
cout << "Tamaño de bloque : " <<  tamaBloque << endl;
cout << "Tiempo de kernel original : " << tKernelOriginal << endl;
cout << "Tiempo de kernel por filas : " << tKernelRow << endl;
cout << "Tiempo de kernel por columnas : " << tKernelColumn << endl<<endl;

cout << "Ganancia de kernel por filas : " << tKernelOriginal / tKernelRow << endl;
cout << "Ganancia de kernel por columnas : " << tKernelOriginal / tKernelColumn << endl<<endl<<endl;


/* Print c */
//for (i=0; i<NN;i++)
  //printf(" c[%d]=%f\n",i,C_column[i]);

/* Free the memory */
free(A); free(B); free(C_original); free(C_row); free(C_column);
cudaFree(A_d); cudaFree(B_d);cudaFree(C_d_original);cudaFree(C_d_row);cudaFree(C_d_column);

}
