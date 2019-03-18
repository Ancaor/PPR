#include <iostream>
#include <fstream>
#include <string.h>
#include <sys/time.h>
#include "Graph.h"

// CUDA runtime
//#include <cuda_runtime.h>

// helper functions and utilities to work with CUDA
//#include <helper_functions.h>
//#include <helper_cuda.h>

#define blocksize 64

#define blocksize2D 8

#define blocksizeReduction 256

using namespace std;

//**************************************************************************
double cpuSecond()
{
	struct timeval tp;
	gettimeofday(&tp, NULL);
	return((double)tp.tv_sec + (double)tp.tv_usec*1e-6);
}

//**************************************************************************
__global__ void floyd_kernel(int * M, const int nverts, const int k) {
	int ij = threadIdx.x + blockDim.x * blockIdx.x;
  if (ij < nverts * nverts) {
		int Mij = M[ij];
    int i= ij / nverts;
    int j= ij - i * nverts;
    if (i != j && i != k && j != k) {
			int Mikj = M[i * nverts + k] + M[k * nverts + j];
    	Mij = (Mij > Mikj) ? Mikj : Mij;
    	M[ij] = Mij;
		}
  }
}

__global__ void floyd_2Dkernel(int * M, const int nverts, const int k) {
	
	int i = threadIdx.y + blockDim.y * blockIdx.y;
	int j = threadIdx.x + blockDim.x * blockIdx.x;

  if (i < nverts && j < nverts) {

	int ij = i*nverts + j;
	int Mij = M[ij];
    int i= ij / nverts;
    int j= ij - i * nverts;
    if (i != j && i != k && j != k) {
			int Mikj = M[i * nverts + k] + M[k * nverts + j];
    	Mij = (Mij > Mikj) ? Mikj : Mij;
    	M[ij] = Mij;
		}
  }
}

__global__ void reduceMin(int * V_in, int * V_out, const int N) {
	extern __shared__ int sdata[];

	int tid = threadIdx.x;
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	sdata[tid] = ((i < N) ? V_in[i] : -10000000);
	__syncthreads();

	for(int s = blockDim.x/2; s > 0; s >>= 1){
	  if (tid < s) 
		if(sdata[tid] < sdata[tid+s]) 
                    sdata[tid] = sdata[tid+s];		
	  __syncthreads();
	}
	if (tid == 0) 
           V_out[blockIdx.x] = sdata[0];
}


/***************************** main ****************************/



int main (int argc, char *argv[]) {

	if (argc != 2) {
		cerr << "Sintaxis: " << argv[0] << " <archivo de grafo>" << endl;
		return(-1);
	}

	// This will pick the best possible CUDA capable device
	// int devID = findCudaDevice(argc, (const char **)argv);

  //Get GPU information
  int devID;
  cudaDeviceProp props;
  cudaError_t err;
  err = cudaGetDevice(&devID);
  if(err != cudaSuccess) {
		cout << "ERRORRR" << endl;
	}

  cudaGetDeviceProperties(&props, devID);
  printf("Device %d: \"%s\" with Compute %d.%d capability\n", devID, props.name, props.major, props.minor);

	Graph G;
	G.lee(argv[1]);// Read the Graph

	//cout << "EL Grafo de entrada es:"<<endl;
	//G.imprime();
	const int nverts = G.vertices;
	const int niters = nverts;

	const int nverts2 = nverts * nverts;

	int *c_Out_M = new int[nverts2];
	int size = nverts2*sizeof(int);
	int * d_In_M = NULL;

	int *c_Out_M_2D = new int[nverts2];
	int * d_In_M_2D = NULL;

	

	err = cudaMalloc((void **) &d_In_M, size);
	if (err != cudaSuccess) {
		cout << "ERROR RESERVA" << endl;
	}

	err = cudaMalloc((void **) &d_In_M_2D, size);
	if (err != cudaSuccess) {
		cout << "ERROR RESERVA" << endl;
	}


	int *A = G.Get_Matrix();



	/******************************* GPU phase 1D ***************************/
	double  t1 = cpuSecond();

	err = cudaMemcpy(d_In_M, A, size, cudaMemcpyHostToDevice);
	if (err != cudaSuccess) {
		cout << "ERROR COPIA A GPU" << endl;
	}

	for(int k = 0; k < niters; k++) {
		//printf("CUDA kernel launch \n");
	 	int threadsPerBlock = blocksize;
	 	int blocksPerGrid = (nverts2 + threadsPerBlock - 1) / threadsPerBlock;

	  floyd_kernel<<<blocksPerGrid,threadsPerBlock >>>(d_In_M, nverts, k);
	  err = cudaGetLastError();

	  if (err != cudaSuccess) {
	  	fprintf(stderr, "Failed to launch kernel!\n");
	  	exit(EXIT_FAILURE);
		}
	}

	cudaMemcpy(c_Out_M, d_In_M, size, cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();
	double Tgpu = cpuSecond()-t1;

	cout << "Tiempo gastado GPU 1D = " << Tgpu << endl << endl;



	/******************************* GPU phase 2D ***************************/

	 t1 = cpuSecond();

	err = cudaMemcpy(d_In_M_2D, A, size, cudaMemcpyHostToDevice);
	if (err != cudaSuccess) {
		cout << "ERROR COPIA A GPU" << endl;
	}

	dim3 threadsPerBlock2D(blocksize2D,blocksize2D);

	dim3 blocksPerGrid2D( ceil( (float) nverts / blocksize2D), ceil( (float) nverts / blocksize2D));

	for(int k = 0; k < niters; k++) {
		//printf("CUDA kernel launch \n");
	 	

	  floyd_kernel<<<blocksPerGrid2D,threadsPerBlock2D>>>(d_In_M_2D, nverts, k);
	  err = cudaGetLastError();

	  if (err != cudaSuccess) {
	  	fprintf(stderr, "Failed to launch kernel!\n");
	  	exit(EXIT_FAILURE);
		}
	}

	cudaMemcpy(c_Out_M_2D, d_In_M_2D, size, cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();
	
	double Tgpu_2D = cpuSecond()-t1;

	cout << "Tiempo gastado GPU 2D = " << Tgpu_2D << endl << endl;

	/******************************* CPU phase ***************************/

	t1 = cpuSecond();

	// BUCLE PPAL DEL ALGORITMO
	int inj, in, kn;
	for(int k = 0; k < niters; k++) {
          kn = k * nverts;
	  for(int i=0;i<nverts;i++) {
			in = i * nverts;
			for(int j = 0; j < nverts; j++)
	       			if (i!=j && i!=k && j!=k){
			 	    inj = in + j;
			 	    A[inj] = min(A[in+k] + A[kn+j], A[inj]);
	       }
	   }
	}

  double t2 = cpuSecond() - t1;
  cout << "Tiempo gastado CPU= " << t2 << endl << endl;
  cout << "Ganancia 1D= " << t2 / Tgpu << endl;
  cout << "Ganancia 2D= " << t2 / Tgpu_2D << endl;

  for(int i = 0; i < nverts; i++)
    for(int j = 0;j < nverts; j++)
       if (abs(c_Out_M[i*nverts+j] - G.arista(i,j)) > 0)
		 cout << "Error (" << i << "," << j << ")   " << c_Out_M[i*nverts+j] << "..." << G.arista(i,j) << endl;

  /******************************* Reduction phase ***************************/

  

	dim3 threadsPerBlockReduce(blocksizeReduction,1);
	dim3 numBlocks ( ceil((float(nverts2)/(float)threadsPerBlockReduce.x)),1 );

	int * vmax;
	vmax = (int*) malloc(numBlocks.x*sizeof(int));

	int *vmax_d;
	cudaMalloc ((int **) &vmax_d, sizeof(int)*numBlocks.x);

	int smemSize = threadsPerBlockReduce.x * sizeof(float);

	reduceMin<<<numBlocks, threadsPerBlockReduce, smemSize>>>(d_In_M,vmax_d, nverts2);

	cudaMemcpy(vmax, vmax_d, numBlocks.x*sizeof(int),cudaMemcpyDeviceToHost);

	int max_gpu = -10000000;
	for (int i=0; i<numBlocks.x; i++) 
	{
	max_gpu =max(max_gpu,vmax[i]); //printf(" vmin[%d]=%d\n",i,vmax[i]);
	}

	int max_cpu = -10000000;
	for (int i=0; i<nverts2;i++){
		max_cpu=max(max_cpu, c_Out_M[i]);
		}

	cout << endl << "Camino máximo GPU: " <<max_gpu<< endl;
	cout << endl << "Camino máximo CPU: " <<max_cpu<< endl;


}

	

