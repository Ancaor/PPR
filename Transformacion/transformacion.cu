#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <math.h>
#include <string.h>
#include <sys/time.h>

using namespace std;



//**************************************************************************
double cpuSecond()
{
	struct timeval tp;
	gettimeofday(&tp, NULL);
	return((double)tp.tv_sec + (double)tp.tv_usec*1e-6);
}


/************************Kernel memoria global
*
* El kernel realiza todo el computo del ejercicio propuesto.
* Se divide en 2 fases, una primera en la que se calcula el vector C usando
* memoria global y una segunda fase en la que se hace una reduccion para 
* calcular tanto D como mx, ambos usando memroria compartida.
*/


__global__ void transformacion_kernel_global(float * A, float * B, float * C, 
  float * D, float * mx)
{

int tid = threadIdx.x; // id del thread dentro del bloque
int Bsize = blockDim.x; // tamaño de bloque
int i= tid + Bsize * blockIdx.x; // identificador de thread global
float c = 0.0; // valor a calcular

extern __shared__ float sdata[]; // memoria compartida
float *sdata_A = sdata; 	   // Puntero al primer valor de A
float *sdata_B = sdata+Bsize;    // Puntero al primer valor de B
float *sdata_C = sdata+Bsize*2;  // Puntero al primer valor de C
float *sdata_C2 = sdata+Bsize*3; // Puntero al primer valor de una copia de C


// Paso a memoria compartida de A y B
*(sdata_A+tid) = A[i];
*(sdata_B+tid) = B[i];

__syncthreads();

//***************************Calculo de C *******************

int jinicio = blockIdx.x * Bsize; // inicio del bloque al que pertenece C[i]
int jfin = jinicio + Bsize; // fin del bloque al que pertenece C[i]

for (int j = jinicio; j < jfin; j++){ 
 float a = A[j] * i;
 int signo = int(ceil(a))%2 == 0 ? 1 : -1;
 c += a + B[j] * signo;
}

C[i] = c;
*(sdata_C+tid) = c;
*(sdata_C2+tid) = c;

__syncthreads();

//***************************Calculo de D y mx (Reduccion) *******************

float n, m;

for (int s=blockDim.x/2; s>0; s>>=1){
 if (tid < s){
    n = *(sdata_C2+tid);
    m = *(sdata_C2+tid+s);
    *(sdata_C+tid) += *(sdata_C+tid+s);
    *(sdata_C2+tid) = (n > m) ? n : m;
 }
 __syncthreads(); // garantiza que todos los demas threads estan en el mismo momento del calculo, sirve para dos cosas, 1) que todos los threads tengan valores correctos para reducir 2) que permita al thread 0 guardar el resultado correcto en memoria global
}

//El primer thread del bloque guarda el resultado de la reduccion en el vector de memoria global

if (tid == 0){
 D[blockIdx.x] = *(sdata_C);
 mx[blockIdx.x] = *(sdata_C2);
}

}


/************************Kernel memoria compartida
*
* El kernel es exactamente el mismo que en memoria global a excepción del bloque de cálculo
* de C en el que se utiliza Memoria compartida
*/




__global__ void transformacion_kernel_shared(float * A, float * B, float * C, 
  float * D, float * mx)
{

int tid = threadIdx.x;
int Bsize = blockDim.x;
int i= tid + Bsize * blockIdx.x;
float c = 0.0; // valor a calcular

extern __shared__ float sdata[]; // memoria compartida
float *sdata_A = sdata; 	   // Puntero al primer valor de A
float *sdata_B = sdata+Bsize;    // Puntero al primer valor de B
float *sdata_C = sdata+Bsize*2;  // Puntero al primer valor de C
float *sdata_C2 = sdata+Bsize*3; // Puntero al primer valor de una copia de C


// Paso a memoria compartida de A y B
*(sdata_A+tid) = A[i];
*(sdata_B+tid) = B[i];

__syncthreads();

//***************************Calculo de C *******************

// Se prescinde de jinicio y jfin ya que como trabajamos con memoria compartida.

for (int j = 0; j < Bsize; j++){ 
 float a = *(sdata_A+j) * i;
 int signo = int(ceil(a))%2 == 0 ? 1 : -1;
 c += a + *(sdata_B+j) * signo;
}

C[i] = c;
*(sdata_C+tid) = c;
*(sdata_C2+tid) = c;

__syncthreads();

//***************************Calculo de D y mx (Reduccion) *******************

float n, m;

for (int s=blockDim.x/2; s>0; s>>=1){
 if (tid < s){
    n = *(sdata_C2+tid);
    m = *(sdata_C2+tid+s);
    *(sdata_C+tid) += *(sdata_C+tid+s);
    *(sdata_C2+tid) = (n > m) ? n : m;
 }
 __syncthreads(); // garantiza que todos los demas threads estan en el mismo momento del calculo, sirve para dos cosas, 1) que todos los threads tengan valores correctos para reducir 2) que permita al thread 0 guardar el resultado correcto en memoria global
}

//El primer thread del bloque guarda el resultado de la reduccion en el vector de memoria global

if (tid == 0){
 D[blockIdx.x] = *(sdata_C);
 mx[blockIdx.x] = *(sdata_C2);
}

}

//**************************************************************************
int main(int argc, char *argv[])
{
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

int Bsize, NBlocks;
if (argc != 3)
  { cout << "Uso: transformacion Num_bloques Tam_bloque  "<<endl;
    return(0);
  }
else
  {NBlocks = atoi(argv[1]);
   Bsize= atoi(argv[2]);
  }

const int   N=Bsize*NBlocks;

//* pointers to host memory */
float *A, *B, *C,*D;

//* Allocate arrays a, b and c on host*/
A = new float[N];
B = new float[N];
C = new float[N];
D = new float[NBlocks];


dim3 threadsPerBlock(Bsize, 1);
dim3 numBlocks(NBlocks, 1);


float mx; // maximum of C

int sizeVectoresEntrada = N*sizeof(float);
int sizeVectoresSalida = NBlocks*sizeof(float);

// resultado kernel 

float* D_global = new float[NBlocks];
float* D_shared = new float[NBlocks];
float* mx_global = new float[NBlocks];
float* mx_shared = new float[NBlocks];

// variables device de vectores
float* d_A, *d_B, *d_C,*d_D_global, *d_mx_global, *d_D_shared, *d_mx_shared;

// reserva de espacio device
d_A = NULL;
err = cudaMalloc((void **) &d_A, sizeVectoresEntrada);
if(err != cudaSuccess) cout << "ERROR RESERVA A" << endl;

d_B = NULL;
err = cudaMalloc((void **) &d_B, sizeVectoresEntrada);
if(err != cudaSuccess) cout << "ERROR RESERVA B" << endl;

d_C = NULL;
err = cudaMalloc((void **) &d_C, sizeVectoresEntrada);
if(err != cudaSuccess) cout << "ERROR RESERVA C" << endl;

d_D_global = NULL;
err = cudaMalloc((void **) &d_D_global, sizeVectoresSalida);
if(err != cudaSuccess) cout << "ERROR RESERVA D" << endl;

d_mx_global = NULL;
err = cudaMalloc((void **) &d_mx_global, sizeVectoresSalida);
if(err != cudaSuccess) cout << "ERROR RESERVA MX" << endl;

d_D_shared = NULL;
err = cudaMalloc((void **) &d_D_shared, sizeVectoresSalida);
if(err != cudaSuccess) cout << "ERROR RESERVA D" << endl;

d_mx_shared = NULL;
err = cudaMalloc((void **) &d_mx_shared, sizeVectoresSalida);
if(err != cudaSuccess) cout << "ERROR RESERVA MX" << endl;

//* Initialize arrays A and B */
for (int i=0; i<N;i++)
  { A[i]= (float) (1  -(i%100)*0.001);
    B[i]= (float) (0.5+(i%10) *0.1  );    
  }

// Time measurement  
double t1;//=clock();

/************************* GPU Global Memory Phase ********************************/

t1 = cpuSecond();

// copy A and B to device
err = cudaMemcpy(d_A, A, sizeVectoresEntrada, cudaMemcpyHostToDevice);
if (err != cudaSuccess) cout << "ERROR COPIA A" << endl;

err = cudaMemcpy(d_B, B, sizeVectoresEntrada, cudaMemcpyHostToDevice);
if (err != cudaSuccess) cout << "ERROR COPIA B" << endl;
 
int smemSize = Bsize*4*sizeof(float); // 4 * tamaño de 1 bloque (A, B, C, C2)

transformacion_kernel_global<<<numBlocks, threadsPerBlock, smemSize>>>(
     d_A, d_B, d_C, d_D_global, d_mx_global);

cudaMemcpy(D_global, d_D_global, NBlocks*sizeof(float), cudaMemcpyDeviceToHost);
cudaMemcpy(mx_global, d_mx_global, NBlocks*sizeof(float), cudaMemcpyDeviceToHost);

float mx_global_final = mx_global[0];
cudaDeviceSynchronize();

// final reduction on CPU
for (int k = 1; k<NBlocks; k++)
   mx_global_final = (mx_global_final > mx_global[k]) ? mx_global_final : mx_global[k];

double tgpu_global=cpuSecond()-t1;



/************************* GPU Shared Memory Phase ********************************/

t1 = cpuSecond();

transformacion_kernel_shared<<<numBlocks, threadsPerBlock, smemSize>>>(
     d_A, d_B, d_C, d_D_shared, d_mx_shared);

cudaMemcpy(D_shared, d_D_shared, NBlocks*sizeof(float), cudaMemcpyDeviceToHost);
cudaMemcpy(mx_shared, d_mx_shared, NBlocks*sizeof(float), cudaMemcpyDeviceToHost);

float mx_shared_final = mx_shared[0];
cudaDeviceSynchronize();

// final reduction on CPU
for (int k = 1; k<NBlocks; k++)
mx_shared_final = (mx_shared_final > mx_shared[k]) ? mx_shared_final : mx_shared[k];

double tgpu_shared=cpuSecond()-t1;


/************************* CPU Phase ********************************/

t1 = cpuSecond();

// Compute C[i], d[K] and mx
for (int k=0; k<NBlocks;k++){
   int istart=k*Bsize;
   int iend  =istart+Bsize;
   D[k]=0.0;
   for (int i=istart; i<iend;i++){
     C[i]=0.0;
     for (int j=istart; j<iend;j++){ 
       float a=A[j]*i;
       if ((int)ceil(a) % 2 ==0)
	      C[i]+= a + B[j];
       else
 	      C[i]+= a - B[j];
     }
   D[k]+=C[i];
   mx=(i==1)?C[0]:max(C[i],mx);
  }
}

double tcpu = cpuSecond()-t1;



//Imprimir datos por pantalla
cout << endl;
cout << "Tiempo de cpu : " << tcpu << endl;
cout << "Tiempo de gpu (memoria global) : " << tgpu_global << endl;
cout << "Tiempo de gpu (memoria compartida) : " << tgpu_shared << endl << endl;

cout << "Ganancia de gpu (memoria global) : " << tcpu / tgpu_global << endl;
cout << "Ganancia de gpu (memoria compartida) : " << tcpu / tgpu_shared << endl<<endl<<endl;


cout << "Valor máximo de C (cpu) : " << mx << endl;
cout << "Valor máximo de C (gpu memoria global) : " << mx_global_final << endl;
cout << "Valor máximo de C (gpu memoria compartida) : " << mx_shared_final << endl;



//* Free the memory */
delete(A); 
delete(B); 
delete(C);
delete(D);

}