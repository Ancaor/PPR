#include <math.h>
#include "mpi.h"
#include <cstdlib> // Incluido para el uso de atoi
#include <iostream>
using namespace std;
 
int main(int argc, char *argv[]) 
{ 
 
	// Calculo de PI

    int n,rank,size;
	
	double PI25DT = 3.141592653589793238462643;
	double h;
	double sum;
    double mypi, pi;

    MPI_Init(&argc, &argv); // Inicializamos los procesos
    MPI_Comm_size(MPI_COMM_WORLD, &size); // Obtenemos el numero total de procesos
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); // Obtenemos el valor de nuestro identificador

    if(rank == 0){
        cout<<"introduce la precision del calculo (n > 0): ";
	    cin>>n;
    }

    MPI_Bcast(&n,1,MPI_INT,0,MPI_COMM_WORLD);

    if(n < 0){
        MPI_Finalize();
        exit(0);
    }else{
        h= 1.0 / (double) n;
        sum = 0.0;

        for (int i = rank + 1; i <= n; i+=size) {
            double x = h * ((double)i - 0.5);
            sum += (4.0 / (1.0 + x*x));
	    }
	    mypi = sum * h;


        MPI_Reduce(&mypi,&pi,1,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD);

        if(rank == 0){
            cout << "Valor aproximado de Pi: "<< pi<< endl;
            cout << "Error: " << fabs(pi-PI25DT) << endl;
        }

    }

    MPI_Finalize();
	return 0;
 
}