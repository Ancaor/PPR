#include <iostream>
#include <fstream>
#include <string.h>
#include <time.h>
#include "../include/Graph.h"
#include <omp.h>
#include <cstdlib>
#include <cmath>



using namespace std;

//**************************************************************************

int main (int argc, char *argv[])
{   

if (argc != 3) 
	{
	 cerr << "Sintaxis: " << argv[0] << " <archivo de grafo>" << " <num de hebras>" <<endl;
	return(-1);
	}

int threads = atoi(argv[2]);
omp_set_dynamic(0);
omp_set_num_threads(threads);
Graph G;
G.lee(argv[1]);		// Read the Graph
//cout << "EL Grafo de entrada es:"<<endl;
//G.imprime();

int nverts=G.vertices;
const int dim_submatriz = nverts/sqrt(threads), // Filas, columnas de las submatrices
        bsize = dim_submatriz*dim_submatriz, // Tamaño de bloque
        grid_dim= sqrt(threads); // Dimension del grid
int submatriz[dim_submatriz][dim_submatriz];
int filaKprivada[dim_submatriz]; 
int columnaKprivada[dim_submatriz];
int columnaK[nverts];
int filaK[nverts];
int k,j,vikj;

double  t1=omp_get_wtime();

#pragma omp parallel private(submatriz, filaKprivada, columnaKprivada) // Comienza la seccion paralela y establezco las variables que seran privadas a cada thread
{

	int tID = omp_get_thread_num();
	int i,j,vikj;

	int iGlobal = (tID/grid_dim)*dim_submatriz;  //i global de la matriz
	int jGlobal;								 //j global de la matriz


	// Copia privada de la submatriz correspondiente a cada hebra
	for (i = 0; i < dim_submatriz; i++, iGlobal++){
    	jGlobal = (tID%grid_dim)*dim_submatriz;
    	for (j = 0; j < dim_submatriz; j++, jGlobal++)
        	submatriz[i][j] = G.arista(iGlobal,jGlobal);
	}

	
	for(int k=0;k<nverts;k++){

		iGlobal = (tID/grid_dim)*dim_submatriz;
		jGlobal = (tID%grid_dim)*dim_submatriz;

		// Comprueba si la fila y columna k-ésima está en su submatriz
        bool tienesFilaK = k >= iGlobal && k < iGlobal+dim_submatriz;
        bool tienesColumnaK = k >= jGlobal && k < jGlobal+dim_submatriz;

		// Si tiene la fila y/o la culumna K la copia a memoria compartida
		if (tienesFilaK)
            memcpy(filaK, &submatriz[k%dim_submatriz][0], sizeof(int)*dim_submatriz);

        if (tienesColumnaK)
            for (i = 0; i < dim_submatriz; i++)
                columnaK[i] = submatriz[i][k%dim_submatriz];

		#pragma omp barrier // Bloqueamos todos los threads hasta que la fila y columna K este copiada a memoria compartida

		iGlobal = (tID/grid_dim)*dim_submatriz;
		jGlobal = (tID%grid_dim)*dim_submatriz;

		// Privatizo la fila y columna K

		for(i = 0; i < dim_submatriz; i++, jGlobal++){
			filaKprivada[i] = filaK[jGlobal];  
		}

		for(i = 0; i < dim_submatriz; i++, iGlobal++){
			columnaKprivada[i] = columnaK[iGlobal];  
		}

		iGlobal = (tID/grid_dim)*dim_submatriz;

		//Comienzo de algoritmo de Floyd

		for (i = 0; i < dim_submatriz; i++, iGlobal++){
            jGlobal = (tID%grid_dim)*dim_submatriz;
            for (j = 0; j < dim_submatriz; j++, jGlobal++)
                if (iGlobal!=jGlobal && iGlobal!=k && jGlobal!=k){
                    vikj=min(filaKprivada[j]+columnaKprivada[i], submatriz[i][j]);
                    submatriz[i][j]=vikj;
                }
        }

		#pragma omp barrier // Sincronizo hebras antes de cambiar de K
	}

	//Construccion del grafo a partir de los threads
	iGlobal = (tID/grid_dim)*dim_submatriz;

	for (i = 0; i < dim_submatriz; i++, iGlobal++){
        jGlobal = (tID%grid_dim)*dim_submatriz;
        for (j = 0; j < dim_submatriz; j++, jGlobal++)
			G.inserta_arista(iGlobal,jGlobal,submatriz[i][j]);
	}

	#pragma omp barrier // Sincronizo hebras antes de tomar tiempo

}
 double t2=omp_get_wtime();
 t2=t2-t1;
// cout << endl<<"EL Grafo con las distancias de los caminos más cortos es:"<<endl<<endl;
 //G.imprime();
 cout <<"OMP 2D - Input: "<< argv[1] <<",  Tiempo OMP 2D: " << t2<< endl;

}



