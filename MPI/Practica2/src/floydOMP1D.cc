#include <iostream>
#include <fstream>
#include <string.h>
#include <time.h>
#include "../include/Graph.h"
#include <omp.h>


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
omp_set_num_threads(threads);
Graph G;
G.lee(argv[1]);		// Read the Graph
//cout << "EL Grafo de entrada es:"<<endl;
//G.imprime();

int nverts=G.vertices;
int filaK[nverts];
int k,j,vikj;
double  t1=omp_get_wtime();

#pragma omp parallel private(filaK, k, j, vikj) // establezco las variables que seran privadas a cada thread
{

	for(int k=0;k<nverts;k++){

		for(j = 0; j < nverts; j++){
			filaK[j] = G.arista(k,j);  // obtengo la fila K-esima necesaria
		}

		#pragma omp for schedule(static)
		for(int i=0;i<nverts;i++)
			for(int j=0;j<nverts;j++)	 
				if (i!=j && i!=k && j!=k)    
		{      
				int vikj=min(G.arista(i,k)+filaK[j],G.arista(i,j));    
				G.inserta_arista(i,j,vikj);   	      
		}
	}    
}
 double t2=omp_get_wtime();
 t2=t2-t1;
// cout << endl<<"EL Grafo con las distancias de los caminos más cortos es:"<<endl<<endl;
 //G.imprime();
 cout <<"OMP 1D - Input: "<< argv[1] <<",  Tiempo OMP 1D: " << t2<< endl;

}



