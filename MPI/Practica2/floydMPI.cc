#include <iostream>
#include <fstream>
#include <string.h>
#include <time.h>
#include "Graph.h"
#include "mpi.h"
#include <cstdlib>
#include <cmath>

using namespace std;
 
#define TAMA 12
 
int main(int argc, char** argv) {
 
    int rank, size, nverts;
    MPI_Comm COMM_VERTICAL, COMM_HORIZONTAL;
    Graph grafo;
    MPI_Datatype MPI_BLOQUE;
 
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
 
    //** Lectura del grafo **//


    if(rank == 0){
        
        grafo.lee(argv[1]); // Lee el grafo del archivo indicado

        nverts = grafo.vertices;
    }

    // Broadcast de nverts a todos los procesos

    MPI_Bcast(&nverts, 1, MPI_INT, 0, MPI_COMM_WORLD);

    int dim = sqrt(size);
    int subFilCol = nverts/dim; // tamaño de submatriz
    int bsize = subFilCol * subFilCol;

    // creo un comunicador vertical y otro horizontal

    MPI_Comm_split(MPI_COMM_WORLD,rank/dim,rank,&COMM_VERTICAL);
    MPI_Comm_split(MPI_COMM_WORLD,rank%dim,rank,&COMM_HORIZONTAL);
    
    //** Reparto de bloques **//

    //matriz de recepcion
    int *matrizRecepcion = new int[nverts*nverts];

    //submatriz de cada bloque
    int submatriz[subFilCol][subFilCol];
    

    if(rank == 0){

        //matriz original
        int *I = grafo.getMatriz();


        //Creo el tipo de dato Bloque
        MPI_Type_vector (subFilCol,subFilCol,nverts,MPI_INT,&MPI_BLOQUE);
        MPI_Type_commit(&MPI_BLOQUE);

        //Empaqueto los bloques 

        for(int i=0, pos = 0; i < size; i++){
            int filaP = i/dim; // fila del proceso i
            int columnaP = i%dim; // columna del proceso i
            int posComienzo = (columnaP * subFilCol) + (filaP * bsize * dim);

            MPI_Pack(I+posComienzo,1,MPI_BLOQUE,matrizRecepcion,sizeof(int)*nverts*nverts,&pos,MPI_COMM_WORLD);

        }
    }

    // Reparto de bloques 

    MPI_Scatter(matrizRecepcion,sizeof(int)*bsize,MPI_PACKED,submatriz,bsize,MPI_INT,0,MPI_COMM_WORLD);

    // Barrera de sincronizacion
    MPI_Barrier(MPI_COMM_WORLD);

    double t1 = MPI_Wtime();

    /** Algoritmo de Floyd **/


    int fila_k[subFilCol];
    int columna_k[subFilCol];
    int i_global, j_global; // indices globales
    int idProcesoK; // id del proceso que contiene la fila y columna k-esima

    for(int k = 0; k < nverts; k++){
        i_global = (rank/dim)*subFilCol;
        j_global = (rank%dim)*subFilCol;

        idProcesoK = k/subFilCol;

        //compruebo si la fila y columna k estan en nuestra submatriz
        bool tiene_fila_k = (k >= i_global) && (k < i_global + subFilCol);
        bool tiene_columna_k = (k >= j_global) && (k < j_global + subFilCol);

        if(tiene_fila_k)
            memcpy(fila_k, &submatriz[k%subFilCol][0],sizeof(int)*subFilCol);

        if(tiene_columna_k)
            for(int i = 0; i < subFilCol; i++){
                columna_k[i] = submatriz[i][k%subFilCol];
            }
    

        //Broadcast de la fila y columna k-esima
        MPI_Bcast(fila_k,subFilCol,MPI_INT, idProcesoK,COMM_HORIZONTAL);
        MPI_Bcast(columna_k,subFilCol,MPI_INT, idProcesoK,COMM_VERTICAL);


        for(int i = 0; i < subFilCol; i++, i_global++){
            int j_global = (rank%dim)*subFilCol;
            for(int j = 0; j < subFilCol; j++, j_global++){
                if(i_global!=j_global && i_global!=k && j_global!=k){
                    int vikj = min(fila_k[j]+columna_k[i],submatriz[i][j]);
                    submatriz[i][j] = vikj;
                }
            }
        }

        MPI_Barrier(MPI_COMM_WORLD);

        double t2 = MPI_Wtime();


        /** Obtencion de resultados **/

        //primero realizo un gather para obtener todos los resultados en el proceso 0

        MPI_Gather(submatriz,bsize,MPI_INT,matrizRecepcion,sizeof(int)*bsize,MPI_PACKED,0,MPI_COMM_WORLD);

        int * matriz_resultado = new int[nverts*nverts];

        if(rank == 0){

            // desempaqueto 

            for(int i=0, pos = 0; i < size; i++){
                int filaP = i/dim; // fila del proceso i
                int columnaP = i%dim; // columna del proceso i
                int posComienzo = (columnaP * subFilCol) + (filaP * bsize * dim);

                MPI_Unpack(matrizRecepcion,sizeof(int)*nverts*nverts,&pos,matriz_resultado+posComienzo,1,MPI_BLOQUE,MPI_COMM_WORLD);
            }

            //Libero el tipo de dato bloque
            MPI_Type_free(&MPI_BLOQUE);

            delete matriz_resultado;

            cout <<endl <<"Tiempo de Floyd 2D con MPI : " << t2-t1 << endl;
            
        }
        MPI_Barrier(MPI_COMM_WORLD);
        delete matrizRecepcion;
        MPI_Finalize();

        return 0;
    }


}