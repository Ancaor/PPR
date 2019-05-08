#include <iostream>
#include <fstream>
#include <string.h>
#include <time.h>
#include "../include/Graph.h"
#include "mpi.h"
#include <cstdlib>
#include <cmath>

using namespace std;

//**************************************************************************

int main (int argc, char *argv[])
{

    int rank, size;
    MPI_Datatype MPI_BLOQUE;
    MPI_Comm COMM_HORIZONTAL, COMM_VERTICAL;

    MPI_Init(&argc, &argv); // Inicializamos la comunicacion de los procesos
    MPI_Comm_size(MPI_COMM_WORLD, &size); // Obtenemos el numero total de hebras
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); // Obtenemos el valor de nuestro identificador


    if (argc != 2) {
        cerr << "LLamada correcta: mpirun -np <nº hebras> " << argv[0] << " <archivo de grafo>"
        << endl;
        return(-1);
    }


    /****************************** Lectura del grafo *****************************/

    int nverts;
    Graph G;

    if (rank == 0){
        G.lee(argv[1]);      
        nverts=G.vertices;
    }

    // Broadcast de nverts a todos los procesos
    MPI_Bcast(&nverts, 1, MPI_INT, 0, MPI_COMM_WORLD);

    const int dim_submatriz = nverts/sqrt(size), // Filas, columnas de las submatrices
        bsize = dim_submatriz*dim_submatriz, // Tamaño de bloque
        grid_dim= sqrt(size); // Dimension del grid
                       

    // Comunicadores horizontal y vertical
    MPI_Comm_split(MPI_COMM_WORLD, rank/grid_dim, rank, &COMM_VERTICAL);
    MPI_Comm_split(MPI_COMM_WORLD, rank%grid_dim, rank, &COMM_HORIZONTAL);

    int i, j, vikj, pos, fila_P, columna_P, comienzo;


    /****************************** Reparto de bloques *****************************/

    int submatriz[dim_submatriz][dim_submatriz];
    int *matriz_grafo = new int[nverts*nverts];

    if (rank == 0){

    int * matrizOriginal = G.getMatriz(); // Matriz original que representa el grafo
    
    // Defino y creo el tipo bloque cuadrado
    MPI_Type_vector(dim_submatriz,dim_submatriz,nverts,MPI_INT,&MPI_BLOQUE);
    MPI_Type_commit (&MPI_BLOQUE);

    //Empaquetado del bloque
    for (i=0, pos=0; i<size; i++) {
        // Calculo la posicion de comienzo de cada submatriz
        fila_P = i/grid_dim;
        columna_P = i%grid_dim;
        comienzo = columna_P * dim_submatriz + fila_P * bsize * grid_dim;

        MPI_Pack (matrizOriginal + comienzo,1,MPI_BLOQUE,matriz_grafo,sizeof(int)*nverts*nverts,&pos,MPI_COMM_WORLD);
    }

    }

    // Distribuimos la matriz entre los procesos
    MPI_Scatter(matriz_grafo,sizeof(int)*bsize,MPI_PACKED,submatriz,bsize,MPI_INT,0,MPI_COMM_WORLD);

    // Sincronizamos los procesos para medir tiempos
    MPI_Barrier(MPI_COMM_WORLD);

    double t1=MPI_Wtime();




    /****************************** Algoritmo de Floyd *****************************/

    int idProcesoK;
    int filaK[dim_submatriz]; 
    int columnaK[dim_submatriz];

    // indices globales de filas y columnas
    int i_global;
    int j_global;

    for (int k=0; k < nverts; k++)
    {
        i_global = (rank/grid_dim)*dim_submatriz;
        j_global = (rank%grid_dim)*dim_submatriz;

        // proceso que tiene la fila y columnaK
        idProcesoK = k/dim_submatriz;

        // Comprobamos si la fila y columna k-ésima están en nuestra submatriz
        bool tienesFilaK = k >= i_global && k < i_global+dim_submatriz;
        bool tienesColumnaK = k >= j_global && k < j_global+dim_submatriz;

        if (tienesFilaK)
            memcpy(filaK, &submatriz[k%dim_submatriz][0], sizeof(int)*dim_submatriz);

        if (tienesColumnaK)
            for (i = 0; i < dim_submatriz; i++)
                columnaK[i] = submatriz[i][k%dim_submatriz];

        // Broadcast de la fila y columna k-ésima
        MPI_Bcast(filaK, dim_submatriz, MPI_INT, idProcesoK, COMM_HORIZONTAL);
        MPI_Bcast(columnaK, dim_submatriz, MPI_INT, idProcesoK, COMM_VERTICAL);

        i_global = (rank/grid_dim)*dim_submatriz;

        for (i = 0; i < dim_submatriz; i++, i_global++){
            j_global = (rank%grid_dim)*dim_submatriz;
            for (j = 0; j < dim_submatriz; j++, j_global++)
                if (i_global!=j_global && i_global!=k && j_global!=k){
                    vikj=min(filaK[j]+columnaK[i], submatriz[i][j]);
                    submatriz[i][j]=vikj;
                }
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);

    double  t2=MPI_Wtime();
    t2 = t2-t1;




    /****************************** Obtencion de resultados *****************************/

    MPI_Gather(submatriz,bsize,MPI_INT,matriz_grafo,sizeof(int)*bsize,MPI_PACKED,0,MPI_COMM_WORLD);

    int * matrizResultado = new int[nverts*nverts];

    if (rank == 0){

    for (i=0, pos=0; i<size; i++) {
        // Calculo la posicion de comienzo de cada submatriz de cada proceso
        fila_P = i/grid_dim;
        columna_P = i%grid_dim;
        comienzo = columna_P * dim_submatriz + fila_P * bsize * grid_dim;

        MPI_Unpack(matriz_grafo,sizeof(int)*nverts*nverts,&pos,matrizResultado + comienzo,1,MPI_BLOQUE,MPI_COMM_WORLD);
    }

    MPI_Type_free (&MPI_BLOQUE);

    cout <<"MPI-2D - Input: "<< argv[1] <<",  Tiempo MPI-2D: " << t2 << endl;

/*
    cout << endl<<"EL Grafo con las distancias de los caminos más cortos es:\n";
    for(i=0;i<nverts;i++){
        cout << "A["<<i << ",*]= ";
        for(j=0;j<nverts;j++){
        if (matrizResultado[i*nverts+j]==INF) cout << "INF";
        else cout << matrizResultado[i*nverts+j];
        if (j<nverts-1) cout << ",";
        else cout << endl;
        }
    }
*/
    delete matrizResultado;
    }

    MPI_Barrier(MPI_COMM_WORLD);
    delete matriz_grafo;
    MPI_Finalize();

    return 0;
}