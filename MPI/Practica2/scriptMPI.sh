#!/bin/bash

ARCHIVO=salidaScript.output

echo "Secuencial:" > $ARCHIVO

./bin/floyd ./input/input60 >> $ARCHIVO
./bin/floyd ./input/input240 >> $ARCHIVO
./bin/floyd ./input/input750 >> $ARCHIVO
./bin/floyd ./input/input1200 >> $ARCHIVO

echo "MPI para 4 procesadores:" >> $ARCHIVO

mpirun -np 4 ./bin/floydMPI ./input/input60 >> $ARCHIVO
mpirun -np 4 ./bin/floydMPI ./input/input240 >> $ARCHIVO
mpirun -np 4 ./bin/floydMPI ./input/input750 >> $ARCHIVO
mpirun -np 4 ./bin/floydMPI ./input/input1200 >> $ARCHIVO

echo "MPI para 9 procesadores:" >> $ARCHIVO

mpirun -np 9 ./bin/floydMPI ./input/input60 >> $ARCHIVO
mpirun -np 9 ./bin/floydMPI ./input/input240 >> $ARCHIVO
mpirun -np 9 ./bin/floydMPI ./input/input750 >> $ARCHIVO
mpirun -np 9 ./bin/floydMPI ./input/input1200 >> $ARCHIVO

echo "OMP para 4 threads:" >> $ARCHIVO

./bin/floydOMP1D ./input/input60 4 >> $ARCHIVO
./bin/floydOMP1D ./input/input240 4 >> $ARCHIVO
./bin/floydOMP1D ./input/input750 4 >> $ARCHIVO
./bin/floydOMP1D ./input/input1200 4 >> $ARCHIVO

echo "OMP para 9 threads:" >> $ARCHIVO

./bin/floydOMP1D ./input/input60 9 >> $ARCHIVO
./bin/floydOMP1D ./input/input240 9 >> $ARCHIVO
./bin/floydOMP1D ./input/input750 9 >> $ARCHIVO
./bin/floydOMP1D ./input/input1200 9 >> $ARCHIVO

