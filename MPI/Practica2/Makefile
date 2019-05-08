
CC = g++ 
MPICC = mpicxx
NAME = ./bin/floyd
NAME_MPI = ./bin/floydMPI
CFLAGS = -O3

MODULES = ./obj/floyd.o ./obj/Graph.o
MODULES_MPI = ./obj/floydMPI.o ./obj/Graph.o
MODULES_OMP = ./obj/floydOMP1D.o ./obj/Graph.o

all 		:	./bin/floyd ./bin/floydMPI ./bin/floydOMP1D

./obj/Graph.o		:       ./src/Graph.cc ./include/Graph.h
	$(CC) $(CFLAGS) -c ./src/Graph.cc -o ./obj/Graph.o

./obj/floyd.o   :	./src/floyd.cc
	$(CC) $(CFLAGS) -c ./src/floyd.cc -o ./obj/floyd.o 

./obj/floydMPI.o	: ./src/floydMPI.cc
	$(MPICC) -c ./src/floydMPI.cc -o ./obj/floydMPI.o

./obj/floydOMP1D.o	:	./src/floydOMP1D.cc
	$(CC) $(CFLAGS) -c ./src/floydOMP1D.cc -o ./obj/floydOMP1D.o -fopenmp

./bin/floydMPI :		$(MODULES_MPI)
	$(MPICC) $(MODULES_MPI) -o $(NAME_MPI) 

./bin/floydOMP1D :		$(MODULES_OMP)
	$(CC) $(CFLAGS) $(MODULES_OMP) -o ./bin/floydOMP1D -fopenmp

./bin/floyd :		$(MODULES)
	$(CC) $(CFLAGS) $(MODULES) -o $(NAME) $(EXTRALIBS)

clean	:
	rm -f $(MODULES)  $(NAME)  $(NAME_MPI)  $(MODULES_MPI)
