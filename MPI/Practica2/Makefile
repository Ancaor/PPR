
CC = g++ 
MPICC = mpicxx
NAME = ./bin/floyd
NAME_MPI = ./bin/floydMPI
CFLAGS = -O3

MODULES = ./obj/floyd.o ./obj/Graph.o
MODULES_MPI = ./obj/floydMPI.o ./obj/Graph.o

all 		:	./bin/floyd ./bin/floydMPI
./obj/Graph.o		:       ./src/Graph.cc ./include/Graph.h
	$(CC) $(CFLAGS) -c ./src/Graph.cc -o ./obj/Graph.o
./obj/floyd.o   :	./src/floyd.cc
	$(CC) $(CFLAGS) -c ./src/floyd.cc -o ./obj/floyd.o 
./obj/floydMPI.o	: ./src/floydMPI.cc
	$(MPICC) -c ./src/floydMPI.cc -o ./obj/floydMPI.o
./bin/floydMPI :		$(MODULES_MPI)
	$(MPICC) $(MODULES_MPI) -o $(NAME_MPI) 
./bin/floyd :		$(MODULES)
	$(CC) $(CFLAGS) $(MODULES) -o $(NAME) $(EXTRALIBS)
clean	:
	rm -f $(MODULES)  $(NAME)  $(NAME_MPI)  $(MODULES_MPI)
