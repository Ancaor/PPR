
CC = g++ 
MPICC = mpicxx
NAME = floyd
NAME_MPI = floydMPI
CFLAGS = -O3

MODULES = floyd.o Graph.o
MODULES_MPI = floydMPI.o Graph.o

all 		:	floyd floydMPI
Graph.o		:       Graph.cc Graph.h
	$(CC) $(CFLAGS) -c Graph.cc
floyd.o   :	floyd.cc
	$(CC) $(CFLAGS) -c floyd.cc
floydMPI.o	: floydMPI.cc
	$(MPICC) -c floydMPI.cc
floydMPI :		$(MODULES_MPI)
	$(MPICC) $(MODULES_MPI) -o $(NAME_MPI) 
floyd :		$(MODULES)
	$(CC) $(CFLAGS) $(MODULES) -o $(NAME) $(EXTRALIBS)
clean	:
	rm -f $(MODULES)  $(NAME)  $(NAME_MPI)  $(MODULES_MPI)
