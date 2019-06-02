
all : ./bin/bbseq ./bin/bbpar


./bin/bbpar: ./obj/bbpar.o ./obj/libbb.o
	mpicxx -O3 ./obj/bbpar.o ./obj/libbb.o -o ./bin/bbpar

./bin/bbseq: ./obj/bbseq.o ./obj/libbb.o
	mpicxx -O3 ./obj/bbseq.o ./obj/libbb.o -o ./bin/bbseq
	
./obj/bbseq.o: ./src/bbseq.cc
	mpicxx -O3 -c ./src/bbseq.cc -o ./obj/bbseq.o

./obj/bbpar.o: ./src/bbpar.cc
	mpicxx -O3 -c ./src/bbpar.cc -o ./obj/bbpar.o

./obj/libbb.o: ./lib/libbb.cc ./lib/libbb.h 
	mpicxx -O3 -c  ./lib/libbb.cc -o ./obj/libbb.o


clean:
	/bin/rm -f ./obj/*.o ./bin/bbseq ./bin/bbpar



