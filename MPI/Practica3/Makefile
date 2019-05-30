
./bin/bbseq: ./obj/bbseq.o ./obj/libbb.o
	mpicxx -O3 ./obj/bbseq.o ./obj/libbb.o -o ./bin/bbseq
	
./obj/bbseq.o: ./src/bbseq.cc
	mpicxx -O3 -c ./src/bbseq.cc -o ./obj/bbseq.o


./obj/libbb.o: ./lib/libbb.cc ./lib/libbb.h 
	mpicxx -O3 -c  ./lib/libbb.cc -o ./obj/libbb.o


clean:
	/bin/rm -f ./obj/*.o ./bin/bbseq



