CPP= g++
CFLAGS= -O3 -w -m64 -std=c++11 -fopenmp
EXECUTABLE=KP

kp: data.o main.o
	$(CPP) $(CFLAGS) -o $(EXECUTABLE) data.o main.cpp

data.o: ../data.cpp ../data.h
	$(CPP) $(CFLAGS) -c ../data.cpp

main.o: main.cpp ../data.h
	$(CPP) $(CFLAGS) -c main.cpp

clean:
	rm -f ${EXECUTABLE} *.o

