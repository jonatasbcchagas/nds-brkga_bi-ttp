# macros
CPP= g++
CFLAGS= -O3 -w -m64 -std=c++11 -fopenmp
EXECUTABLE=BITTP

# targets
all: data.o ndsbrkga.o main.o
	$(CPP) $(CFLAGS) -o $(EXECUTABLE) data.o ndsbrkga.o main.o

data.o: data.cpp data.h
	$(CPP) $(CFLAGS) -c data.cpp

ndsbrkga.o: ndsbrkga.cpp ndsbrkga.h data.h
	$(CPP) $(CFLAGS) -c ndsbrkga.cpp

main.o: main.cpp data.h ndsbrkga.h
	$(CPP) $(CFLAGS) -c main.cpp


# remove
clean:
	rm -f ${EXECUTABLE} *.o
