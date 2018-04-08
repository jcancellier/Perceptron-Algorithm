CC=g++
CFLAGS=-Wall -std=c++11
RM=rm -f
TESTFILE=iris-binary-normalized.csv

perceptron.out: perceptron.o
	$(CC) $(CFLAGS) -o perceptron.out perceptron.o

perceptron.o: perceptron.cpp
	$(CC) $(CFLAGS) -c perceptron.cpp

clean: 
	$(RM) *.o perceptron

test: perceptron.out $(TESTFILE)
	./perceptron.out $(TESTFILE)
