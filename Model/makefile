BIN_DIR=../Bin

CC = gcc
CFLAGS = -lm -pthread -O2 -Wall -funroll-loops

all: ReHession 

ReHession: ReHessionN.c
	$(CC) ReHessionN.c -o ${BIN_DIR}/ReHession $(CFLAGS) -DDROPOUT -DACTIVE

clean:
	pushd ${BIN_DIR}