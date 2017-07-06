BIN_DIR=./

CC = gcc
CFLAGS = -lm -pthread -O2 -Wall -funroll-loops

all: ReHession 

ReHession: ReHession.c
	$(CC) ReHession.c -o ${BIN_DIR}/ReHession $(CFLAGS) -DDROPOUT -DACTIVE

clean:
	pushd ${BIN_DIR}