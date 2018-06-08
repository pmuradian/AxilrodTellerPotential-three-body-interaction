CC=cc
CFLAGS=\-o
EXEC=mpiexec

build: main.c
	$(CC) -o3 $(CFLAGS) body3 main.c

buildlocal: main.c
	mpicc $(CFLAGS) body3 main.c

clean:
	rm body3

execute: buildlocal
	$(EXEC) -n 4 ./body3 ./input.txt ./particles_out 5 0.5 -v
