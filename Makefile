CC=mpicc
CFLAGS=\-o
EXEC=mpiexec

build: main.c
	$(CC) $(CFLAGS) body3 main.c

clean:
	rm body3

execute:
	$(EXEC) -n 4 ./body3 ./input.txt particles_out 1 0.5 -v
