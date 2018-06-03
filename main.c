#include <stdio.h>
#include <math.h>
#include <string.h>
#include <mpi.h>
#include <stdlib.h>

#define PARTICLE_SIZE 10

struct Particle {
    double x;
    double y;
    double z;
    double vel_x;
    double vel_y;
    double vel_z;
    double a_x;
    double a_y;
    double a_z;
    int id;
};
typedef struct Particle Particle;

// Formula constants
const double e = 4.69041575982343e-08;
const double min_x = 10e-10;
const double E = 1.0;

void populate(Particle* p, double* vals) {
    p->x = vals[0];
    p->y = vals[1];
    p->z = vals[2];
    p->vel_x = vals[3];
    p->vel_y = vals[4];
    p->vel_z = vals[5];
    p->a_x = vals[6];
    p->a_y = vals[7];
    p->a_z = vals[8];
    p->id = vals[9];
}

void copy(Particle* p0, struct Particle p) {
    p0->x = p.x;
    p0->y = p.y;
    p0->z = p.z;
    p0->vel_x = p.vel_x;
    p0->vel_y = p.vel_y;
    p0->vel_z = p.vel_z;
}

int compare(Particle p1, Particle p2) {
    if (p1.id == p2.id) {
        return 1;
    }
    return 0;
}

void toArray(Particle p, double *arr) {
    arr[0] = p.x;
    arr[1] = p.y;
    arr[2] = p.z;
    arr[3] = p.vel_x;
    arr[4] = p.vel_y;
    arr[5] = p.vel_z;
    arr[6] = p.a_x;
    arr[7] = p.a_y;
    arr[8] = p.a_z;
    arr[9] = p.id;
}

double axilrod_teller_potential(double r1, double r2, double r3) {

    if (r1 < min_x) {
        r1 = min_x;
    }
    if (r2 < min_x) {
        r2 = min_x;
    }
    if (r3 < min_x) {
        r3 = min_x;
    }

    double p_r1 = pow(r1, 2);
    double p_r2 = pow(r2, 2);
    double p_r3 = pow(r3, 2);

    double a = 1 / pow(r1 * r2 * r3, 3);
    double b = 3 * (-1 * p_r1 + p_r2 + p_r3) * (p_r1 - p_r2 + p_r3) * (p_r1 + p_r2 - p_r3);
    double c = 8 * pow(r1 * r2 * r3, 5);

    return E * (a + b / c);
}

double distance_between(Particle p1, Particle p2) {
    return sqrt(fabs(p1.x * p1.x - p2.x * p2.x) + fabs(p1.y * p1.y - p2.y * p2.y) + fabs(p1.z * p1.z - p2.z * p2.z));
}


double axilrod_teller_potential_particles(Particle p1, Particle p2, Particle p3) {
    double r1 = distance_between(p1, p2);
    double r2 = distance_between(p1, p3);
    double r3 = distance_between(p2, p3);

    return axilrod_teller_potential(r1, r2, r3);
}

double axilrod_teller_potential_derivative_x(Particle p1, Particle p2, Particle p3) {
    double h = e * p1.x;
    Particle p1_min, p1_plus;
    copy(&p1_min, p1);
    copy(&p1_plus, p1);
    p1_min.x -= h;
    p1_plus.x += h;

    return (axilrod_teller_potential_particles(p1_plus, p2, p3) - axilrod_teller_potential_particles(p1_min, p2, p3)) / (p1_plus.x - p1_min.x);
}

double axilrod_teller_potential_derivative_y(Particle p1, Particle p2, Particle p3) {
    double h = e * p1.y;
    Particle p1_min, p1_plus;
    copy(&p1_min, p1);
    copy(&p1_plus, p1);
    p1_min.y -= h;
    p1_plus.y += h;

    return (axilrod_teller_potential_particles(p1_plus, p2, p3) - axilrod_teller_potential_particles(p1_min, p2, p3)) / (p1_plus.y - p1_min.y);
}

double axilrod_teller_potential_derivative_z(Particle p1, Particle p2, Particle p3) {
    double h = e * p1.z;
    Particle p1_min, p1_plus;
    copy(&p1_min, p1);
    copy(&p1_plus, p1);
    p1_min.z -= h;
    p1_plus.z += h;

    return (axilrod_teller_potential_particles(p1_plus, p2, p3) - axilrod_teller_potential_particles(p1_min, p2, p3)) / (p1_plus.z - p1_min.z);
}

double coordinate_change(double axis, double vel, double a, double t) {
    return axis + vel * t + (a * t * t) / 2;
}

double acceleration_change(double axis, double vel, double a, double t) {
    return axis + vel * t + (a * t * t) / 2;
}

double velocity_change(double vel, double a, double d_a, double t) {
    return vel + ((a + d_a) * t) / 2;
}

void calculate_forces(Particle *p_0, Particle *p_1, Particle *p_2, int p0_size, int p1_size, int p2_size, int rank,
                      double time) {

//    if (rank == 0)
//        printf("new");
    for (int i = 0; i < p1_size; i++) {
        Particle p1 = p_1[i];
        int must_update = 0;
//        if (rank == 0)
//            printf("initial coordinates are x = %.15f y = %.15f z = %.15f\n", p1.x, p1.y, p1.z);
        for (int j = 0; j < p0_size; j++) {
            Particle p2 = p_0[j];
            if (compare(p1, p2)) {
                continue;
            }
//            if (rank == 0)
//                printf("initial coordinates are x = %.15f y = %.15f z = %.15f\n", p2.x, p2.y, p2.z);
            for (int k = 0; k < p2_size; k++) {
                Particle p3 = p_2[k];
                if (compare(p2, p3) || compare(p1, p3)) {
                    continue;
                }
//                if (rank == 0)
//                    printf("initial coordinates are x = %.15f y = %.15f z = %.15f\n", p3.x, p3.y, p3.z);

                double f1 = axilrod_teller_potential_derivative_x(p1, p2, p3);
//                p1.x = coordinate_change(p1.x, p1.vel_x, p1.a_x, time);

                double f2 = axilrod_teller_potential_derivative_y(p1, p2, p3);
//                p1.y = coordinate_change(p1.y, p1.vel_y, p1.a_y, time);

                double f3 = axilrod_teller_potential_derivative_z(p1, p2, p3);

                p1.a_x += f1;
                p2.a_x += f1;
                p3.a_x += f1;
                p1.a_y += f2;
                p2.a_y += f2;
                p3.a_y += f2;
                p1.a_z += f3;
                p2.a_z += f3;
                p3.a_z += f3;
//                p1.z = coordinate_change(p1.z, p1.vel_z, p1.a_z, time);
//                if (rank == 0) {
//                    printf("id = %d, %d, %d\n", p1.id, p2.id, p3.id);
//                    printf("x = %.15f y = %.15f z = %.15f\n", p1.x, p1.y, p1.z);
//                }
                must_update = 1;
            }
        }
        if (must_update) {
//            p1.x = coordinate_change(p1.x, p1.vel_x, p1.a_x, time);
//            p1.y = coordinate_change(p1.y, p1.vel_y, p1.a_y, time);
//            p1.z = coordinate_change(p1.z, p1.vel_z, p1.a_z, time);
            p_1[i] = p1;
//            if (rank == 0)
//                printf("x = %.15f y = %.15f z = %.15f\n", p1.x, p1.y, p1.z);
        }
    }
}

Particle* initializeFromFile(char *path, int *p_count) {
    FILE *file = fopen(path, "r");

    if (file == NULL) {
        printf("Unable to open file %s", path);
        fflush(stdout);
        return 0;
    }

    char *line;
    int len = 0;
    int particle_count = 0;

    while (getline(&line, &len, file) != -1) {
        particle_count++;
    }
    fseek(file, 0, SEEK_SET);

    *p_count = particle_count;

    Particle *particles = malloc(sizeof(Particle) * particle_count);

    int j = 0;
    while (getline(&line, &len, file) != -1) {
        double arr[10] = {0};


        char *vals;
        int i = 0;
        while ((vals = strsep(&line, " "))) {
            arr[i++] = atof(vals);
        }
        Particle p;
        populate(&p, arr);
        p.id = j;
        particles[j++] = p;
    }

    return particles;
}

int next_rank(int rank, int count) {
    return rank == count - 1 ? 0 : rank + 1;
}

int prev_rank(int rank, int count) {
    return rank == 0 ? count - 1 : rank - 1;
}

void shift_right(double *new_particles, double *particles, int count, int rank, int num_proc) {
    MPI_Request request = MPI_REQUEST_NULL;
    MPI_Barrier(MPI_COMM_WORLD);

    MPI_Isend(particles, count, MPI_DOUBLE, next_rank(rank, num_proc), 0, MPI_COMM_WORLD, &request);
    MPI_Recv(new_particles, count, MPI_DOUBLE, prev_rank(rank, num_proc), 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
}

void print_buffer(double *buffer, int size) {
    for (int i = 0; i < size; i++) {
        printf("buffer value = %f\n", buffer[i]);
    }
}

void set_particles(double *buffer, Particle *particles, int buffer_size, int particle_size) {
    for (int i = 0; i < buffer_size; i++) {
        double values[particle_size];
        for (int j = 0; j < particle_size; j++) {
            values[j] = buffer[i * particle_size + j];
        }
        Particle p;
        populate(&p, values);
        particles[i] = p;
    }
}

void set_buffer(double *buffer, Particle *particles, int particle_cnt, int size_of_particle) {
    int index = 0;
    double *message = malloc(size_of_particle * sizeof(double));

    // populate buffer with particles
    for (int i = 0; i < particle_cnt; i++) {
        toArray(particles[i], message);
        for (int j = 0; j < size_of_particle; j++) {
            buffer[index * size_of_particle + j] = message[j];
        }
        index++;
    }
    free(message);
}

void swap(int *a, int *b) {
    int temp = *b;
    *b = *a;
    *a = temp;
}

int* sort(int a, int b, int c) {
    if (a > c)
        swap(&a, &c);
    if (a > b)
        swap(&a, &b);
    if (b > c)
        swap(&b, &c);

    int arr[3] = {a, b, c};
    return arr;
}

// 3 body interaction algorithm
void calculate_interactions(double *b0, double *b1, double *b2, int buffer_size, int size_of_particle, int particles_per_process, int numProcesses, int rank,
                            double time) {
    Particle particles_b0[particles_per_process];
    Particle particles_b1[particles_per_process];
    Particle particles_b2[particles_per_process];

    double *b[3] = {b0, b1, b2};
    Particle *p[3] = {particles_b0, particles_b1, particles_b2};
    int buf_index = 0;

    for (int i = 0; i < particles_per_process; i++) {
        double values_b0[size_of_particle];
        double values_b1[size_of_particle];
        double values_b2[size_of_particle];
        for (int j = 0; j < size_of_particle; j++) {
            values_b0[j] = b0[i * size_of_particle + j];
            values_b1[j] = b1[i * size_of_particle + j];
            values_b2[j] = b2[i * size_of_particle + j];
        }
        Particle p0, p1, p2;
        populate(&p0, values_b0);
        populate(&p1, values_b1);
        populate(&p2, values_b2);
        particles_b0[i] = p0;
        particles_b1[i] = p1;
        particles_b2[i] = p2;
    }

    // buffer to process mapping, so it can be used later to send buffers to original processes
    int buffer_to_rank[3] = {prev_rank(rank, numProcesses), rank, next_rank(rank, numProcesses)};
    int my_buffer_location = rank;

    int calculated_buffers[1000][3] = { -1 };
    int *selected_buffers = malloc(sizeof(int) * numProcesses * 3);

    for (int i = numProcesses; i > 0; i -= 3) {
        for (int j = 0; j < i; j++) {

            if (j != 0 || j != numProcesses - 3) {
                // shift buffer
                double *new_b = malloc(sizeof(double) * buffer_size);
                shift_right(new_b, b[buf_index], buffer_size, rank, numProcesses);
                free(b[buf_index]);
                b[buf_index] = new_b;
                // set new particles
                set_particles(b[buf_index], p[buf_index], particles_per_process, size_of_particle);
                // update buffer to process mapping
                buffer_to_rank[buf_index] = prev_rank(buffer_to_rank[buf_index], numProcesses);
                if (buf_index == 1) {
                    my_buffer_location = next_rank(my_buffer_location, numProcesses);
                }
            }
            else {
                calculate_forces(p[1], p[1], p[1], particles_per_process, particles_per_process, particles_per_process, rank, time);
                calculate_forces(p[1], p[1], p[2], particles_per_process, particles_per_process, particles_per_process, rank, time);
                calculate_forces(p[0], p[0], p[2], particles_per_process, particles_per_process, particles_per_process, rank, time);
            }

            int *buffer_ids = sort(buffer_to_rank[0], buffer_to_rank[1], buffer_to_rank[2]);
            MPI_Allgather(buffer_ids, 3, MPI_INT, selected_buffers, numProcesses, MPI_INT, MPI_COMM_WORLD);
            int skip_calculation = 0;

            for (int k = 0; k < numProcesses * 3; k += 3) {
                for (int l = 0; l < 1000; l++) {
                    if (calculated_buffers[l][0] == -1) {

                    }

                    if (calculated_buffers[l][0] == selected_buffers[k] && calculated_buffers[l][1] == selected_buffers[k + 1] && calculated_buffers[l][2] == selected_buffers[k + 2]) {
                        if (k / 3 == rank) {
                            skip_calculation = 1;
                        }
                        break;
                    }
                }
            }

            if (j == numProcesses - 3) {
                calculate_forces(p[0], p[1], p[1], particles_per_process, particles_per_process, particles_per_process, rank, time);
            }
            calculate_forces(p[0], p[1], p[2], particles_per_process, particles_per_process, particles_per_process, rank, time);
        }

        buf_index = (buf_index + 1) % 3;
    }

    if (numProcesses % 3 == 0) {
        // shift buffer
        double *new_b = malloc(sizeof(double) * buffer_size);
        buf_index = prev_rank(buf_index, 3);
        shift_right(new_b, b[buf_index], buffer_size, rank, numProcesses);
        free(b[buf_index]);
        b[buf_index] = new_b;

        // set new particles
        set_particles(b[buf_index], p[buf_index], particles_per_process, size_of_particle);
        // update buffer to process mapping
        buffer_to_rank[buf_index] = prev_rank(buffer_to_rank[buf_index], numProcesses);

        if ((rank / (numProcesses / 3)) == 0) {
            calculate_forces(p[0], p[1], p[2], particles_per_process, particles_per_process, particles_per_process, rank, time);
        }
    }

    MPI_Request request = MPI_REQUEST_NULL;


    for (int i = 0; i < 3; i++) {
        // update buffer values from calculated particles
        set_buffer(b[i], p[i], particles_per_process, size_of_particle);

        // send buffers to initial processes
        MPI_Isend(b[i], buffer_size, MPI_DOUBLE, buffer_to_rank[i], 0, MPI_COMM_WORLD, &request);
        double *new_b = malloc(sizeof(double) * buffer_size);
        MPI_Recv(new_b, buffer_size, MPI_DOUBLE, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        b[i] = new_b;
    }

//    MPI_Barrier(MPI_COMM_WORLD);

    for (int i = 0; i < 3; i++) {
        free(b[i]);
    }
}

int main(int argc, char *argv[]) {

    int step_count = 1;
    int verbose = 0;
    char *output_file = "";
    char *input_file = "";
    double time_interval = 0.5;

    if (argc >= 5) {
        input_file = argv[1];
        output_file = argv[2];
        step_count = atoi(argv[3]);
        time_interval = atof(argv[4]);
        if (argc == 6) {
            verbose = 1;
        }
    }
    else {
//        printf("Not enough arguments passed");
//        return 0;
    }

    MPI_Init(&argc, &argv);

    int numProcesses, myRank;
    MPI_Comm_size(MPI_COMM_WORLD, &numProcesses);
    MPI_Comm_rank(MPI_COMM_WORLD, &myRank);
    double *b0;
    double *b1;
    double *b2;
    double *buffer;
    int size_of_particle = PARTICLE_SIZE;
    int particles_per_process = 0;
    int particle_cnt = 0;
    int buffer_size = 0;
    Particle *particles;

    // initialize particles from input file
    if (myRank == 0) {
        particles = initializeFromFile(input_file, &particle_cnt);
    }

    // broadcast particles count to other processes
    MPI_Bcast(&particle_cnt, 1, MPI_INT, 0, MPI_COMM_WORLD);

    particles_per_process = particle_cnt / numProcesses;
    buffer_size = size_of_particle * particles_per_process;

    b0 = malloc(sizeof(double) * buffer_size);
    b1 = malloc(sizeof(double) * buffer_size);
    b2 = malloc(sizeof(double) * buffer_size);

    if (myRank == 0) {
        buffer = malloc(particle_cnt * size_of_particle * sizeof(double));
        set_buffer(buffer, particles, particle_cnt, size_of_particle);
    }

    // scatter particles to processes
    MPI_Scatter(buffer, buffer_size, MPI_DOUBLE, b1, buffer_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    if (myRank == 0) {
        free(buffer);
    }

    MPI_Request request = MPI_REQUEST_NULL;
    // send buffer to neighbors
    MPI_Isend(b1, buffer_size, MPI_DOUBLE, prev_rank(myRank, numProcesses), 0, MPI_COMM_WORLD, &request);
    MPI_Isend(b1, buffer_size, MPI_DOUBLE, next_rank(myRank, numProcesses), 0, MPI_COMM_WORLD, &request);

    // receive buffers from neighbors
    MPI_Recv(b0, buffer_size, MPI_DOUBLE, prev_rank(myRank, numProcesses), 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    MPI_Recv(b2, buffer_size, MPI_DOUBLE, next_rank(myRank, numProcesses), 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    while (step_count > 0) {
        calculate_interactions(b0, b1, b2, buffer_size, size_of_particle, particles_per_process, numProcesses, myRank, time_interval);
        step_count--;
    }

    MPI_Finalize();

    return 0;
}