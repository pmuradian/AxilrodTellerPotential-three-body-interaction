#include <stdio.h>
#include <math.h>
#include <string.h>
#include <mpi.h>
#include <stdlib.h>
#include <unistd.h>

#define PARTICLE_SIZE 13

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
    double delta_a_x;
    double delta_a_y;
    double delta_a_z;
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
    p->delta_a_x = vals[9];
    p->delta_a_y = vals[10];
    p->delta_a_z = vals[11];
    p->id = vals[12];
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
    arr[9] = p.delta_a_x;
    arr[10] = p.delta_a_y;
    arr[11] = p.delta_a_z;
    arr[12] = p.id;
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

//    printf("%.8f ", E * (a + b / c));

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

double velocity_change(double vel, double a, double d_a, double t) {
    return vel + ((a + d_a) * t) / 2;
}

void calculate_forces(Particle *p_0, Particle *p_1, Particle *p_2, int p0_size, int p1_size, int p2_size, int round, int rank) {

    for (int i = 0; i < p1_size; i++) {
        Particle p1 = p_1[i];
        int must_update = 0;
        for (int j = 0; j < p0_size; j++) {
            Particle p2 = p_0[j];
            if (compare(p1, p2)) {
                continue;
            }
            for (int k = 0; k < p2_size; k++) {
                Particle p3 = p_2[k];
                if (compare(p2, p3) || compare(p1, p3)) {
                    continue;
                }

                double f1 = axilrod_teller_potential_derivative_x(p1, p2, p3);
                double f2 = axilrod_teller_potential_derivative_y(p1, p2, p3);
                double f3 = axilrod_teller_potential_derivative_z(p1, p2, p3);

                if (round == 0) {
                    p1.a_x += f1;
                    p1.a_y += f2;
                    p1.a_z += f3;
                    p2.a_x += f1;
                    p2.a_y += f2;
                    p2.a_z += f3;
                    p3.a_x += f1;
                    p3.a_y += f2;
                    p3.a_z += f3;
                    printf("triple %d%d%d in rank %d\n", p1.id, p2.id, p3.id, rank);
                    printf("coordinates %.15f%.15f%.15f in rank %d\n", p1.x, p2.x, p3.x, rank);
                    printf("triple %.15f%.15f%.15f in rank %d\n", f1, f2, f3, rank);
                    fflush(stdout);
                } else {
                    p1.delta_a_x += f1;
                    p1.delta_a_y += f2;
                    p1.delta_a_y += f3;
                    p2.delta_a_x += f1;
                    p2.delta_a_y += f2;
                    p2.delta_a_y += f3;
                    p3.delta_a_x += f1;
                    p3.delta_a_y += f2;
                    p3.delta_a_y += f3;
                }
                must_update = 1;
                if (must_update) {
                    p_2[k] = p3;
                }
            }
            if (must_update) {
                p_0[j] = p2;
            }
        }
        if (must_update) {
            p_1[i] = p1;
        }
    }
}

// initializes particles from input file
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
        double arr[13] = {0};


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
    MPI_Barrier(MPI_COMM_WORLD);

    if (rank % 2 == 0) {
        MPI_Send(particles, count, MPI_DOUBLE, next_rank(rank, num_proc), 0, MPI_COMM_WORLD);
        MPI_Recv(new_particles, count, MPI_DOUBLE, prev_rank(rank, num_proc), 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    } else {
        MPI_Recv(new_particles, count, MPI_DOUBLE, prev_rank(rank, num_proc), 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Send(particles, count, MPI_DOUBLE, next_rank(rank, num_proc), 0, MPI_COMM_WORLD);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    memcpy(particles, new_particles, sizeof(double) * count);
}

void print_buffer(double *buffer, int size) {
    for (int i = 0; i < size; i++) {
        printf("buffer value = %f\n", buffer[i]);
    }
}

void buffer_to_particles(double *buffer, Particle *particles, int buffer_size, int particle_size) {
    double values[particle_size];
    for (int i = 0; i < buffer_size; i++) {
        for (int j = 0; j < particle_size; j++) {
            values[j] = buffer[i * particle_size + j];
        }
        Particle p;
        populate(&p, values);
        particles[i] = p;
    }
}

void buffer_from_particles(double *buffer, Particle *particles, int particle_cnt, int size_of_particle) {
    int index = 0;
    double message[size_of_particle];

    // populate buffer with particles
    for (int i = 0; i < particle_cnt; i++) {
        toArray(particles[i], message);
        for (int j = 0; j < size_of_particle; j++) {
            buffer[index * size_of_particle + j] = message[j];
        }
        index++;
    }
}

void swap(int *a, int *b) {
    int temp = *b;
    *b = *a;
    *a = temp;
}

int sort_and_concat(int a, int b, int c) {
//    if (a > c)
//        swap(&a, &c);
//    if (a > b)
//        swap(&a, &b);
//    if (b > c)
//        swap(&b, &c);

    return a * 100 + b * 10 + c;
}

int check_if_calculated(int rank, int numProcesses, int *buffer_to_rank, int *triples, int *current_check_index) {
    int skip_this_one = 0;
    if (rank == 0) {
        for (int i = 0; i < numProcesses; i++) {
            int new_triple = 0;

            if (i == 0) {
                new_triple = sort_and_concat(buffer_to_rank[0], buffer_to_rank[1], buffer_to_rank[2]);
            } else {
                MPI_Recv(&new_triple, 1, MPI_INT, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }

            int k = 0;
            for (k = 0; k < *current_check_index; k++) {
//                printf("checking %d with %d\n", new_triple, triples[k]);
                if (new_triple == triples[k]) {
                    skip_this_one = 1;
                    break;
                }
            }
            if (!skip_this_one) {
                triples[*current_check_index] = new_triple;
                *current_check_index += 1;
            }
            if (i != 0) {
                MPI_Send(&skip_this_one, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
            }

            skip_this_one = 0;
        }
    } else {
        int sorted_triple = sort_and_concat(buffer_to_rank[0], buffer_to_rank[1], buffer_to_rank[2]);
        MPI_Send(&sorted_triple, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
        MPI_Recv(&skip_this_one, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    return skip_this_one;
}

// 3 body interaction algorithm
void calculate_interactions(double *b0, double *b1, double *b2, int buffer_size, int size_of_particle, int particles_per_process, int numProcesses, int rank,
                            double time, int round) {
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
    double new_b[buffer_size];
    int *triples = malloc(sizeof(double) * 3 * numProcesses);
    int current_check_index = 0;

    for (int i = numProcesses - 3; i > 0; i -= 3) {
        for (int j = 0; j < i; j++) {
            // check if buffer is already calculated
//            int skip_this_one = check_if_calculated(rank, numProcesses, buffer_to_rank, triples, &current_check_index);

            printf("rank %d has triplet %d\n", rank, sort_and_concat(buffer_to_rank[0], buffer_to_rank[1], buffer_to_rank[2]));

            if (j != 0 || j != numProcesses - 3) {
                // shift buffer
                shift_right(new_b, b[buf_index], buffer_size, rank, numProcesses);
                // set new particles
                buffer_to_particles(b[buf_index], p[buf_index], particles_per_process, size_of_particle);
                // update buffer to process mapping
                buffer_to_rank[buf_index] = prev_rank(buffer_to_rank[buf_index], numProcesses);
                if (buf_index == 1) {
                    my_buffer_location = next_rank(my_buffer_location, numProcesses);
                }
            }
            else {
                calculate_forces(p[1], p[1], p[1], particles_per_process, particles_per_process, particles_per_process, round, rank);
                calculate_forces(p[1], p[1], p[2], particles_per_process, particles_per_process, particles_per_process, round, rank);
                calculate_forces(p[0], p[0], p[2], particles_per_process, particles_per_process, particles_per_process, round, rank);
            }

            if (j == numProcesses - 3) {
                calculate_forces(p[0], p[1], p[1], particles_per_process, particles_per_process, particles_per_process, round, rank);
            }
            calculate_forces(p[0], p[1], p[2], particles_per_process, particles_per_process, particles_per_process, round, rank);
        }
        buffer_from_particles(b[0], p[0], particles_per_process, size_of_particle);
        buffer_from_particles(b[1], p[1], particles_per_process, size_of_particle);
        buffer_from_particles(b[2], p[2], particles_per_process, size_of_particle);
        buf_index = (buf_index + 1) % 3;
    }

    if (numProcesses % 3 == 0) {
        // shift buffer
        shift_right(new_b, b[buf_index], buffer_size, rank, numProcesses);
        // set new particles
        buffer_to_particles(b[buf_index], p[buf_index], particles_per_process, size_of_particle);
        // update buffer to process mapping
        buffer_to_rank[buf_index] = prev_rank(buffer_to_rank[buf_index], numProcesses);

        if ((rank / (numProcesses / 3)) == 0) {
            calculate_forces(p[0], p[1], p[2], particles_per_process, particles_per_process, particles_per_process, round, rank);
        }
    }

    MPI_Request request = MPI_REQUEST_NULL;

    for (int i = 0; i < 3; i++) {
        // update buffer values from calculated particles
        buffer_from_particles(b[i], p[i], particles_per_process, size_of_particle);

        MPI_Isend(b[i], buffer_size, MPI_DOUBLE, buffer_to_rank[i], 0, MPI_COMM_WORLD, &request);
        MPI_Recv(new_b, buffer_size, MPI_DOUBLE, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        memcpy(b[i], new_b, sizeof(double) * buffer_size);
    }

    // sum accelerations for particles
    for (int i = 0; i < particles_per_process; i++) {
        int index_of_a_x = 6 + i * size_of_particle;
        int index_of_a_y = 7 + i * size_of_particle;
        int index_of_a_z = 8 + i * size_of_particle;
        b[1][index_of_a_x] += b[0][index_of_a_x] + b[2][index_of_a_x]; // a_x
        b[1][index_of_a_y] += b[0][index_of_a_y] + b[2][index_of_a_y]; // a_y
        b[1][index_of_a_z] += b[0][index_of_a_z] + b[2][index_of_a_z]; // a_z
    }

    // update positions after acceleration sum
    buffer_to_particles(b[1], p[1], particles_per_process, size_of_particle);
//    buffer_from_particles(b[1], p[1], particles_per_process, size_of_particle);

    if (round == 0) {
        Particle *par = p[1];
        for (int i = 0; i < particles_per_process; i++) {
            Particle ptc = par[i];
            ptc.x = coordinate_change(ptc.x, ptc.vel_x, ptc.a_x, time);
            ptc.y = coordinate_change(ptc.y, ptc.vel_y, ptc.a_y, time);
            ptc.z = coordinate_change(ptc.z, ptc.vel_z, ptc.a_z, time);
            par[i] = ptc;
        }
    } else {
        Particle *par = p[1];
        for (int i = 0; i < particles_per_process; i++) {
            Particle ptc = par[i];
            ptc.vel_x = velocity_change(ptc.vel_x, ptc.a_x, ptc.delta_a_x, time);
            ptc.vel_y = velocity_change(ptc.vel_y, ptc.a_y, ptc.delta_a_y, time);
            ptc.vel_z = velocity_change(ptc.vel_z, ptc.a_z, ptc.delta_a_z, time);
            par[i] = ptc;
        }
    }

    free(triples);

    buffer_from_particles(b[1], p[1], particles_per_process, size_of_particle);
}

void output_to_file(char * path, double *values, int size, int particle_size) {
    FILE *f;

    if((f = fopen(path, "w")) == NULL) {
        perror("File cannot be opened");
        exit(1);
    }

    for (int i = 0; i < size; i += particle_size) {
        for (int j = 0; j < 6; j++)
            fprintf(f, "%0.15lf ", values[i * particle_size + j]);
        fprintf(f, "\n");
    }

    fclose(f);
}

void temp_file_name(char *final, char *original, int rank, int step) {
    strcpy(final, original);
    strcat(final, "_step_");
    char integer_string[32];
    sprintf(integer_string, "%d", step);
    strcat(final, integer_string);
    strcat(final, "_rank_");
    sprintf(integer_string, "%d", rank);
    strcat(final, integer_string);
    strcat(final, ".txt");
}

int main(int argc, char *argv[]) {

    int step_count = 0;
    int verbose = 0;
    char output_file[80];
    char input_file[80];
    double time_interval = 0;

    if (argc >= 5) {
        strcpy(input_file, argv[1]);
        strcpy(output_file, argv[2]);
        step_count = atoi(argv[3]);
        time_interval = atof(argv[4]);
        if (argc == 6) {
            verbose = 1;
        }
    }
    else {
        printf("Not enough arguments passed");
        return 0;
    }

    int numProcesses, myRank;
    double *b0;
    double *b1;
    double *b2;
    double *buffer;
    int size_of_particle = PARTICLE_SIZE;
    int particles_per_process = 0;
    int particle_cnt = 0;
    int buffer_size = 0;
    Particle *particles;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &numProcesses);
    MPI_Comm_rank(MPI_COMM_WORLD, &myRank);

    // initialize particles from input file
    if (myRank == 0) {
        particles = initializeFromFile(input_file, &particle_cnt);
    }

    // broadcast particles count to other processes
    MPI_Bcast(&particle_cnt, 1, MPI_INT, 0, MPI_COMM_WORLD);

    particles_per_process = particle_cnt / numProcesses;
    buffer_size = size_of_particle * particles_per_process;

    b1 = malloc(sizeof(double) * buffer_size);
    b0 = malloc(sizeof(double) * buffer_size);
    b2 = malloc(sizeof(double) * buffer_size);

    if (myRank == 0) {
        buffer = malloc(particle_cnt * size_of_particle * sizeof(double));
        buffer_from_particles(buffer, particles, particle_cnt, size_of_particle);
    }

    free(particles);

    // scatter particles to processes
    MPI_Scatter(buffer, buffer_size, MPI_DOUBLE, b1, buffer_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    int step = 0;
    while (step < step_count) {

        for (int i = 0; i < 2; i++) {
            // for i = 0 update coordinates
            // for i = 1 update velocities

            MPI_Barrier(MPI_COMM_WORLD);
            // send buffer to neighbors
            if (myRank % 2 == 0) {
                MPI_Send(b1, buffer_size, MPI_DOUBLE, prev_rank(myRank, numProcesses), 0, MPI_COMM_WORLD);
                MPI_Send(b1, buffer_size, MPI_DOUBLE, next_rank(myRank, numProcesses), 0, MPI_COMM_WORLD);

                MPI_Recv(b0, buffer_size, MPI_DOUBLE, prev_rank(myRank, numProcesses), 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                MPI_Recv(b2, buffer_size, MPI_DOUBLE, next_rank(myRank, numProcesses), 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
            else {
                MPI_Recv(b2, buffer_size, MPI_DOUBLE, next_rank(myRank, numProcesses), 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                MPI_Recv(b0, buffer_size, MPI_DOUBLE, prev_rank(myRank, numProcesses), 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

                MPI_Send(b1, buffer_size, MPI_DOUBLE, next_rank(myRank, numProcesses), 0, MPI_COMM_WORLD);
                MPI_Send(b1, buffer_size, MPI_DOUBLE, prev_rank(myRank, numProcesses), 0, MPI_COMM_WORLD);
            }

            calculate_interactions(b0, b1, b2, buffer_size, size_of_particle, particles_per_process, numProcesses, myRank, time_interval, i);
        }
        step++;

        if (verbose) {
            // create local output filename
            char tmp[80];
            temp_file_name(tmp, output_file, myRank, step);
            output_to_file(tmp, b1, buffer_size, size_of_particle);
        }
    }

    // does not work correctly, only b1 from rank 0 is gathered, the rest are zeros
    MPI_Gather(b1, buffer_size, MPI_DOUBLE, buffer, buffer_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (myRank == 0) {
        strcat(output_file, "_stepcount.txt");
        output_to_file(output_file, buffer, particle_cnt * size_of_particle, size_of_particle);
    }

    if (myRank == 0) {
        free(buffer);
    }

    free(b0);
    free(b1);
    free(b2);

    printf("finalize in rank %d\n", myRank);
    fflush(stdout);

    MPI_Finalize();

    return 0;
}