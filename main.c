#include <stdio.h>
#include <math.h>
#include <string.h>
#include <stddef.h>
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
    return (p1.id == p2.id);
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

void print_particle(Particle p) {
    double arr[PARTICLE_SIZE];
    toArray(p, arr);

    printf("description ");
    for (int i = 0; i < PARTICLE_SIZE; i++ ) {
        printf("%.15f ", arr[i]);
    }
    printf("\n");
}

MPI_Datatype createParticleDataType()
{
    int blocklengths[PARTICLE_SIZE] = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
    MPI_Datatype types[PARTICLE_SIZE] = {MPI_DOUBLE, MPI_DOUBLE, MPI_DOUBLE, MPI_DOUBLE, MPI_DOUBLE, MPI_DOUBLE, MPI_DOUBLE,
                                         MPI_DOUBLE, MPI_DOUBLE, MPI_DOUBLE, MPI_DOUBLE, MPI_DOUBLE, MPI_INT};
    MPI_Datatype MPI_PARTICLE;
    MPI_Aint offsets[PARTICLE_SIZE];

    offsets[0] = offsetof(Particle, x);
    offsets[1] = offsetof(Particle, y);
    offsets[2] = offsetof(Particle, z);
    offsets[3] = offsetof(Particle, vel_x);
    offsets[4] = offsetof(Particle, vel_y);
    offsets[5] = offsetof(Particle, vel_z);
    offsets[6] = offsetof(Particle, a_x);
    offsets[7] = offsetof(Particle, a_y);
    offsets[8] = offsetof(Particle, a_z);
    offsets[9] = offsetof(Particle, delta_a_x);
    offsets[10] = offsetof(Particle, delta_a_y);
    offsets[11] = offsetof(Particle, delta_a_z);
    offsets[12] = offsetof(Particle, id);

    MPI_Type_create_struct(PARTICLE_SIZE, blocklengths, offsets, types, &MPI_PARTICLE);

    return MPI_PARTICLE;
}

double axilrod_teller_potential(double r1, double r2, double r3)
{
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
    return sqrt((p1.x - p2.x) * (p1.x - p2.x) + (p1.y - p2.y) * (p1.y - p2.y) + (p1.z - p2.z) * (p1.z - p2.z));
}

double axilrod_teller_potential_particles(Particle p1, Particle p2, Particle p3) {
    double r1 = distance_between(p1, p2);
    double r2 = distance_between(p1, p3);
    double r3 = distance_between(p2, p3);

    return fabs(axilrod_teller_potential(r1, r2, r3));
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

void calculate_forces(Particle *p_0, Particle *p_1, Particle *p_2, int p0_size, int p1_size, int p2_size, int round, int rank)
{
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

                if (round == 0) {
                    p1.a_x += axilrod_teller_potential_derivative_x(p1, p2, p3);
                    p1.a_y += axilrod_teller_potential_derivative_y(p1, p2, p3);
                    p1.a_z += axilrod_teller_potential_derivative_z(p1, p2, p3);

                    p2.a_x += axilrod_teller_potential_derivative_x(p2, p1, p3);
                    p2.a_y += axilrod_teller_potential_derivative_y(p2, p1, p3);
                    p2.a_z += axilrod_teller_potential_derivative_z(p2, p1, p3);

                    p3.a_x += axilrod_teller_potential_derivative_x(p3, p2, p1);
                    p3.a_y += axilrod_teller_potential_derivative_y(p3, p2, p1);
                    p3.a_z += axilrod_teller_potential_derivative_z(p3, p2, p1);
                } else {
                    p1.delta_a_x += axilrod_teller_potential_derivative_x(p1, p2, p3);
                    p1.delta_a_y += axilrod_teller_potential_derivative_y(p1, p2, p3);
                    p1.delta_a_y += axilrod_teller_potential_derivative_z(p1, p2, p3);

                    p2.delta_a_x += axilrod_teller_potential_derivative_x(p2, p1, p3);
                    p2.delta_a_y += axilrod_teller_potential_derivative_y(p2, p1, p3);
                    p2.delta_a_y += axilrod_teller_potential_derivative_z(p2, p1, p3);

                    p3.delta_a_x += axilrod_teller_potential_derivative_x(p3, p2, p1);
                    p3.delta_a_y += axilrod_teller_potential_derivative_y(p3, p2, p1);
                    p3.delta_a_y += axilrod_teller_potential_derivative_z(p3, p2, p1);
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
Particle* initializeFromFile(char *path, int *p_count)
{
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

void shift_right(Particle *new_particles, Particle *particles, int count, int rank, int num_proc) {
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Datatype MPI_PARTICLE = createParticleDataType();
    MPI_Type_commit(&MPI_PARTICLE);

    if (rank % 2 == 0) {
        MPI_Send(particles, count, MPI_PARTICLE, next_rank(rank, num_proc), 0, MPI_COMM_WORLD);
        MPI_Recv(new_particles, count, MPI_PARTICLE, prev_rank(rank, num_proc), 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    } else {
        MPI_Recv(new_particles, count, MPI_PARTICLE, prev_rank(rank, num_proc), 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Send(particles, count, MPI_PARTICLE, next_rank(rank, num_proc), 0, MPI_COMM_WORLD);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    memcpy(particles, new_particles, sizeof(Particle) * count);
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

// 3 body interaction algorithm
void calculate_interactions(Particle *p0, Particle *p1, Particle *p2, int particles_per_process, int numProcesses, int rank, double time, int round)
{
    Particle *p[3] = {p0, p1, p2};
    int buf_index = 0;

    MPI_Datatype MPI_PARTICLE = createParticleDataType();
    MPI_Type_commit(&MPI_PARTICLE);

    // buffer to process mapping, so it can be used later to send buffers to original processes
    int buffer_to_rank[3] = { prev_rank(rank, numProcesses), rank, next_rank(rank, numProcesses) };
    int my_buffer_location = rank;
    Particle new_p[particles_per_process];
    int current_check_index = 0;

    for (int i = numProcesses - 3; i > 0; i -= 3) {
        for (int j = 0; j < i; j++) {
            // check if buffer is already calculated
            if (j != 0 || j != numProcesses - 3) {
                // shift buffer
                shift_right(new_p, p[buf_index], particles_per_process, rank, numProcesses);
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
        buf_index = (buf_index + 1) % 3;
    }

    if (numProcesses % 3 == 0) {
        // shift buffer
        shift_right(new_p, p[buf_index], particles_per_process, rank, numProcesses);
        // update buffer to process mapping
        buffer_to_rank[buf_index] = prev_rank(buffer_to_rank[buf_index], numProcesses);

        if ((rank / (numProcesses / 3)) == 0) {
            calculate_forces(p[0], p[1], p[2], particles_per_process, particles_per_process, particles_per_process, round, rank);
        }
    }

    MPI_Request request = MPI_REQUEST_NULL;

    for (int i = 0; i < 3; i++) {
        MPI_Isend(p[i], particles_per_process, MPI_PARTICLE, buffer_to_rank[i], 0, MPI_COMM_WORLD, &request);
        MPI_Recv(new_p, particles_per_process, MPI_PARTICLE, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        memcpy(p[i], new_p, sizeof(Particle) * particles_per_process);
    }

    // sum accelerations for particles


    // update positions after acceleration sum
    if (round == 0) {
        Particle *par = p[1];
        for (int i = 0; i < particles_per_process; i++) {
            Particle ptc = par[i];
            ptc.a_x += p[0][i].a_x + p[2][i].a_x;
            ptc.a_y += p[0][i].a_y + p[2][i].a_y;
            ptc.a_z += p[0][i].a_z + p[2][i].a_z;
            ptc.a_x *= 2;
            ptc.a_y *= 2;
            ptc.a_z *= 2;
            ptc.x = coordinate_change(ptc.x, ptc.vel_x, ptc.a_x, time);
            ptc.y = coordinate_change(ptc.y, ptc.vel_y, ptc.a_y, time);
            ptc.z = coordinate_change(ptc.z, ptc.vel_z, ptc.a_z, time);
            par[i] = ptc;
        }
    } else {
        Particle *par = p[1];
        for (int i = 0; i < particles_per_process; i++) {
            Particle ptc = par[i];
            ptc.delta_a_x += p[0][i].delta_a_x + p[2][i].delta_a_x;
            ptc.delta_a_y += p[0][i].delta_a_y + p[2][i].delta_a_y;
            ptc.delta_a_z += p[0][i].delta_a_z + p[2][i].delta_a_z;
            ptc.delta_a_x *= 2;
            ptc.delta_a_y *= 2;
            ptc.delta_a_z *= 2;
            printf("ax = %.15f ay = %.15f az = %.15f in rank %d\n", p[1][i].a_x, p[1][i].a_y, p[1][i].a_z, rank);
            ptc.vel_x = velocity_change(ptc.vel_x, ptc.a_x, ptc.delta_a_x, time);
            ptc.vel_y = velocity_change(ptc.vel_y, ptc.a_y, ptc.delta_a_y, time);
            ptc.vel_z = velocity_change(ptc.vel_z, ptc.a_z, ptc.delta_a_z, time);
            par[i] = ptc;
        }
    }
}

void output_to_file(char * path, Particle *particle, int count) {
    FILE *f;
    if ((f = fopen(path, "w")) == NULL) {
        perror("File cannot be opened");
        exit(1);
    }

    double values[PARTICLE_SIZE];
    toArray(*particle, values);

    for (int i = 0; i < count; i += PARTICLE_SIZE) {
        for (int j = 0; j < 6; j++)
            fprintf(f, "%0.16lf ", values[i * PARTICLE_SIZE + j]);
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

int main(int argc, char *argv[])
{
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

    // create MPI_PARTICLE type
    MPI_Datatype MPI_PARTICLE = createParticleDataType();
    MPI_Type_commit(&MPI_PARTICLE);

    Particle *p0 = malloc(sizeof(Particle) * particle_cnt);
    Particle *p1 = malloc(sizeof(Particle) * particle_cnt);
    Particle *p2 = malloc(sizeof(Particle) * particle_cnt);

    Particle p_1, p_2, p_3, p_4;

    p_1 = particles[0];
    p_2 = particles[1];
    p_3 = particles[2];
    p_4 = particles[3];

    p_1.a_x = 2 * (axilrod_teller_potential_derivative_x(p_1, p_2, p_3) + axilrod_teller_potential_derivative_x(p_1, p_3, p_4) + axilrod_teller_potential_derivative_x(p_1, p_2, p_4));
    p_1.a_y = 2 * (axilrod_teller_potential_derivative_y(p_1, p_2, p_3) + axilrod_teller_potential_derivative_y(p_1, p_3, p_4) + axilrod_teller_potential_derivative_y(p_1, p_2, p_4));
    p_1.a_z = 2 * (axilrod_teller_potential_derivative_z(p_1, p_2, p_3) + axilrod_teller_potential_derivative_z(p_1, p_3, p_4) + axilrod_teller_potential_derivative_z(p_1, p_2, p_4));

    p_1.x = coordinate_change(p_1.x, p_1.vel_x, p_1.a_x, time_interval);
    p_1.y = coordinate_change(p_1.y, p_1.vel_y, p_1.a_y, time_interval);
    p_1.z = coordinate_change(p_1.z, p_1.vel_z, p_1.a_z, time_interval);

    p_1.delta_a_x = 2 * (axilrod_teller_potential_derivative_x(p_1, p_2, p_3) + axilrod_teller_potential_derivative_x(p_1, p_3, p_4) + axilrod_teller_potential_derivative_x(p_1, p_2, p_4));
    p_1.delta_a_y = 2 * (axilrod_teller_potential_derivative_y(p_1, p_2, p_3) + axilrod_teller_potential_derivative_y(p_1, p_3, p_4) + axilrod_teller_potential_derivative_y(p_1, p_2, p_4));
    p_1.delta_a_z = 2 * (axilrod_teller_potential_derivative_z(p_1, p_2, p_3) + axilrod_teller_potential_derivative_z(p_1, p_3, p_4) + axilrod_teller_potential_derivative_z(p_1, p_2, p_4));

    p_1.vel_x = velocity_change(p_1.vel_x, p_1.a_x, p_1.delta_a_x, time_interval);
    p_1.vel_y = velocity_change(p_1.vel_y, p_1.a_y, p_1.delta_a_y, time_interval);
    p_1.vel_z = velocity_change(p_1.vel_z, p_1.a_z, p_1.delta_a_z, time_interval);

    printf(">>>>>> %.15f, %.15f <<<<<<<<\n", p_1.x, p_1.vel_x);

    // scatter particles to processes
    MPI_Scatter(particles, particles_per_process, MPI_PARTICLE, p1, particles_per_process, MPI_PARTICLE, 0, MPI_COMM_WORLD);

    int step = 0;
    while (step < step_count) {

        for (int i = 0; i < 2; i++) {
            // for i = 0 update coordinates
            // for i = 1 update velocities

            MPI_Barrier(MPI_COMM_WORLD);
            // send buffer to neighbors
            if (myRank % 2 == 0) {
                MPI_Send(p1, particles_per_process, MPI_PARTICLE, prev_rank(myRank, numProcesses), 0, MPI_COMM_WORLD);
                MPI_Send(p1, particles_per_process, MPI_PARTICLE, next_rank(myRank, numProcesses), 0, MPI_COMM_WORLD);

                MPI_Recv(p0, particles_per_process, MPI_PARTICLE, prev_rank(myRank, numProcesses), 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                MPI_Recv(p2, particles_per_process, MPI_PARTICLE, next_rank(myRank, numProcesses), 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
            else {
                MPI_Recv(p2, particles_per_process, MPI_PARTICLE, next_rank(myRank, numProcesses), 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                MPI_Recv(p0, particles_per_process, MPI_PARTICLE, prev_rank(myRank, numProcesses), 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

                MPI_Send(p1, particles_per_process, MPI_PARTICLE, next_rank(myRank, numProcesses), 0, MPI_COMM_WORLD);
                MPI_Send(p1, particles_per_process, MPI_PARTICLE, prev_rank(myRank, numProcesses), 0, MPI_COMM_WORLD);
            }
            calculate_interactions(p0, p1, p2, particles_per_process, numProcesses, myRank, time_interval, i);
        }
        step++;

        if (verbose) {
            // create local output filename
            char tmp[80];
            temp_file_name(tmp, output_file, myRank, step);
            output_to_file(tmp, p1, particles_per_process);
        }
    }

    // does not work correctly, only b1 from rank 0 is gathered, the rest are zeros
    MPI_Gather(p1, particles_per_process, MPI_PARTICLE, particles, particle_cnt, MPI_PARTICLE, 0, MPI_COMM_WORLD);

    if (myRank == 0) {
        strcat(output_file, "_stepcount.txt");
        output_to_file(output_file, particles, particle_cnt);
    }

    if (myRank == 0) {
        free(buffer);
    }

    free(p0);
    free(p1);
    free(p2);

    printf("finalize in rank %d\n", myRank);
    fflush(stdout);

    MPI_Finalize();

    return 0;
}