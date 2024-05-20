#include <stdlib.h>
#include <gmp.h>
#include <cstdint>
#include <time.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/mman.h>
#include <semaphore.h>
#include <unistd.h>
#include <cstring>

#define TLC 12

typedef struct {
    uint32_t limbs[TLC];
} storage;

typedef struct {
    storage x;
    storage y;
} affine_point;

void print_storage(const storage *s) {
    for(int i = 0; i < 12; i++) {
    	printf("\'0x%08x\', ", s->limbs[i]);
    }
    printf("\n");
}

void read_points_from_file(affine_point *points, uint32_t num_points, const char* filename) {

    FILE *file;
    char buffer[130];
    mpz_t big_num;

    // Initialize GMP integer
    mpz_init(big_num);

    // Open the file
    file = fopen(filename, "r");
    if (file == NULL) {
        perror("Error opening file");
        return;
    }

    // Read points from the file
    int i = 0;
    while (i < 2 * num_points && fscanf(file, "%s\n", buffer) == 1) {
        mpz_set_str(big_num, buffer, 10); // Read the big number in decimal

        // Extract limbs from the big number
        for (int j = 0; j < TLC; j+=2) {
            mp_limb_t limb = mpz_getlimbn(big_num, j / 2);
            //
            if (i % 2 == 0) {
                points[i / 2].x.limbs[j] = static_cast<uint32_t>(limb & 0xFFFFFFFF);
                points[i / 2].x.limbs[j + 1] = static_cast<uint32_t>((limb >> 32) & 0xFFFFFFFF);

            } else {
                points[i / 2].y.limbs[j] = static_cast<uint32_t>(limb & 0xFFFFFFFF);
                points[i / 2].y.limbs[j + 1] = static_cast<uint32_t>((limb >> 32) & 0xFFFFFFFF);
            }
        }

        i++;
    }

    // Close the file
    fclose(file);
    mpz_clear(big_num);
}

int main() {

    int num_points = 64 * 1024;
    
    affine_point *points = (affine_point *)malloc(num_points * sizeof(affine_point));
    read_points_from_file(points, num_points, "bls12-381_points.txt");

    affine_point *results = (affine_point *)malloc(sizeof(affine_point));

    affine_point *shm_points, *shm_results;
    sem_t *sem_points_full, *sem_points_empty;
    sem_t *sem_results_full, *sem_results_empty;
    int fd_points, fd_results;

    // Open the shared memory
    fd_points = shm_open("/bls_shared_points", O_RDWR, 0666);
    fd_results = shm_open("/bls_shared_results", O_RDONLY, 0666);
    if (fd_points == -1 || fd_results == -1) {
        perror("shm_open");
        exit(EXIT_FAILURE);
    }

    // Map the shared memory
    shm_points = (affine_point *) mmap(NULL, num_points * sizeof(affine_point), PROT_WRITE, MAP_SHARED, fd_points, 0);
    shm_results = (affine_point *) mmap(NULL, sizeof(affine_point), PROT_READ, MAP_SHARED, fd_results, 0);
    if (shm_points == MAP_FAILED || shm_results == MAP_FAILED) {
        perror("mmap");
        exit(EXIT_FAILURE);
    }

    // Open semaphores
    sem_points_full = sem_open("/sem_points_full", 0);
    sem_points_empty = sem_open("/sem_points_empty", 0);
    sem_results_full = sem_open("/sem_results_full", 0);
    sem_results_empty = sem_open("/sem_results_empty", 0);
    if (sem_points_full == SEM_FAILED || sem_points_empty == SEM_FAILED || sem_results_full == SEM_FAILED || sem_results_empty == SEM_FAILED) {
        perror("sem_open");
        exit(EXIT_FAILURE);
    }

    printf("Starting...\n");

    // Write points to shared memory
    int points_sent, points_received = 0;
    bool all_points_processed = false;
    while (!all_points_processed) {

        if (sem_trywait(sem_points_empty) == -1) {
        } else if (points_sent < 10) {
            printf("Empty, write new data\n");
            memcpy(shm_points, points, num_points * sizeof(affine_point));
            sem_post(sem_points_full);
            points_sent++;
        }


        if (sem_trywait(sem_results_full) == -1) {
        } else {
            printf("New results\n");
            memcpy(results, shm_results, sizeof(affine_point));
            sem_post(sem_results_empty);
            points_received++;
            print_storage(&results->x);
            print_storage(&results->y);
            printf("\n");
        }

        if (points_received == points_sent && points_sent == 10) {
            all_points_processed = true;
        }

    }

    munmap(shm_points, num_points * sizeof(affine_point));
    munmap(shm_results, sizeof(affine_point));
    close(fd_points);
    close(fd_results);
    sem_close(sem_points_full);
    sem_close(sem_points_empty);

}