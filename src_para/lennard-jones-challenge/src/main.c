#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

#include "lennard-jones.h"

void print_help(const char *exe) {
    printf("Usage: %s [N] [nsteps]\n", exe);
}

// DEMO main. Your main.c will not be compiled into the final app for the competition, 
// but you can use it for testing and benchmarking your code. 

int main(int argc, char **argv) {
    // default parameters
    unsigned int nsteps = 100;
    unsigned int n = 100;
    double density = 0.95;
    double temperature = 0.5;
    unsigned int seed = 42;
    double particle_box_size;
    double box_size;
    double box_fraction;
    
    Particle *particles = NULL;
    SimulationResult result;

    // read command line arguments
    if (argc > 1) {
        n = (unsigned int)strtoul(argv[1], NULL, 10);
    }
    if (argc > 2) {
        nsteps = (unsigned int)strtoul(argv[2], NULL, 10);
    }
    if (argc > 3) {
        print_help(argv[0]);
        return 1;
    }

    // simulation box size is determined by the number of particles and the desired density
    particle_box_size = ceil(cbrt((double)n / density));
    box_size = (4 / 3.0) * particle_box_size;
    box_fraction = particle_box_size / box_size;

    // allocate memory for particles
    if (!(particles = calloc(n, sizeof(Particle)))) {
        fprintf(stderr, "Failed to allocate simulation arrays.\n");
        return 1;
    }

    // initalize particles with random positions and velocities
    if (!initialize_particles(
            particles,
            n,
            box_size,
            box_fraction,
            seed,
            temperature
        )) {
        fprintf(stderr, "Failed to initialize particles.\n");
        free(particles);
        return 1;
    }

    //run simulation and measure time
    double start = omp_get_wtime();
    result = run_simulation(particles, n, nsteps, box_size, 1);
    double stop = omp_get_wtime();
    printf("\nFinished simulation.\n");
    printf("Final KE: %10.4f | delta: %+.4f\n", result.final_kinetic, result.final_kinetic - result.start_kinetic);
    printf("Final PE: %10.4f | delta: %+.4f\n", result.final_potential, result.final_potential - result.start_potential);
    printf("Final E:  %10.4f | delta: %+.4f\n", result.final_total, result.final_total - result.start_total);
    printf("Simulation time %d steps: %.3f seconds\n", nsteps, stop - start);

    free(particles);
    return 0;
}
