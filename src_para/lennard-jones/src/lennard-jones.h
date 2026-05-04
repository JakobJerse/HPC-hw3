#ifndef LJ_H
#define LJ_H

#ifdef __cplusplus
extern "C" {
#endif

#define DT 0.002
#define SIGMA 1.0
#define EPSILON 1.0
#define R_CUT 2.5
#define JITTER 0.05

#define GENERATE_GIF 0
#define FRAME_WIDTH 800
#define FRAME_HEIGHT 800
#define FRAME_EVERY 5
#define FRAME_PARTICLE_RADIUS 2
#define FRAME_DELAY 3
#define GIF_FILE "simulation.gif"



typedef struct {
    double x;
    double y;
    double vx;
    double vy;
    double fx;
    double fy;
} Particle;

typedef struct {
    unsigned int n;
    const Particle *particles;
    double start_kinetic;
    double start_potential;
    double start_total;
    double final_kinetic;
    double final_potential;
    double final_total;
} SimulationResult;

int initialize_particles(
    Particle *particles,
    unsigned int n,
    double box_size,
    double placement_fraction,
    unsigned int seed,
    double temperature
);
void wrap_positions(Particle *particles, unsigned int n, double box_size);

double compute_v_shift(void);
double compute_forces(
    Particle *particles,
    unsigned int n,
    double box_size
);
double leapfrog_step(
    Particle *particles,
    unsigned int n,
    double box_size
);
SimulationResult run_simulation(Particle *particles, unsigned int n, unsigned int nsteps, double box_size, int log_steps);

#ifdef __cplusplus
}
#endif

#endif
