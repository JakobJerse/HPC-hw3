#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Include CUDA headers
// #include <cuda_runtime.h>
// #include <cuda.h>


#include "gifenc.h"
#include "lennard-jones.h"

// plotting functions
#if GENERATE_GIF
uint8_t palette[] = {
                             0, 0, 0,
                             255, 255, 0};

void set_pixel(uint8_t *img, int w, int h, int x, int y, uint8_t index) {
    if (x < 0 || y < 0 || x >= w || y >= h) {
        return;
    }
    size_t idx = (size_t)y * (size_t)w + (size_t)x;
    img[idx] = index;
}


void render_frame_gif(ge_GIF *gif, const Particle *particles, unsigned int n, double box_size) {

    memset(gif->frame, 0, FRAME_WIDTH * FRAME_HEIGHT);

    for (unsigned int i = 0; i < n; ++i) {

        int px = (int)(particles[i].x / box_size * (double)(FRAME_WIDTH - 1));
        int py = (int)(particles[i].y / box_size * (double)(FRAME_HEIGHT - 1));
        py = (FRAME_HEIGHT - 1) - py;

        for (int dy = -FRAME_PARTICLE_RADIUS; dy <= FRAME_PARTICLE_RADIUS; ++dy) {
            for (int dx = -FRAME_PARTICLE_RADIUS; dx <= FRAME_PARTICLE_RADIUS; ++dx) {
                if (dx * dx + dy * dy <= FRAME_PARTICLE_RADIUS * FRAME_PARTICLE_RADIUS) {
                    set_pixel(gif->frame, FRAME_WIDTH, FRAME_HEIGHT, px + dx, py + dy, 1);
                }
            }
        }
    }
}
#endif
double random_double(void) {
    return (double)rand() / (double)RAND_MAX;
}

// compute kinetic energy of the system
double compute_ke(const Particle *particles, unsigned int n) {
    double ke = 0.0;
    for (unsigned int i = 0; i < n; ++i) {
        const Particle *p = &particles[i];
        ke += 0.5 * (p->vx * p->vx + p->vy * p->vy);
    }
    return ke;
}

int initialize_particles(Particle *particles, unsigned int n, double box_size, double placement_fraction, unsigned int seed, double temperature) {
    
    srand(seed);
    unsigned int n_side = (unsigned int)ceil(sqrt((double)n));
    double placement_size = placement_fraction * box_size;
    double offset = 0.5 * (box_size - placement_size);
    double delta = placement_size / (double)n_side;

    double mean_vx = 0.0;
    double mean_vy = 0.0;
    // place particles int he middle of the grid with some random jitter and assign random velocities
    for (unsigned int k = 0; k < n; k++) {
        double x0 = offset + (0.5 + (double)(k % n_side)) * delta;
        double y0 = offset + (0.5 + (double)(k / n_side)) * delta;

        particles[k].x = x0 + (2.0 * random_double() - 1.0) * JITTER * delta;
        particles[k].y = y0 + (2.0 * random_double() - 1.0) * JITTER * delta;

        particles[k].vx = 2.0 * random_double() - 1.0;
        particles[k].vy = 2.0 * random_double() - 1.0;
        
        mean_vx += particles[k].vx;
        mean_vy += particles[k].vy;
    }

    mean_vx /= (double)n;
    mean_vy /= (double)n;
    double ke = 0.0;
    // subtract mean velocity to ensure zero net momentum and compute initial kinetic energy
    for (unsigned int k = 0; k < n; k++) {
        particles[k].vx -= mean_vx;
        particles[k].vy -= mean_vy;
        ke += 0.5 * (
            particles[k].vx * particles[k].vx +
            particles[k].vy * particles[k].vy
        );
    }

    double current_temperature = ke / (double)n;
    if (current_temperature <= 0.0) {
        return 0;
    }

    // scale velocities to match the desired initial temperature of the system
    double scale = sqrt(temperature / current_temperature);
    for (unsigned int k = 0; k < n; k++) {
        particles[k].vx *= scale;
        particles[k].vy *= scale;
    }

    return 1;
}

// apply periodic boundary conditions to ensure particles stay within the simulation box
void wrap_positions(Particle *particles, unsigned int n, double box_size) {
    for (unsigned int i = 0; i < n; ++i) {
        Particle *p = &particles[i];
        double wx = fmod(p->x, box_size);
        double wy = fmod(p->y, box_size);

        if (wx < 0.0) {
            wx += box_size;
        }
        if (wy < 0.0) {
            wy += box_size;
        }

        p->x = wx;
        p->y = wy;
    }
}

// shift potential to ensure it goes to zero at the cutoff distance, improving energy conservation
double compute_v_shift(void) {
    return 4.0 * EPSILON * (pow(SIGMA / R_CUT, 12.0) - pow(SIGMA / R_CUT, 6.0));
}

double compute_forces(Particle *particles, unsigned int n, double box_size) {

    for (unsigned int i = 0; i < n; ++i) {
        particles[i].fx = 0.0;
        particles[i].fy = 0.0;
    }
    double pe = 0.0;
    double v_shift = compute_v_shift();
    for (unsigned int i = 0; i < n; ++i) {
        for (unsigned int j = 0; j < n; ++j) {
            if (j == i) {
                continue;
            }
            Particle *pi = &particles[i];
            Particle *pj = &particles[j];
            
            // compute distance between particles with periodic boundary conditions
            double dx = pi->x - pj->x;
            double dy = pi->y - pj->y;

            dx -= box_size * nearbyint(dx / box_size);
            dy -= box_size * nearbyint(dy / box_size);

            // compute Lennard-Jones force and potential energy contribution if particles are within the cutoff distance
            double r = sqrt(dx * dx + dy * dy);
            if (r >= R_CUT || r == 0.0) {
                continue;
            }
            double sr = SIGMA / r;

            double fij = 24.0 * EPSILON * (2.0 * pow(sr, 12.0) - pow(sr, 6.0)) / r;
            double fx = fij * dx / r;
            double fy = fij * dy / r;

            pi->fx += fx;
            pi->fy += fy;

            double vij = 4.0 * EPSILON * (pow(sr, 12.0) - pow(sr, 6.0)) - v_shift;
            pe += 0.5 * vij;
        }
    }

    return pe;
}

double leapfrog_step(Particle *particles, unsigned int n, double box_size) {
    // update velocities by half a time step, then update positions by a full time step, 
    //and finally update velocities by another half time step to complete the leapfrog integration step
    for (unsigned int i = 0; i < n; ++i) {
        Particle *p = &particles[i];
        p->vx += 0.5 * DT * p->fx;
        p->vy += 0.5 * DT * p->fy;

        p->x += DT * p->vx;
        p->y += DT * p->vy;
    }

    wrap_positions(particles, n, box_size);

    double pe = compute_forces(particles, n, box_size);

    for (unsigned int i = 0; i < n; ++i) {
        Particle *p = &particles[i];
        p->vx += 0.5 * DT * p->fx;
        p->vy += 0.5 * DT * p->fy;
    }

    return pe;
}

SimulationResult run_simulation(Particle *particles, unsigned int n, unsigned int nsteps, double box_size, int log_steps) {
    
    SimulationResult out;
    out.start_potential= compute_forces(particles, n, box_size);
    out.start_kinetic = compute_ke(particles, n);
    out.start_total = out.start_kinetic + out.start_potential;

    
#if GENERATE_GIF
    ge_GIF *gif = NULL;

    gif = ge_new_gif(GIF_FILE, (uint16_t)FRAME_WIDTH, (uint16_t)FRAME_HEIGHT, palette, 8, -1, 0);
    if (!gif) {
        fprintf(stderr, "Warning: failed to create GIF output %s\n", GIF_FILE);
    } else {
        render_frame_gif(gif, particles, n, box_size);
        ge_add_frame(gif, FRAME_DELAY);
    }
#endif

    for (unsigned int step = 0; step < nsteps; step++) {
        out.final_potential = leapfrog_step(particles, n, box_size);
        out.final_kinetic = compute_ke(particles, n);
        out.final_total = out.final_kinetic + out.final_potential;
        if (log_steps) {
            printf(
                "step=%6u  KE=%12.6f  PE=%12.6f  E=%12.6f\n",
                step,
                out.final_kinetic,
                out.final_potential,
                out.final_total
            );
        }

    
#if GENERATE_GIF
        if (gif && FRAME_EVERY > 0 && (step + 1) % FRAME_EVERY == 0) {
            render_frame_gif(gif, particles, n, box_size);
            ge_add_frame(gif, FRAME_DELAY);
        }
#endif
    }

#if GENERATE_GIF
    if (gif) {
        ge_close_gif(gif);
    }
#endif

    out.n = n;
    out.particles = particles;
    return out;
}