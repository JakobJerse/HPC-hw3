/*
 * lennard-jones.cu  –  CUDA parallel 2-D Lennard-Jones simulation
 *
 * Design summary
 * --------------
 *  Memory layout
 *    Particles arrive as AoS (struct Particle).  They are converted to SoA
 *    (six separate double* arrays) before upload so that every kernel reads
 *    and writes coalesced memory.
 *
 *  GPU residency
 *    All six arrays live on the device for the entire simulation loop.
 *    Data crosses PCIe exactly once at the start (upload) and once at the
 *    end (download).  The only additional host<->device traffic per step is
 *    two double scalars when energy logging is enabled.
 *
 *  Kernels
 *    compute_forces_kernel  - 1 thread per particle i; inner loop over all j.
 *                             PE is accumulated with atomicAdd into a single
 *                             device double so no separate reduction pass is
 *                             needed.
 *    leapfrog_half_kick     - 1 thread per particle, v += 0.5*dt*f.
 *    leapfrog_drift         - 1 thread per particle, x += dt*v + PBC wrap.
 *    compute_ke_kernel      - 1 thread per particle, KE via atomicAdd.
 *
 *  Block-size auto-tuning
 *    autotune_block_size() benchmarks {32,64,128,256,512,1024} on the force
 *    kernel (dominant cost) and selects the fastest for the full run.
 *
 *  Reference functions
 *    initialize_particles, wrap_positions, compute_v_shift, compute_forces,
 *    and leapfrog_step are kept identical to the sequential reference so the
 *    header contract is fully satisfied.
 */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <cuda_runtime.h>
#include <cuda.h>

#include "lennard-jones.h"

/* ------------------------------------------------------------------ */
/*  CUDA error-checking macro                                          */
/* ------------------------------------------------------------------ */
#define CHECK(call)                                                        \
    do {                                                                   \
        cudaError_t _err = (call);                                         \
        if (_err != cudaSuccess) {                                         \
            fprintf(stderr, "CUDA error at %s:%d - %s\n",                 \
                    __FILE__, __LINE__, cudaGetErrorString(_err));         \
            exit(EXIT_FAILURE);                                            \
        }                                                                  \
    } while (0)

/* ------------------------------------------------------------------ */
/*  Device constants (read from fast constant cache)                   */
/* ------------------------------------------------------------------ */
__constant__ double c_dt      = DT;
__constant__ double c_epsilon = EPSILON;
__constant__ double c_sigma   = SIGMA;
__constant__ double c_r_cut   = R_CUT;

/* ================================================================== */
/*  KERNELS                                                            */
/* ================================================================== */

/*
 * compute_forces_kernel
 *
 * One thread handles one particle i.  It iterates over every other particle
 * j, applies the minimum-image convention, and accumulates:
 *   - the net force components (fx[i], fy[i])
 *   - 0.5 * V_shifted(r_ij) into the global PE scalar via atomicAdd
 *
 * fij_over_r2 = 24*eps/r^2 * [2*(sigma/r)^12 - (sigma/r)^6]
 * so force components are simply fij_over_r2 * dx and fij_over_r2 * dy,
 * avoiding a sqrt in the force path (only r2 is needed).
 */
__global__ void compute_forces_kernel(
        const double * __restrict__ x,
        const double * __restrict__ y,
        double       * __restrict__ fx,
        double       * __restrict__ fy,
        double       * __restrict__ pe_out,
        unsigned int  n,
        double        box_size,
        double        v_shift)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    const double half_L = 0.5 * box_size;
    const double r_cut2 = c_r_cut * c_r_cut;
    const double eps    = c_epsilon;
    const double sig2   = c_sigma * c_sigma;

    double xi = x[i], yi = y[i];
    double fxi = 0.0, fyi = 0.0;
    double pe_i = 0.0;

    for (unsigned int j = 0; j < n; ++j) {
        if (j == i) continue;

        double dx = xi - x[j];
        double dy = yi - y[j];

        /* minimum-image convention: fold into (-L/2, L/2] */
        if      (dx >  half_L) dx -= box_size;
        else if (dx < -half_L) dx += box_size;
        if      (dy >  half_L) dy -= box_size;
        else if (dy < -half_L) dy += box_size;

        double r2 = dx * dx + dy * dy;
        if (r2 >= r_cut2 || r2 == 0.0) continue;

        /* (sigma/r)^2, ^6, ^12 computed without sqrt */
        double r2inv = 1.0 / r2;
        double sr2   = sig2 * r2inv;
        double sr6   = sr2 * sr2 * sr2;
        double sr12  = sr6 * sr6;

        /* force: F/r^2 * (dx, dy) */
        double fij_r2 = 24.0 * eps * (2.0 * sr12 - sr6) * r2inv;
        fxi  += fij_r2 * dx;
        fyi  += fij_r2 * dy;

        /* shifted potential with 0.5 for double-counting */
        pe_i += 0.5 * (4.0 * eps * (sr12 - sr6) - v_shift);
    }

    fx[i] = fxi;
    fy[i] = fyi;
    atomicAdd(pe_out, pe_i);
}

/*
 * leapfrog_half_kick
 * v(t +/- dt/2) = v(t) + 0.5*dt * f(t)
 */
__global__ void leapfrog_half_kick(
        double       * __restrict__ vx,
        double       * __restrict__ vy,
        const double * __restrict__ fx,
        const double * __restrict__ fy,
        unsigned int  n)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    double hdt = 0.5 * c_dt;
    vx[i] += hdt * fx[i];
    vy[i] += hdt * fy[i];
}

/*
 * leapfrog_drift
 * x(t+dt) = x(t) + dt*v(t+dt/2), with periodic boundary conditions.
 */
__global__ void leapfrog_drift(
        double       * __restrict__ x,
        double       * __restrict__ y,
        const double * __restrict__ vx,
        const double * __restrict__ vy,
        unsigned int  n,
        double        box_size)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    double xi = x[i] + c_dt * vx[i];
    double yi = y[i] + c_dt * vy[i];

    /* wrap into [0, box_size) */
    xi = fmod(xi, box_size); if (xi < 0.0) xi += box_size;
    yi = fmod(yi, box_size); if (yi < 0.0) yi += box_size;

    x[i] = xi;
    y[i] = yi;
}

/*
 * compute_ke_kernel
 * Accumulates KE = sum_i 0.5*(vx^2 + vy^2) via atomicAdd.
 */
__global__ void compute_ke_kernel(
        const double * __restrict__ vx,
        const double * __restrict__ vy,
        double       * __restrict__ ke_out,
        unsigned int  n)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    atomicAdd(ke_out, 0.5 * (vx[i] * vx[i] + vy[i] * vy[i]));
}

/* ================================================================== */
/*  Block-size auto-tuner                                              */
/* ================================================================== */

static int autotune_block_size(unsigned int n, double box_size, double v_shift)
{
    printf("--- Auto-tuning CUDA block size ---\n");

    const int candidates[] = {32, 64, 128, 256, 512, 1024};
    const int nc = (int)(sizeof(candidates) / sizeof(candidates[0]));

    double *tx, *ty, *tfx, *tfy, *tpe;
    CHECK(cudaMalloc(&tx,  n * sizeof(double)));
    CHECK(cudaMalloc(&ty,  n * sizeof(double)));
    CHECK(cudaMalloc(&tfx, n * sizeof(double)));
    CHECK(cudaMalloc(&tfy, n * sizeof(double)));
    CHECK(cudaMalloc(&tpe, sizeof(double)));
    CHECK(cudaMemset(tx,  0, n * sizeof(double)));
    CHECK(cudaMemset(ty,  0, n * sizeof(double)));
    CHECK(cudaMemset(tpe, 0, sizeof(double)));

    int   best_bs = 128;
    float best_t  = 1e9f;

    for (int c = 0; c < nc; ++c) {
        int  bs  = candidates[c];
        dim3 blk(bs);
        dim3 grd((n + bs - 1) / bs);

        /* warm-up */
        for (int w = 0; w < 3; ++w)
            compute_forces_kernel<<<grd, blk>>>(tx, ty, tfx, tfy, tpe, n, box_size, v_shift);
        cudaDeviceSynchronize();

        cudaEvent_t e0, e1;
        CHECK(cudaEventCreate(&e0));
        CHECK(cudaEventCreate(&e1));
        CHECK(cudaEventRecord(e0));
        for (int r = 0; r < 5; ++r)
            compute_forces_kernel<<<grd, blk>>>(tx, ty, tfx, tfy, tpe, n, box_size, v_shift);
        CHECK(cudaEventRecord(e1));
        CHECK(cudaEventSynchronize(e1));

        float ms = 0.0f;
        CHECK(cudaEventElapsedTime(&ms, e0, e1));
        float avg = ms / 5.0f;
        printf("  block %4d : %.4f ms/call\n", bs, avg);
        if (avg < best_t) { best_t = avg; best_bs = bs; }

        CHECK(cudaEventDestroy(e0));
        CHECK(cudaEventDestroy(e1));
    }

    printf(">>> Best block size: %d  (%.4f ms/call)\n\n", best_bs, best_t);

    CHECK(cudaFree(tx));  CHECK(cudaFree(ty));
    CHECK(cudaFree(tfx)); CHECK(cudaFree(tfy));
    CHECK(cudaFree(tpe));
    return best_bs;
}

/* ================================================================== */
/*  Reference sequential functions (required by lennard-jones.h)      */
/* ================================================================== */

double random_double(void) {
    return (double)rand() / (double)RAND_MAX;
}

int initialize_particles(Particle *particles, unsigned int n, double box_size,
                         double placement_fraction, unsigned int seed, double temperature)
{
    srand(seed);
    unsigned int n_side  = (unsigned int)ceil(sqrt((double)n));
    double placement_sz  = placement_fraction * box_size;
    double offset        = 0.5 * (box_size - placement_sz);
    double delta         = placement_sz / (double)n_side;

    double mean_vx = 0.0, mean_vy = 0.0;
    for (unsigned int k = 0; k < n; k++) {
        double x0 = offset + (0.5 + (double)(k % n_side)) * delta;
        double y0 = offset + (0.5 + (double)(k / n_side)) * delta;
        particles[k].x  = x0 + (2.0 * random_double() - 1.0) * JITTER * delta;
        particles[k].y  = y0 + (2.0 * random_double() - 1.0) * JITTER * delta;
        particles[k].vx = 2.0 * random_double() - 1.0;
        particles[k].vy = 2.0 * random_double() - 1.0;
        mean_vx += particles[k].vx;
        mean_vy += particles[k].vy;
    }
    mean_vx /= n; mean_vy /= n;

    double ke = 0.0;
    for (unsigned int k = 0; k < n; k++) {
        particles[k].vx -= mean_vx;
        particles[k].vy -= mean_vy;
        ke += 0.5 * (particles[k].vx * particles[k].vx +
                     particles[k].vy * particles[k].vy);
    }
    double T_cur = ke / (double)n;
    if (T_cur <= 0.0) return 0;
    double scale = sqrt(temperature / T_cur);
    for (unsigned int k = 0; k < n; k++) {
        particles[k].vx *= scale;
        particles[k].vy *= scale;
    }
    return 1;
}

void wrap_positions(Particle *particles, unsigned int n, double box_size) {
    for (unsigned int i = 0; i < n; ++i) {
        double wx = fmod(particles[i].x, box_size);
        double wy = fmod(particles[i].y, box_size);
        if (wx < 0.0) wx += box_size;
        if (wy < 0.0) wy += box_size;
        particles[i].x = wx;
        particles[i].y = wy;
    }
}

double compute_v_shift(void) {
    double sr = SIGMA / R_CUT;
    return 4.0 * EPSILON * (pow(sr, 12.0) - pow(sr, 6.0));
}

double compute_ke(const Particle *particles, unsigned int n) {
    double ke = 0.0;
    for (unsigned int i = 0; i < n; ++i)
        ke += 0.5 * (particles[i].vx * particles[i].vx +
                     particles[i].vy * particles[i].vy);
    return ke;
}

double compute_forces(Particle *particles, unsigned int n, double box_size) {
    for (unsigned int i = 0; i < n; ++i) {
        particles[i].fx = 0.0;
        particles[i].fy = 0.0;
    }
    double pe      = 0.0;
    double v_shift = compute_v_shift();
    for (unsigned int i = 0; i < n; ++i) {
        for (unsigned int j = 0; j < n; ++j) {
            if (j == i) continue;
            double dx = particles[i].x - particles[j].x;
            double dy = particles[i].y - particles[j].y;
            dx -= box_size * nearbyint(dx / box_size);
            dy -= box_size * nearbyint(dy / box_size);
            double r = sqrt(dx * dx + dy * dy);
            if (r >= R_CUT || r == 0.0) continue;
            double sr  = SIGMA / r;
            double fij = 24.0 * EPSILON * (2.0 * pow(sr, 12.0) - pow(sr, 6.0)) / r;
            particles[i].fx += fij * dx / r;
            particles[i].fy += fij * dy / r;
            pe += 0.5 * (4.0 * EPSILON * (pow(sr, 12.0) - pow(sr, 6.0)) - v_shift);
        }
    }
    return pe;
}

double leapfrog_step(Particle *particles, unsigned int n, double box_size) {
    for (unsigned int i = 0; i < n; ++i) {
        particles[i].vx += 0.5 * DT * particles[i].fx;
        particles[i].vy += 0.5 * DT * particles[i].fy;
        particles[i].x  += DT * particles[i].vx;
        particles[i].y  += DT * particles[i].vy;
    }
    wrap_positions(particles, n, box_size);
    double pe = compute_forces(particles, n, box_size);
    for (unsigned int i = 0; i < n; ++i) {
        particles[i].vx += 0.5 * DT * particles[i].fx;
        particles[i].vy += 0.5 * DT * particles[i].fy;
    }
    return pe;
}

/* ================================================================== */
/*  run_simulation  -  GPU parallel main entry point                  */
/* ================================================================== */
SimulationResult run_simulation(Particle *particles, unsigned int n,
                                unsigned int nsteps, double box_size, int log_steps)
{
    SimulationResult out;
    memset(&out, 0, sizeof(out));

    const size_t sz      = n * sizeof(double);
    const double v_shift = compute_v_shift();
    const double zero    = 0.0;

    /* ---- flatten AoS -> SoA ---- */
    double *h_x  = (double *)malloc(sz);
    double *h_y  = (double *)malloc(sz);
    double *h_vx = (double *)malloc(sz);
    double *h_vy = (double *)malloc(sz);
    double *h_fx = (double *)malloc(sz);
    double *h_fy = (double *)malloc(sz);

    if (!h_x || !h_y || !h_vx || !h_vy || !h_fx || !h_fy) {
        fprintf(stderr, "run_simulation: host malloc failed\n");
        exit(EXIT_FAILURE);
    }

    for (unsigned int i = 0; i < n; ++i) {
        h_x[i]  = particles[i].x;
        h_y[i]  = particles[i].y;
        h_vx[i] = particles[i].vx;
        h_vy[i] = particles[i].vy;
        h_fx[i] = 0.0;
        h_fy[i] = 0.0;
    }

    /* ---- device allocation ---- */
    double *d_x, *d_y, *d_vx, *d_vy, *d_fx, *d_fy, *d_pe, *d_ke;
    CHECK(cudaMalloc(&d_x,  sz)); CHECK(cudaMalloc(&d_y,  sz));
    CHECK(cudaMalloc(&d_vx, sz)); CHECK(cudaMalloc(&d_vy, sz));
    CHECK(cudaMalloc(&d_fx, sz)); CHECK(cudaMalloc(&d_fy, sz));
    CHECK(cudaMalloc(&d_pe, sizeof(double)));
    CHECK(cudaMalloc(&d_ke, sizeof(double)));

    /* ---- upload (included in host wall-clock per assignment spec) ---- */
    CHECK(cudaMemcpy(d_x,  h_x,  sz, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_y,  h_y,  sz, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_vx, h_vx, sz, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_vy, h_vy, sz, cudaMemcpyHostToDevice));
    CHECK(cudaMemset(d_fx, 0, sz));
    CHECK(cudaMemset(d_fy, 0, sz));

    /* ---- auto-tune block size ---- */
    int  bs  = autotune_block_size(n, box_size, v_shift);
    dim3 blk(bs);
    dim3 grd((n + bs - 1) / bs);

    /* ---- initial forces + starting energies ---- */
    CHECK(cudaMemcpy(d_pe, &zero, sizeof(double), cudaMemcpyHostToDevice));
    compute_forces_kernel<<<grd, blk>>>(d_x, d_y, d_fx, d_fy, d_pe, n, box_size, v_shift);
    CHECK(cudaGetLastError());

    CHECK(cudaMemcpy(d_ke, &zero, sizeof(double), cudaMemcpyHostToDevice));
    compute_ke_kernel<<<grd, blk>>>(d_vx, d_vy, d_ke, n);
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());

    double start_pe, start_ke;
    CHECK(cudaMemcpy(&start_pe, d_pe, sizeof(double), cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(&start_ke, d_ke, sizeof(double), cudaMemcpyDeviceToHost));
    out.start_potential = start_pe;
    out.start_kinetic   = start_ke;
    out.start_total     = start_ke + start_pe;

    /* ================================================================
     *  Leapfrog loop
     *  Step:  half-kick -> drift+PBC -> forces -> half-kick
     * ============================================================= */
    double final_pe = start_pe;
    double final_ke = start_ke;

    for (unsigned int step = 0; step < nsteps; ++step) {

        /* 1 - half velocity kick */
        leapfrog_half_kick<<<grd, blk>>>(d_vx, d_vy, d_fx, d_fy, n);
        CHECK(cudaGetLastError());

        /* 2 - position update + periodic boundary conditions */
        leapfrog_drift<<<grd, blk>>>(d_x, d_y, d_vx, d_vy, n, box_size);
        CHECK(cudaGetLastError());

        /* 3 - force recompute; zero PE accumulator first */
        CHECK(cudaMemcpy(d_pe, &zero, sizeof(double), cudaMemcpyHostToDevice));
        compute_forces_kernel<<<grd, blk>>>(d_x, d_y, d_fx, d_fy, d_pe, n, box_size, v_shift);
        CHECK(cudaGetLastError());

        /* 4 - second half kick */
        leapfrog_half_kick<<<grd, blk>>>(d_vx, d_vy, d_fx, d_fy, n);
        CHECK(cudaGetLastError());

        /* synchronise before any host reads */
        if (log_steps || step == nsteps - 1)
            CHECK(cudaDeviceSynchronize());

        /* optional per-step energy output */
        if (log_steps) {
            CHECK(cudaMemcpy(&final_pe, d_pe, sizeof(double), cudaMemcpyDeviceToHost));

            CHECK(cudaMemcpy(d_ke, &zero, sizeof(double), cudaMemcpyHostToDevice));
            compute_ke_kernel<<<grd, blk>>>(d_vx, d_vy, d_ke, n);
            CHECK(cudaDeviceSynchronize());
            CHECK(cudaMemcpy(&final_ke, d_ke, sizeof(double), cudaMemcpyDeviceToHost));

            printf("step=%6u  KE=%12.6f  PE=%12.6f  E=%12.6f\n",
                   step, final_ke, final_pe, final_ke + final_pe);
        }
    }

    /* collect final energies when not logging */
    if (!log_steps) {
        CHECK(cudaMemcpy(&final_pe, d_pe, sizeof(double), cudaMemcpyDeviceToHost));

        CHECK(cudaMemcpy(d_ke, &zero, sizeof(double), cudaMemcpyHostToDevice));
        compute_ke_kernel<<<grd, blk>>>(d_vx, d_vy, d_ke, n);
        CHECK(cudaDeviceSynchronize());
        CHECK(cudaMemcpy(&final_ke, d_ke, sizeof(double), cudaMemcpyDeviceToHost));
    }

    out.final_potential = final_pe;
    out.final_kinetic   = final_ke;
    out.final_total     = final_ke + final_pe;

    /* ---- download final state ---- */
    CHECK(cudaMemcpy(h_x,  d_x,  sz, cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(h_y,  d_y,  sz, cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(h_vx, d_vx, sz, cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(h_vy, d_vy, sz, cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(h_fx, d_fx, sz, cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(h_fy, d_fy, sz, cudaMemcpyDeviceToHost));

    /* write back SoA -> AoS */
    for (unsigned int i = 0; i < n; ++i) {
        particles[i].x  = h_x[i];
        particles[i].y  = h_y[i];
        particles[i].vx = h_vx[i];
        particles[i].vy = h_vy[i];
        particles[i].fx = h_fx[i];
        particles[i].fy = h_fy[i];
    }

    out.n         = n;
    out.particles = particles;

    /* ---- cleanup ---- */
    CHECK(cudaFree(d_x));  CHECK(cudaFree(d_y));
    CHECK(cudaFree(d_vx)); CHECK(cudaFree(d_vy));
    CHECK(cudaFree(d_fx)); CHECK(cudaFree(d_fy));
    CHECK(cudaFree(d_pe)); CHECK(cudaFree(d_ke));

    free(h_x); free(h_y); free(h_vx); free(h_vy); free(h_fx); free(h_fy);

    return out;
}
