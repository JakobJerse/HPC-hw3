# Molecular dynamics: 2D Lennard-Jones Simulation

**Authors:** Uroš Lotrič, Davor Sluga  
**Date:** April 2026

## Introduction

Molecular dynamics simulation is a widely used computational technique for studying the physical behaviour of many-particle systems. By numerically integrating the equations of motion for each particle under the influence of forces, molecular dynamics allows us to observe phenomena such as self-organisation that emerge from simple pairwise interactions.

In this assignment, we consider a 2D $N$-body system of particles interacting via the Lennard-Jones potential, a classical model for noble gases and other simple fluids. The Lennard-Jones potential captures both the short-range repulsion (due to overlapping electron clouds) and the longer-range attraction (van der Waals forces) between a pair of particles at a distance $r$

$$
V(r) = 4\varepsilon \left[ \left(\frac{\sigma}{r}\right)^{12} - \left(\frac{\sigma}{r}\right)^{6} \right] \quad,
$$

where $\varepsilon$ is the depth of the potential well and $\sigma$ is the finite distance, related to the atom size, at which the potential is zero. We use reduced units throughout the simulation, $m = \varepsilon = \sigma = 1$, which simplifies the equations and improves numerical conditioning.

Force equals the gradient of the potential between a pair of particles, $\mathbf{F} = -\nabla V(r)$. Suppose particles $i$ and $j$ are sitting at positions $r_i$ and $r_j$, thus being displaced for vector $r_{ij} = r_i - r_j$. Taking the displacement vector magnitude $r_{ij} = |r_{ij}|$ and its projections to the Cartesian coordinate system $x_{ij}$ and $y_{ij}$, the force of particle $j$ on particle $i$ equals

$$
\mathbf{F}_{ij} = \left(F_{xij}, F_{yij}\right) = F(r_{ij})\left( \frac{x_{ij}}{r_{ij}}, \frac{y_{ij}}{r_{ij}} \right)
$$

where

$$
F(r) = 24\frac{\varepsilon}{r}
\left[ 2\left(\frac{\sigma}{r}\right)^{12} - \left(\frac{\sigma}{r}\right)^{6} \right]
$$

stands for the force magnitude.

The motion of each particle is governed by the forces exerted by all other particles. The net force acting on particle $i$ thus becomes

$$
\mathbf{F}_{i} = \sum_{j \neq i} \mathbf{F}_{ij} \quad,
$$

leading to the $\mathcal{O}(N^{2})$ pairwise force computation in each simulation step.

From the net force, the acceleration of each particle follows directly from Newton's second law, $\mathbf{a}_i = \mathbf{F}_i / m$, from which the velocity and, subsequently, the position of each particle can be obtained by integration. The system at time $t_k$ evolves over a time step $\Delta t$ to the state

$$
\mathbf{v}_{i}(t_k + \Delta t) = \mathbf{v}_{i}(t_k) + \int_{t_k}^{t_k + \Delta t} \mathbf{a}_i(t) dt
$$

$$
\mathbf{r}_{i}(t_k + \Delta t) = \mathbf{r}_{i}(t_k) + \int_{t_k}^{t_k + \Delta t} \mathbf{v}_i(t) dt \quad .
$$

## Simulation

A reference C implementation along with corresponding build and run scripts is provided in the [repository](src/lennard-jones/). The main components are described below.

### Particle Initialisation

Particles are placed in the shape of a regular 2D lattice with optional random jitter in the centre of the simulation box, with a side length

$$
L = {{4}\over{3}}\sqrt{N\over\rho},
$$

where $\rho$ is the target reduced number density. Velocities are randomly generated, shifted to remove centre-of-mass drift, and rescaled to match the equation:

$$
E_k = \tfrac{1}{2}m\sum_i |\mathbf{v}_i|^2 = NT \quad,
$$

where $T$ represents the target reduced temperature of the system.

### Force computation

At every simulation step, all pairwise forces are recomputed. To avoid infinite-range interactions, a cut-off radius $r_\text{cut} = 2.5 \sigma$ is applied: pairs with $r \geq r_\text{cut}$ contribute nothing. To eliminate a discontinuity in the potential at the cut-off radius and preserve the total energy, the Lennard-Jones potential is replaced with the shifted version:

$$
V_\text{shifted}(r) = V(r) - V(r_\text{cut}) \quad .
$$

### Periodic boundary conditions

The simulation domain is a square box with periodic boundaries - particles that leave the box re-enter from the opposite side. Since the simulation domain is finite but the system mimics its infinity, each particle has infinitely many periodic images. Thus, when computing the displacement vector $\mathbf{r}_{ij}$, we always select the image of particle $j$ that is closest to particle $i$. In practice, this is achieved by wrapping each component of the displacement vector into the interval $(-L/2, L/2]$,

$$
\mathbf{r}_{ij} = (\mathbf{r}_i - \mathbf{r}_j) - L \cdot \text{round}\left(\frac{\mathbf{r}_i - \mathbf{r}_j}{L}\right) \quad .
$$

### Time integration

The simulation advances using the Leapfrog scheme, which is time-reversible and conserves energy well over long runs. One simulation step of length $\Delta t$ consists of the following equations:

$$
\mathbf{v}_i\left(t + \tfrac{\Delta t}{2}\right) = \mathbf{v}_i(t) + \tfrac{1}{2} \mathbf{a}_i(t) \Delta t \quad,
$$

$$
\mathbf{r}_i(t + \Delta t) = \mathbf{r}_i(t) + \mathbf{v}_i\left(t + \tfrac{\Delta t}{2}\right)\Delta t \quad,
$$

$$
\mathbf{a}_i(t+\Delta t) = \mathbf{F}_i(t+\Delta t) / m \quad,
$$

$$
\mathbf{v}_i(t + \Delta t) = \mathbf{v}_i\left(t + \tfrac{\Delta t}{2}\right) + \tfrac{1}{2} \mathbf{a}_i(t + \Delta t)\Delta t \quad .
$$

Besides, the kinetic energy and potential energy of the system,

$$
E_k=\sum_i \frac{1}{2}m|\mathbf{v}_i|^2 \quad \mathrm{and} \quad E_p = \sum_i\sum_{j\neq i} \frac{1}{2}V_\mathrm{shifted}(r_{ij}) \quad,
$$

are computed at each simulation step to verify the conservation of the total energy

$$
E = E_k + E_p \quad .
$$

The below figures show the behavior of the system.

![Simulation.](img/lennard-jones.gif)

## Assignment

Implement a parallel Lennard-Jones simulation in C/C++ using CUDA based on the [reference implementation](src/lennard-jones/). The algorithm should work for an arbitrary number of particles and steps. Note that the C code is poorly optimised; feel free to improve it. It also includes optional code to generate animations, which you can use to examine the results.  

### Reference Code organisation

- `Makefile` -> Project build rules.
- `run-lj.sh` -> Sbatch script to acquire resources on the Arnes cluster, build and run the simulator.
- `src/`
  - `main.c` -> Main project file.
  - `lennard-jones.cu` -> Lennard-Jones simulation code.
  - `gifenc.c` -> Code for generating gif animations; taken from [here](https://github.com/lecram/gifenc).

### Basic tasks (for grades 6-8)

- Parallelise the algorithm using CUDA as efficiently as possible. Avoid unnecessary memory transfers between the host and the device. When dividing the workload, find the optimal number of threads and thread block size. Allow the option to track the system's energy at each step.
- Measure the execution time of the algorithm on the Arnes cluster for different numbers of particles. Use the particle numbers: 1000, 2000, 4000, and 8000. Benchmark the algorithm on 5000 simulation steps. When measuring time, the data transfers to and from the GPU must also be included.
- Compute the speed-up $S=t_s/t_p$ of your algorithm for each particle number; $t_s$ is the execution time of the sequential algorithm on the CPU, and $t_p$ is the execution time of the parallel algorithm on the GPU. Run the algorithm multiple times (at least 5) and average the measurements. Note that the base code for high particle counts takes a long time to run. You can do only one run in such a case.
- Visualise the resulting final state (don't put it in the report, but store it separately). You can even create an animation that shows how the system behaves over time. Do not include the time required to produce the animation in the time measurements or speed-ups.
- Write a short report (1-2 pages) summarising your solution and presenting the measurements performed on the cluster. The main focus should be on presenting and explaining the time measurements and speed-ups.
- Hand in your code and the report (one submission per pair) to ucilnica through the appropriate form by the specified deadline (**5. 5. 2026**) and defend your code and report during labs.

### Bonus tasks (for grades 9-10)

- Parallelise (with OpenMP) the provided sequential code. Use your improved code as the baseline (with the optimal number of cores) for the measurements when computing speed-ups against the GPU.
- Improve the reference code: note that due to Newton's 3rd law, for every action (force) in nature, there is an equal and opposite reaction. When one object exerts a force on a second object, the second object simultaneously exerts a force equal in magnitude and opposite in direction on the first, thus you only need to compute half of the interactions: $N(N-1)/2$.
- To further reduce the number of interactions computed, think about keeping a record of particle neighbourhoods. Remember, particles which are separated by more than $r_\text{cut}$ don't affect each other.
- Think about splitting the work between GPU threads, try to find a solution that utilises the GPU best. One thread per particle may not be the best option.
- Experiment with and optimise how the particles are stored in memory, utilise shared memory where you see fit.
- Utilise two GPUs to perform the simulation; split the work between them evenly and exchange the data as necessary.

## HPC Challenge

Produce a highly optimised implementation of the Lennard-Jones simulation code for the 3D case, which generates results aligned with the reference code.
Prepare a C/C++ solution that supports graphics accelerators using CUDA. You are encouraged to combine CUDA with shared-memory systems using the OpenMP library to optimise execution times.
Template code and run scripts are provided on the [repository](src/lennard-jones-challenge). Your task is to implement the `run_simulation` function in the file `lennard-jones.cu`. The organisers will only consider the solutions built and executed using the script `run-lj.sh`. Each solution will be tested in an isolated environment consisting of one 12-core node, with two Nvidia V100 GPUS. Submit the solutions through the [course web page](https://ucilnica.fri.uni-lj.si/mod/assign/view.php?id=55667).

When submitting your solution, **DO NOT SUBMIT** the file `src/main.c`, as it will be ignored. During benchmarking, we will provide `main.c`, which will call the `run_simulation` function defined in `lennard-jones.h`. Use the provided `main.c` as a reference on how the benchmarks will be performed.

- We will benchmark on multiple particle system configurations (1000+ particles, 1000+ simulation steps). Energy logging will be disabled.
- You can modify the `Makefile` and add additional files as you see fit, as long as the project compiles.
- The sbatch script `run-lj.sh` includes the full allocations of resources, as it will be used during benchmarking your solution.
- You can modify the `run-lj.sh` to change the allocation parameters if needed and submit it as part of your solution. You are limited to one V100 node.
- The function `run_simulation` should return the starting and the final state of the system as defined in the struct `SimulationResult`. We will use it to verify the correctness of your solution. We will treat the solution as correct if the returned values (start energies and final energies) are close enough to the reference solution for a given set of parameters.

### Rules of the game

- You can work in pairs or on your own; any suspicion of plagiarism leads to disqualification. Write the authors of the solution in a `authors.txt` file submitted alongside the code.
- The challenge is open until **Sunday, May 31**.
- The organisers will evaluate the submitted solutions for correctness and performance in terms of running time. Tests will be performed on the Arnes cluster in an isolated environment.
- The solutions will be ranked according to the achieved performance.
- Rewards:
  - Winner gets a bonus of 10/10 perfectly answered questions on the written exam.
  - Second place gets a bonus of 7/10 perfectly answered questions on the written exam.
  - Third place gets a bonus of 5/10 perfectly answered questions on the written exam.
  - The organisers reserve the right to reward other solutions with perfectly answered questions on the written exam at their discretion.
- Notes:
  - The candidate can use the rewards only on the first written exam he takes.
  - The final grade averages the written exam and assignment grades; the written exam grade determines rounding to a whole number.
