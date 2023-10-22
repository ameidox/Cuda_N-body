
#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#include<math.h>
#include <stdio.h>
#include "main.h"



__global__ void computeForcesKernel(Body* bodies, int N) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    float G = 0.0075f;
    float softning = 1.5f;
    float Fx = 0.0f, Fy = 0.0f;

    for (int j = 0; j < N; j++) {
        if (idx != j) {
            float dx = bodies[j].x - bodies[idx].x;
            float dy = bodies[j].y - bodies[idx].y;

            float dist = sqrt(dx * dx + dy * dy) + 1e-6;  // small term to prevent division by zero
            float softenedDist = sqrt(dist * dist + softning * softning);
            float F = (G * bodies[idx].mass * bodies[j].mass) / softenedDist;

            Fx += F * dx;
            Fy += F * dy;
        }
    }

    bodies[idx].vx += Fx / bodies[idx].mass;
    bodies[idx].vy += Fy / bodies[idx].mass;
}


__global__ void updatePositionsKernel(Body* bodies, float dt, int N) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    bodies[idx].x += bodies[idx].vx * dt;
    bodies[idx].y += bodies[idx].vy * dt;
}

extern "C" void runSimulationStep(Body * dev_bodies, float dt, int N) {
    int blockSize = 1024;
    int numBlocks = (N + blockSize - 1) / blockSize;

    computeForcesKernel << <numBlocks, blockSize >> > (dev_bodies, N);
    updatePositionsKernel << <numBlocks, blockSize >> > (dev_bodies, dt, N);

    cudaDeviceSynchronize();
}