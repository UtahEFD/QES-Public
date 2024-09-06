#include "CUDA_concentration.cuh"
#include "CUDA_particle_partition.cuh"

__global__ void collect(int length, particle_array d_particle_list, int *pBox, const ConcentrationParam param)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < length) {
    if (d_particle_list.state[idx] == ACTIVE) {
      // x-direction
      int i = floor((d_particle_list.pos[idx]._1 - param.lbndx) / (param.dx + 1e-9));
      // y-direction
      int j = floor((d_particle_list.pos[idx]._2 - param.lbndy) / (param.dy + 1e-9));
      // z-direction
      int k = floor((d_particle_list.pos[idx]._3 - param.lbndz) / (param.dz + 1e-9));

      if (i >= 0 && i <= param.nx - 1 && j >= 0 && j <= param.ny - 1 && k >= 0 && k <= param.nz - 1) {
        int id = k * param.ny * param.nx + j * param.nx + i;
        atomicAdd(&pBox[id], 1);
        // conc[id] = conc[id] + par.m * par.wdecay * timeStep;
      }
    }
  }
}
