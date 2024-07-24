
#include "CUDA_interpolation_testkernel.h"

#include "util/VectorMath_CUDA.cuh"

#include "plume/IDGenerator.h"

// #include "CUDA_GLE_Solver.cuh"
//  #include "Particle.cuh"

#define PBSTR "||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||"
#define PBWIDTH 60

__device__ __managed__ QESgrid qes_grid;

struct interpWeight
{
  int ii;// nearest cell index to the left in the x direction
  int jj;// nearest cell index to the left in the y direction
  int kk;// nearest cell index to the left in the z direction
  float iw;// normalized distance to the nearest cell index to the left in the x direction
  float jw;// normalized distance to the nearest cell index to the left in the y direction
  float kw;// normalized distance to the nearest cell index to the left in the z direction
};


__device__ void setInterp3Dindex_uFace(vec3 &pos, interpWeight &wgt)
{

  // set a particle position corrected by the start of the domain in each direction
  float par_x = pos._1 - 0.0 * qes_grid.dx;
  float par_y = pos._2 - 0.5 * qes_grid.dy;
  float par_z = pos._3 + 0.5 * qes_grid.dz;

  wgt.ii = floor(par_x / (qes_grid.dx + 1e-7));
  wgt.jj = floor(par_y / (qes_grid.dy + 1e-7));
  wgt.kk = floor(par_z / (qes_grid.dz + 1e-7));

  // fractional distance between the nearest nodes
  wgt.iw = (par_x / qes_grid.dx) - floor(par_x / (qes_grid.dx + 1e-7));
  wgt.jw = (par_y / qes_grid.dy) - floor(par_y / (qes_grid.dy + 1e-7));
  wgt.kw = (par_z / qes_grid.dz) - floor(par_z / (qes_grid.dz + 1e-7));

  // auto itr = std::lower_bound(m_WGD->z.begin(), m_WGD->z.end(), par_zPos);
  // wgt.kk = itr - m_WGD->z.begin() - 1;

  // wgt.kw = (par_zPos - m_WGD->z[wgt.kk]) / (m_WGD->z[wgt.kk + 1] - m_WGD->z[wgt.kk]);
}


__device__ void setInterp3Dindex_vFace(vec3 &pos, interpWeight &wgt)
{

  // set a particle position corrected by the start of the domain in each direction
  float par_x = pos._1 - 0.5 * qes_grid.dx;
  float par_y = pos._2 - 0.0 * qes_grid.dy;
  float par_z = pos._3 + 0.5 * qes_grid.dz;

  wgt.ii = floor(par_x / (qes_grid.dx + 1e-7));
  wgt.jj = floor(par_y / (qes_grid.dy + 1e-7));
  wgt.kk = floor(par_z / (qes_grid.dz + 1e-7));

  // fractional distance between the nearest nodes
  wgt.iw = (par_x / qes_grid.dx) - floor(par_x / (qes_grid.dx + 1e-4));
  wgt.jw = (par_y / qes_grid.dy) - floor(par_y / (qes_grid.dy + 1e-4));
  wgt.kw = (par_z / qes_grid.dz) - floor(par_z / (qes_grid.dz + 1e-4));

  // auto itr = std::lower_bound(m_WGD->z.begin(), m_WGD->z.end(), par_zPos);
  // wgt.kk = itr - m_WGD->z.begin() - 1;

  // wgt.kw = (par_zPos - m_WGD->z[wgt.kk]) / (m_WGD->z[wgt.kk + 1] - m_WGD->z[wgt.kk]);
}

__device__ void setInterp3Dindex_wFace(vec3 &pos, interpWeight &wgt)
{

  // set a particle position corrected by the start of the domain in each direction
  float par_x = pos._1 - 0.5 * qes_grid.dx;
  float par_y = pos._2 - 0.5 * qes_grid.dy;
  float par_z = pos._3 + 1.0 * qes_grid.dz;

  wgt.ii = floor(par_x / (qes_grid.dx + 1e-7));
  wgt.jj = floor(par_y / (qes_grid.dy + 1e-7));
  wgt.kk = floor(par_z / (qes_grid.dz + 1e-7));

  // fractional distance between the nearest nodes
  wgt.iw = (par_x / qes_grid.dx) - floor(par_x / (qes_grid.dx + 1e-7));
  wgt.jw = (par_y / qes_grid.dy) - floor(par_y / (qes_grid.dy + 1e-7));
  wgt.kw = (par_z / qes_grid.dz) - floor(par_z / (qes_grid.dz + 1e-7));

  // auto itr = std::lower_bound(m_WGD->z_face.begin(), m_WGD->z_face.end(), par_zPos);
  // wgt.kk = itr - m_WGD->z_face.begin() - 1;

  // wgt.kw = (par_zPos - m_WGD->z_face[wgt.kk]) / (m_WGD->z_face[wgt.kk + 1] - m_WGD->z_face[wgt.kk]);
}

// always call this after setting the interpolation indices with the setInterp3Dindex_u/v/wFace() function!
__device__ void interp3D_faceVar(const float *data,
                                 const interpWeight &wgt,
                                 float &out)
{

  float cube[2][2][2] = { 0.0 };

  // now set the cube values
  for (int kkk = 0; kkk <= 1; ++kkk) {
    for (int jjj = 0; jjj <= 1; ++jjj) {
      for (int iii = 0; iii <= 1; ++iii) {
        // set the actual indices to use for the linearized Euler data
        int idx = (wgt.kk + kkk) * (qes_grid.ny * qes_grid.nx) + (wgt.jj + jjj) * (qes_grid.nx) + (wgt.ii + iii);
        cube[iii][jjj][kkk] = data[idx];
      }
    }
  }

  // now do the interpolation, with the cube, the counters from the indices,
  // and the normalized width between the point locations and the closest cell left walls
  float u_low = (1 - wgt.iw) * (1 - wgt.jw) * cube[0][0][0]
                + wgt.iw * (1 - wgt.jw) * cube[1][0][0]
                + wgt.iw * wgt.jw * cube[1][1][0]
                + (1 - wgt.iw) * wgt.jw * cube[0][1][0];
  float u_high = (1 - wgt.iw) * (1 - wgt.jw) * cube[0][0][1]
                 + wgt.iw * (1 - wgt.jw) * cube[1][0][1]
                 + wgt.iw * wgt.jw * cube[1][1][1]
                 + (1 - wgt.iw) * wgt.jw * cube[0][1][1];

  out = (u_high - u_low) * wgt.kw + u_low;
}

__device__ void setInterp3Dindex_cellVar(const vec3 &pos, interpWeight &wgt)
{
  float par_x = pos._1 - 0.5 * qes_grid.dx;
  float par_y = pos._2 - 0.5 * qes_grid.dy;
  float par_z = pos._3 + 0.5 * qes_grid.dz;

  // index of the nearest node
  wgt.ii = floor(par_x / (qes_grid.dx + 1e-7));
  wgt.jj = floor(par_y / (qes_grid.dy + 1e-7));
  wgt.kk = floor(par_z / (qes_grid.dz + 1e-7));

  // fractional distance between the nearest nodes
  wgt.iw = (par_x / qes_grid.dx) - floor(par_x / (qes_grid.dx + 1e-7));
  wgt.jw = (par_y / qes_grid.dy) - floor(par_y / (qes_grid.dy + 1e-7));
  wgt.kw = (par_z / qes_grid.dz) - floor(par_z / (qes_grid.dz + 1e-7));

  // auto itr = std::lower_bound(m_WGD->z.begin(), m_WGD->z.end(), par_zPos);
  // wgt.kk = itr - m_WGD->z.begin() - 1;

  // wgt.kw = (par_zPos - m_WGD->z[wgt.kk]) / (m_WGD->z[wgt.kk + 1] - m_WGD->z[wgt.kk]);
}


// always call this after setting the interpolation indices with the setInterp3Dindexing() function!
__device__ void interp3D_cellVar(const float *data,
                                 const interpWeight &wgt,
                                 float &out)
{

  // now set the cube to zero, then fill it using the indices and the counters from the indices
  float cube[2][2][2] = { 0.0 };

  // now set the cube values
  for (int kkk = 0; kkk <= 1; ++kkk) {
    for (int jjj = 0; jjj <= 1; ++jjj) {
      for (int iii = 0; iii <= 1; ++iii) {
        // set the actual indices to use for the linearized Euler data
        int idx = (wgt.kk + kkk) * (qes_grid.ny - 1) * (qes_grid.nx - 1) + (wgt.jj + jjj) * (qes_grid.nx - 1) + (wgt.ii + iii);
        cube[iii][jjj][kkk] = data[idx];
      }
    }
  }

  // now do the interpolation, with the cube, the counters from the indices,
  // and the normalized width between the point locations and the closest cell left walls
  float u_low = (1 - wgt.iw) * (1 - wgt.jw) * cube[0][0][0]
                + wgt.iw * (1 - wgt.jw) * cube[1][0][0]
                + wgt.iw * wgt.jw * cube[1][1][0]
                + (1 - wgt.iw) * wgt.jw * cube[0][1][0];
  float u_high = (1 - wgt.iw) * (1 - wgt.jw) * cube[0][0][1]
                 + wgt.iw * (1 - wgt.jw) * cube[1][0][1]
                 + wgt.iw * wgt.jw * cube[1][1][1]
                 + (1 - wgt.iw) * wgt.jw * cube[0][1][1];

  out = (u_high - u_low) * wgt.kw + u_low;
}

__global__ void interpolate(int length, particle_array d_particle_list, const QESWindsData data)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < length) {
    interpWeight wgt{ 0, 0, 0, 0.0, 0.0, 0.0 };

    float u, v, w;

    // set interpolation indexing variables for uFace variables
    setInterp3Dindex_uFace(d_particle_list.pos[idx], wgt);
    // interpolation of variables on uFace
    interp3D_faceVar(data.u, wgt, u);

    // set interpolation indexing variables for vFace variables
    setInterp3Dindex_vFace(d_particle_list.pos[idx], wgt);
    // interpolation of variables on vFace
    interp3D_faceVar(data.v, wgt, v);

    // set interpolation indexing variables for wFace variables
    setInterp3Dindex_wFace(d_particle_list.pos[idx], wgt);
    // interpolation of variables on wFace
    interp3D_faceVar(data.w, wgt, w);

    d_particle_list.velMean[idx] = { u, v, w };
  }
}

__global__ void interpolate(int length, const vec3 *pos, mat3sym *tau, vec3 *sigma, const QESTurbData data)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < length) {
    // these are the current interp3D variables, as they are used for multiple interpolations for each particle
    interpWeight wgt{ 0, 0, 0, 0.0, 0.0, 0.0 };

    setInterp3Dindex_cellVar(pos[idx], wgt);

    // this is the current reynolds stress tensor
    float txx_out, txy_out, txz_out, tyy_out, tyz_out, tzz_out;
    interp3D_cellVar(data.txx, wgt, txx_out);
    interp3D_cellVar(data.txy, wgt, txy_out);
    interp3D_cellVar(data.txz, wgt, txz_out);
    interp3D_cellVar(data.tyy, wgt, tyy_out);
    interp3D_cellVar(data.tyz, wgt, tyz_out);
    interp3D_cellVar(data.tzz, wgt, tzz_out);
    tau[idx] = { txx_out, txy_out, txz_out, tyy_out, tyz_out, tzz_out };

    float sig_x_out, sig_y_out, sig_z_out;
    sig_x_out = std::sqrt(std::abs(txx_out));
    if (sig_x_out == 0.0)
      sig_x_out = 1e-8;
    sig_y_out = std::sqrt(std::abs(tyy_out));
    if (sig_y_out == 0.0)
      sig_y_out = 1e-8;
    sig_z_out = std::sqrt(std::abs(tzz_out));
    if (sig_z_out == 0.0)
      sig_z_out = 1e-8;
    sigma[idx] = { sig_x_out, sig_y_out, sig_z_out };
  }
}

__global__ void interpolate(int length, particle_array d_particle_list, const QESTurbData data)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < length) {
    // these are the current interp3D variables, as they are used for multiple interpolations for each particle
    interpWeight wgt{ 0, 0, 0, 0.0, 0.0, 0.0 };

    setInterp3Dindex_cellVar(d_particle_list.pos[idx], wgt);

    // this is the CoEps for the particle
    float CoEps_out;
    interp3D_cellVar(data.CoEps, wgt, CoEps_out);
    // make sure CoEps is always bigger than zero
    if (CoEps_out <= 1e-6) {
      CoEps_out = 1e-6;
    }
    d_particle_list.CoEps[idx] = CoEps_out;

    // this is the current reynolds stress tensor
    float txx_out, txy_out, txz_out, tyy_out, tyz_out, tzz_out;
    interp3D_cellVar(data.txx, wgt, txx_out);
    interp3D_cellVar(data.txy, wgt, txy_out);
    interp3D_cellVar(data.txz, wgt, txz_out);
    interp3D_cellVar(data.tyy, wgt, tyy_out);
    interp3D_cellVar(data.tyz, wgt, tyz_out);
    interp3D_cellVar(data.tzz, wgt, tzz_out);
    d_particle_list.tau[idx] = { txx_out, txy_out, txz_out, tyy_out, tyz_out, tzz_out };

    float flux_div_x_out, flux_div_y_out, flux_div_z_out;
    interp3D_cellVar(data.div_tau_x, wgt, flux_div_x_out);
    interp3D_cellVar(data.div_tau_y, wgt, flux_div_y_out);
    interp3D_cellVar(data.div_tau_z, wgt, flux_div_z_out);
    d_particle_list.flux_div[idx] = { flux_div_x_out, flux_div_y_out, flux_div_z_out };

    float nuT_out;
    interp3D_cellVar(data.nuT, wgt, nuT_out);
    d_particle_list.nuT[idx] = nuT_out;
  }
}

void allocate_device_particle_list(particle_array &d_particle_list, int length)
{
  cudaMalloc((void **)&d_particle_list.state, length * sizeof(int));
  cudaMalloc((void **)&d_particle_list.ID, length * sizeof(uint32_t));

  cudaMalloc((void **)&d_particle_list.pos, length * sizeof(vec3));
  cudaMalloc((void **)&d_particle_list.velMean, length * sizeof(vec3));

  cudaMalloc((void **)&d_particle_list.velFluct, length * sizeof(vec3));
  cudaMalloc((void **)&d_particle_list.velFluct_old, length * sizeof(vec3));
  cudaMalloc((void **)&d_particle_list.delta_velFluct, length * sizeof(vec3));

  cudaMalloc((void **)&d_particle_list.CoEps, length * sizeof(float));
  cudaMalloc((void **)&d_particle_list.tau, length * sizeof(mat3sym));
  cudaMalloc((void **)&d_particle_list.tau_old, length * sizeof(mat3sym));

  cudaMalloc((void **)&d_particle_list.flux_div, length * sizeof(vec3));
}

void free_device_particle_list(particle_array &d_particle_list)
{
  cudaFree(d_particle_list.state);
  cudaFree(d_particle_list.ID);

  cudaFree(d_particle_list.CoEps);

  cudaFree(d_particle_list.pos);
  cudaFree(d_particle_list.velMean);

  cudaFree(d_particle_list.velFluct);
  cudaFree(d_particle_list.velFluct_old);
  cudaFree(d_particle_list.delta_velFluct);

  cudaFree(d_particle_list.tau);
  cudaFree(d_particle_list.tau_old);

  cudaFree(d_particle_list.flux_div);
}


void print_particle(const particle &p)
{
  std::string particle_state = "ERROR";
  switch (p.state) {
  case ACTIVE:
    particle_state = "ACTIVE";
    break;
  case INACTIVE:
    particle_state = "INACTIVE";
    break;
  case ROGUE:
    particle_state = "ROGUE";
    break;
  }
  std::cout << "--------------------------------------" << std::endl;
  std::cout << "Particle test print:" << std::endl;
  std::cout << " state   : " << particle_state << std::endl;
  std::cout << " ID      : " << p.ID << std::endl;
  std::cout << " position: " << p.pos._1 << ", " << p.pos._2 << ", " << p.pos._3 << std::endl;
  std::cout << " velocity: " << p.velMean._1 << ", " << p.velMean._2 << ", " << p.velMean._3 << std::endl;
  std::cout << " fluct   : " << p.velFluct._1 << ", " << p.velFluct._2 << ", " << p.velFluct._3 << std::endl;
}


void test_gpu(const int &ntest, const int &new_particle, const int &length)
{

  int gpuID = 0;
  cudaError_t errorCheck = cudaGetDevice(&gpuID);

  int blockCount = 1;
  cudaDeviceGetAttribute(&blockCount, cudaDevAttrMultiProcessorCount, gpuID);
  // std::cout << blockCount << std::endl;

  int threadsPerBlock = 128;
  cudaDeviceGetAttribute(&threadsPerBlock, cudaDevAttrMaxThreadsPerBlock, gpuID);
  // std::cout << threadsPerBlock << std::endl;


  // set QES grid
  qes_grid.dx = 1;
  qes_grid.dy = 1;
  qes_grid.dz = 1;

  qes_grid.nx = 100;
  qes_grid.ny = 100;
  qes_grid.nz = 100;

  int num_cell = qes_grid.nx * qes_grid.ny * qes_grid.nz;
  std::vector<float> data;
  data.resize(num_cell, 1.0);

  if (errorCheck == cudaSuccess) {
    auto gpuStartTime = std::chrono::high_resolution_clock::now();

    // Allocate particle array on the device ONLY
    particle_array d_particle_list[2];
    allocate_device_particle_list(d_particle_list[0], length);
    allocate_device_particle_list(d_particle_list[1], length);

    // initialize on the device
    cudaMemset(d_particle_list[0].state, INACTIVE, length * sizeof(int));
    cudaMemset(d_particle_list[0].ID, 0, length * sizeof(uint32_t));

    cudaMemset(d_particle_list[1].state, INACTIVE, length * sizeof(int));
    cudaMemset(d_particle_list[1].ID, 0, length * sizeof(uint32_t));

    QESWindsData d_qes_winds_data;
    std::fill(data.begin(), data.end(), 1.0);
    cudaMalloc((void **)&d_qes_winds_data.u, num_cell * sizeof(float));
    cudaMemcpy(d_qes_winds_data.u, data.data(), num_cell * sizeof(float), cudaMemcpyHostToDevice);
    std::fill(data.begin(), data.end(), 2.0);
    cudaMalloc((void **)&d_qes_winds_data.v, num_cell * sizeof(float));
    cudaMemcpy(d_qes_winds_data.v, data.data(), num_cell * sizeof(float), cudaMemcpyHostToDevice);
    std::fill(data.begin(), data.end(), 3.0);
    cudaMalloc((void **)&d_qes_winds_data.w, num_cell * sizeof(float));
    cudaMemcpy(d_qes_winds_data.w, data.data(), num_cell * sizeof(float), cudaMemcpyHostToDevice);

    QESTurbData d_qes_turb_data;
    std::fill(data.begin(), data.end(), 0.1);
    cudaMalloc((void **)&d_qes_turb_data.txx, num_cell * sizeof(float));
    cudaMemcpy(d_qes_turb_data.txx, data.data(), num_cell * sizeof(float), cudaMemcpyHostToDevice);
    cudaMalloc((void **)&d_qes_turb_data.txy, num_cell * sizeof(float));
    cudaMemcpy(d_qes_turb_data.txy, data.data(), num_cell * sizeof(float), cudaMemcpyHostToDevice);
    cudaMalloc((void **)&d_qes_turb_data.txz, num_cell * sizeof(float));
    cudaMemcpy(d_qes_turb_data.txz, data.data(), num_cell * sizeof(float), cudaMemcpyHostToDevice);
    cudaMalloc((void **)&d_qes_turb_data.tyy, num_cell * sizeof(float));
    cudaMemcpy(d_qes_turb_data.tyy, data.data(), num_cell * sizeof(float), cudaMemcpyHostToDevice);
    cudaMalloc((void **)&d_qes_turb_data.tyz, num_cell * sizeof(float));
    cudaMemcpy(d_qes_turb_data.tyz, data.data(), num_cell * sizeof(float), cudaMemcpyHostToDevice);
    cudaMalloc((void **)&d_qes_turb_data.tzz, num_cell * sizeof(float));
    cudaMemcpy(d_qes_turb_data.tzz, data.data(), num_cell * sizeof(float), cudaMemcpyHostToDevice);

    cudaMalloc((void **)&d_qes_turb_data.div_tau_x, num_cell * sizeof(float));
    cudaMemcpy(d_qes_turb_data.div_tau_x, data.data(), num_cell * sizeof(float), cudaMemcpyHostToDevice);
    cudaMalloc((void **)&d_qes_turb_data.div_tau_y, num_cell * sizeof(float));
    cudaMemcpy(d_qes_turb_data.div_tau_y, data.data(), num_cell * sizeof(float), cudaMemcpyHostToDevice);
    cudaMalloc((void **)&d_qes_turb_data.div_tau_z, num_cell * sizeof(float));
    cudaMemcpy(d_qes_turb_data.div_tau_z, data.data(), num_cell * sizeof(float), cudaMemcpyHostToDevice);

    cudaMalloc((void **)&d_qes_turb_data.CoEps, num_cell * sizeof(float));
    cudaMemcpy(d_qes_turb_data.CoEps, data.data(), num_cell * sizeof(float), cudaMemcpyHostToDevice);
    cudaMalloc((void **)&d_qes_turb_data.nuT, num_cell * sizeof(float));
    cudaMemcpy(d_qes_turb_data.nuT, data.data(), num_cell * sizeof(float), cudaMemcpyHostToDevice);

    int blockSize = 256;

    int idx = 0, alt_idx = 1;

    // call kernel
    auto kernelStartTime = std::chrono::high_resolution_clock::now();
    for (int k = 0; k < ntest; ++k) {

      cudaMemset(d_particle_list[idx].state, ACTIVE, length * sizeof(int));

      std::vector<vec3> new_pos(length, { 20.0, 50.0, 70.0 });
      cudaMemcpy(d_particle_list[idx].pos, new_pos.data(), length * sizeof(vec3), cudaMemcpyHostToDevice);

      int num_particle = length;// h_lower_count + new_particle;
      // std::cout << num_particle << std::endl;

      int numBlocks_buffer = (length + blockSize - 1) / blockSize;
      int numBlocks_all_particle = (num_particle + blockSize - 1) / blockSize;
      int numBlocks_new_particle = (new_particle + blockSize - 1) / blockSize;

      // these indeces are used to leap-frog the lists of the particles.
      // idx = k % 2;
      // alt_idx = (k + 1) % 2;

      interpolate<<<numBlocks_all_particle, blockSize>>>(num_particle, d_particle_list[idx], d_qes_winds_data);
      interpolate<<<numBlocks_all_particle, blockSize>>>(num_particle, d_particle_list[idx], d_qes_turb_data);

      cudaDeviceSynchronize();
    }

    auto kernelEndTime = std::chrono::high_resolution_clock::now();

    // cudamemcpy back to host
    std::vector<int> particle_state(length);
    std::vector<uint32_t> particle_ID(length);
    // cudaMemcpy(isRogue.data(), d_particle_list.isRogue, length * sizeof(bool), cudaMemcpyDeviceToHost);
    // cudaMemcpy(isActive.data(), &d_particle_list.isActive, length * sizeof(bool), cudaMemcpyDeviceToHost);
    cudaMemcpy(particle_state.data(), d_particle_list[idx].state, length * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(particle_ID.data(), d_particle_list[idx].ID, length * sizeof(uint32_t), cudaMemcpyDeviceToHost);

    std::vector<vec3> pos(length);
    std::vector<vec3> velMean(length);
    std::vector<vec3> velFluct(length);
    // cudaMemcpy(CoEps.data(), d_particle_list.CoEps, length * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(pos.data(), d_particle_list[idx].pos, length * sizeof(vec3), cudaMemcpyDeviceToHost);
    cudaMemcpy(velMean.data(), d_particle_list[idx].velMean, length * sizeof(vec3), cudaMemcpyDeviceToHost);
    cudaMemcpy(velFluct.data(), d_particle_list[idx].velFluct, length * sizeof(vec3), cudaMemcpyDeviceToHost);

    std::vector<particle> particle_list(length);

    for (size_t k = 0; k < particle_list.size(); ++k) {
      particle_list[k].state = particle_state[k];
      particle_list[k].ID = particle_ID[k];
    }
    for (size_t k = 0; k < particle_list.size(); ++k) {
      particle_list[k].pos = pos[k];
    }
    for (size_t k = 0; k < particle_list.size(); ++k) {
      particle_list[k].velMean = velMean[k];
    }
    for (size_t k = 0; k < particle_list.size(); ++k) {
      particle_list[k].velFluct = velFluct[k];
    }

    // cudafree
    free_device_particle_list(d_particle_list[0]);
    free_device_particle_list(d_particle_list[1]);

    auto gpuEndTime = std::chrono::high_resolution_clock::now();


    print_particle(particle_list[0]);
    print_particle(particle_list[1]);

    std::cout << "--------------------------------------" << std::endl;
    std::chrono::duration<double> kernelElapsed = kernelEndTime - kernelStartTime;
    std::cout << "kernel elapsed time: " << kernelElapsed.count() << " s\n";
    std::chrono::duration<double> gpuElapsed = gpuEndTime - gpuStartTime;
    std::cout << "GPU elapsed time:    " << gpuElapsed.count() << " s\n";
  } else {
    printf("CUDA ERROR!\n");
  }
}
