
#include "CUDA_plume_testkernel.h"

#include "CUDA_boundary_conditions.cuh"
// #include "CUDA_particle_partition.cuh"
#include "CUDA_interpolation.cuh"
#include "CUDA_advection.cuh"
#include "CUDA_concentration.cuh"

#include "plume/cuda/Partition.h"
#include "plume/cuda/RandomGenerator.h"

#define PBSTR "||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||"
#define PBWIDTH 60

__device__ __managed__ QESgrid qes_grid;
__device__ __managed__ BC_Params bc_param;
__device__ __managed__ ConcentrationParam param;

__global__ void set_new_particle(int new_particle, particle_array p, float *d_RNG_vals)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < new_particle) {
    // makeRealizable(1.0E-6, p.tau[idx]);
    p.velFluct_old[idx]._1 = p.velFluct_old[idx]._1 * d_RNG_vals[idx];
    p.velFluct_old[idx]._2 = p.velFluct_old[idx]._2 * d_RNG_vals[idx + new_particle];
    p.velFluct_old[idx]._3 = p.velFluct_old[idx]._3 * d_RNG_vals[idx + 2 * new_particle];
  }
}

void print_percentage(const float &percentage)
{
  int val = (int)(percentage * 100);
  int lpad = (int)(percentage * PBWIDTH);
  int rpad = PBWIDTH - lpad;
  printf("\r%3d%% [%.*s%*s]", val, lpad, PBSTR, rpad, "");
  fflush(stdout);
}

float calcRMSE(std::vector<float> &A, std::vector<float> &B)
{
  int nbrBins = (int)A.size();
  float rmse_var = 0.0;
  for (int k = 0; k < nbrBins; ++k) {
    rmse_var += pow(A[k] - B[k], 2);
  }
  return sqrt(rmse_var / (float)nbrBins);
}

float calcMaxAbsErr(std::vector<float> &A, std::vector<float> &B)
{
  int nbrBins = (int)A.size();
  vector<float> tmp(nbrBins);
  for (int k = 0; k < nbrBins; ++k) {
    tmp[k] = std::abs(A[k] - B[k]);
  }
  return *max_element(std::begin(tmp), std::end(tmp));
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
  Timer initTimer("initialization");
  Timer gpuTimer("GPU");

  Timer gpuInitTimer("GPU initialization");
  Timer timeLoopTimer("time loop");
  Timer partitonTimer("particle partitioning");
  Timer interpTimer("interpolation");
  Timer advectTimer("advection");
  Timer concenTimer("concentration");

  float uMean = 2.0;
  float uStar = 0.174;
  float H = 70;
  float C0 = 5.7;
  float zi = 1000;

  // set QES grid
  qes::Domain domain(102, 102, 141, 1.0, 1.0, 1.0);
  std::tie(qes_grid.nx, qes_grid.ny, qes_grid.nz) = domain.getDomainCellNum();
  std::tie(qes_grid.dx, qes_grid.dy, qes_grid.dz) = domain.getDomainSize();

  auto *WGD = new WINDSGeneralData(domain);
  WGD->timestamp.emplace_back("2020-01-01T00:00:00");

  TURBGeneralData *TGD = new TURBGeneralData(WGD);

  for (int k = 0; k < domain.nz(); ++k) {
    for (int j = 0; j < domain.ny(); ++j) {
      for (int i = 0; i < domain.nx(); ++i) {
        WGD->u[domain.face(i, j, k)] = uMean;
      }
    }
  }

  for (int k = 1; k < domain.nz() - 1; ++k) {
    for (int j = 0; j < domain.ny() - 1; ++j) {
      for (int i = 0; i < domain.nx() - 1; ++i) {
        int cellID = domain.cell(i, j, k);
        TGD->txx[cellID] = pow(2.50 * uStar, 2) * pow(1 - domain.z[k] / zi, 3. / 2.);
        TGD->tyy[cellID] = pow(1.78 * uStar, 2) * pow(1 - domain.z[k] / zi, 3. / 2.);
        TGD->tzz[cellID] = pow(1.27 * uStar, 2) * pow(1 - domain.z[k] / zi, 3. / 2.);
        TGD->txz[cellID] = -pow(uStar, 2) * pow(1 - domain.z[k] / zi, 3. / 2.);

        TGD->tke[cellID] = pow(uStar / 0.55, 2.0);
        TGD->CoEps[cellID] = C0 * pow(uStar, 3)
                             / (0.4 * domain.z[k]) * pow(1 - 0.85 * domain.z[k] / zi, 3.0 / 2.0);
      }
    }
  }
  TGD->divergenceStress();

  initTimer.stop();


  int gpuID = 0;
  cudaError_t errorCheck = cudaGetDevice(&gpuID);
  if (errorCheck == cudaSuccess) {

    gpuTimer.start();
    gpuInitTimer.start();

    int blockCount = 1;
    cudaDeviceGetAttribute(&blockCount, cudaDevAttrMultiProcessorCount, gpuID);
    // std::cout << blockCount << std::endl;

    int threadsPerBlock = 128;
    cudaDeviceGetAttribute(&threadsPerBlock, cudaDevAttrMaxThreadsPerBlock, gpuID);
    // std::cout << threadsPerBlock << std::endl;

    curandGenerator_t gen;

    // Create pseudo-random number generator
    // CURAND_CALL(
    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);

    // Set the seed --- not sure how we'll do this yet in general
    // CURAND_CALL()
    curandSetPseudoRandomGeneratorSeed(gen, 1234ULL);

    auto random = new RandomGenerator();

    IDGenerator *id_gen;
    id_gen = IDGenerator::getInstance();

    // QESWindsData d_qes_winds_data;
    // copy_data_gpu(WGD, d_qes_winds_data);
    WGD->allocateDevice();
    WGD->copyDataToDevice();

    QESTurbData d_qes_turb_data;
    copy_data_gpu(TGD, d_qes_turb_data);

    // set boundary condition
    bc_param.xStartDomain = 0 + 0.5 * domain.dx();
    bc_param.yStartDomain = 0 + 0.5 * domain.dy();
    bc_param.zStartDomain = 0;

    bc_param.xEndDomain = 200 - 0.5 * domain.dx();
    bc_param.yEndDomain = 100 - 0.5 * domain.dy();
    bc_param.zEndDomain = 140;

    // concnetration calculation
    param.lbndx = 0.0;
    param.lbndy = 1.0;
    param.lbndz = 1.0;

    param.ubndx = 100.0;
    param.ubndy = 99.0;
    param.ubndz = 139.0;

    param.nx = 20;
    param.ny = 49;
    param.nz = 69;

    param.dx = (param.ubndx - param.lbndx) / (param.nx);
    param.dy = (param.ubndy - param.lbndy) / (param.ny);
    param.dz = (param.ubndz - param.lbndz) / (param.nz);

    std::vector<int> h_pBox(param.nx * param.ny * param.nz, 0.0);

    int *d_pBox;
    cudaMalloc(&d_pBox, param.nx * param.ny * param.nz * sizeof(int));

    int h_lower_count = 0, h_upper_count;

    auto *partition = new Partition(length);

    // Allocate particle array on the device ONLY
    particle_array d_particle[2];
    partition->allocate_device_particle_list(d_particle[0], length);
    partition->allocate_device_particle_list(d_particle[1], length);

    particle_array d_new_particle;
    partition->allocate_device_particle_list(d_new_particle, new_particle);

    // initialize on the device
    cudaMemset(d_particle[0].state, INACTIVE, length * sizeof(int));
    cudaMemset(d_particle[0].ID, 0, length * sizeof(uint32_t));

    cudaMemset(d_particle[1].state, INACTIVE, length * sizeof(int));
    cudaMemset(d_particle[1].ID, 0, length * sizeof(uint32_t));

    // Allocate n floats on device to hold random numbers
    // Allocate numParticle * 3 floats on host
    float *d_RNG_vals, *d_RNG_newvals;
    cudaMalloc((void **)&d_RNG_vals, 3 * length * sizeof(float));
    cudaMalloc((void **)&d_RNG_newvals, 3 * new_particle * sizeof(float));

    random->create("advect", 3 * length);
    random->create("new_particle", 3 * new_particle);

    /*int *d_sorting_index;
      cudaMalloc((void **)&d_sorting_index, length * sizeof(int));*/

    /*int *d_lower_count, *d_upper_count;
    cudaMalloc(&d_lower_count, sizeof(int));
    cudaMalloc(&d_upper_count, sizeof(int));*/

    /*int h_active_count_1, h_empty_count_1;
    int *d_active_count_1, *d_empty_count_1;
    cudaMalloc(&d_active_count_1, sizeof(int));
    cudaMalloc(&d_empty_count_1, sizeof(int));*/

    /*int h_active_count_2, h_empty_count_2;
    int *d_active_count_2, *d_empty_count_2;
    cudaMalloc(&d_active_count_2, sizeof(int));
    cudaMalloc(&d_empty_count_2, sizeof(int));*/

    gpuInitTimer.stop();

    int blockSize = 256;

    float ongoingAveragingTime = 0.0;
    float timeStep = 1.0;
    float volume = param.dx * param.dy * param.dz;

    int idx = 0;

    timeLoopTimer.start();
    // call kernel
    std::cout << "buffer usage: " << std::endl;
    for (int k = 0; k < ntest; ++k) {

      // cudaMemset(d_lower_count, 0, sizeof(int));
      // cudaMemset(d_upper_count, 0, sizeof(int));

      // cudaMemset(d_active_count_1, 0, sizeof(int));
      // cudaMemset(d_empty_count_1, 0, sizeof(int));

      // cudaMemset(d_active_count_2, 0, sizeof(int));
      // cudaMemset(d_empty_count_2, 0, sizeof(int));

      // cudaMemset(d_sorting_index, -1, length * sizeof(int));

      cudaMemset(d_new_particle.state, ACTIVE, new_particle * sizeof(int));
      std::vector<uint32_t> new_ID(new_particle);
      id_gen->get(new_ID);
      cudaMemcpy(d_new_particle.ID, new_ID.data(), new_particle * sizeof(uint32_t), cudaMemcpyHostToDevice);
      std::vector<vec3> new_pos(new_particle, { 20.0, 50.0, 70.0 });
      cudaMemcpy(d_new_particle.pos, new_pos.data(), new_particle * sizeof(vec3), cudaMemcpyHostToDevice);
      // std::vector<vec3> new_sig(new_particle, { 0.2, 0.2, 0.2 });
      // cudaMemcpy(d_new_particle.velFluct_old, new_sig.data(), new_particle * sizeof(vec3), cudaMemcpyHostToDevice);


      // int num_particle = length;
      int num_particle = partition->active() + new_particle;
      // std::cout << num_particle << std::endl;

      int numBlocks_buffer = (length + blockSize - 1) / blockSize;
      int numBlocks_all_particle = (num_particle + blockSize - 1) / blockSize;
      int numBlocks_new_particle = (new_particle + blockSize - 1) / blockSize;

      // these indeces are used to leap-frog the lists of the particles.
      // int idx = k % 2;
      // int alt_idx = (k + 1) % 2;

      curandGenerateNormal(gen, d_RNG_vals, 3 * length, 0.0, 1.0);
      curandGenerateNormal(gen, d_RNG_newvals, 3 * new_particle, 0.0, 1.0);

      random->generate("advect", 0.0, 1.0);
      random->generate("new_particle", 0.0, 1.0);

      partitonTimer.start();

      idx = partition->run(k, d_particle);

      /*partition_particle_select<<<numBlocks_buffer, blockSize>>>(d_particle[alt_idx],
                                                                 d_lower_count,
                                                                 d_upper_count,
                                                                 d_sorting_index,
                                                                 length);
      partition_particle_reset<<<numBlocks_buffer, blockSize>>>(d_particle[idx],
                                                                length);
      partition_particle_sorting<<<numBlocks_buffer, blockSize>>>(d_particle[idx],
                                                                  d_particle[alt_idx],
                                                                  d_sorting_index,
                                                                  length);*/

      /*partition_particle<<<numBlocks_buffer, blockSize>>>(d_particle[idx],
                                                          d_particle[alt_idx],
                                                          d_lower_count,
                                                          d_upper_count,
                                                          length);*/

      // cudaDeviceSynchronize();
      // cudaMemcpy(&h_lower_count, d_lower_count, sizeof(int), cudaMemcpyDeviceToHost);
      // cudaMemcpy(&h_upper_count, d_upper_count, sizeof(int), cudaMemcpyDeviceToHost);

      partitonTimer.stop();

      /*check_buffer<<<numBlocks_buffer, blockSize>>>(d_particle[idx],
                                                    d_active_count_2,
                                                    d_empty_count_2,
                                                    length);
      cudaMemcpy(&h_active_count_2, d_active_count_2, sizeof(int), cudaMemcpyDeviceToHost);
      cudaMemcpy(&h_empty_count_2, d_empty_count_2, sizeof(int), cudaMemcpyDeviceToHost);*/

      interpolate<<<numBlocks_new_particle, blockSize>>>(new_particle,
                                                         d_new_particle.pos,
                                                         d_new_particle.tau,
                                                         d_new_particle.velFluct_old,
                                                         d_qes_turb_data,
                                                         qes_grid);
      // cudaDeviceSynchronize();
      /*set_new_particle<<<numBlocks_new_particle, blockSize>>>(new_particle,
                                                              d_new_particle,
                                                              d_RNG_newvals);*/

      set_new_particle<<<numBlocks_new_particle, blockSize>>>(new_particle,
                                                              d_new_particle,
                                                              random->get("new_particle"));

      // cudaDeviceSynchronize();
      partition->insert(new_particle, d_new_particle, d_particle[idx]);

      /*insert_particle<<<numBlocks_new_particle, blockSize>>>(new_particle,
                                                               d_lower_count,
                                                               d_new_particle,
                                                               d_particle[idx],
                                                               length);*/
      cudaDeviceSynchronize();
      /*check_buffer < < < numBlocks_buffer, blockSize >>> (d_particle[idx],
                                                             d_active_count_1,
                                                             d_empty_count_1,
                                                             length);
      cudaMemcpy(&h_active_count_1, d_active_count_1, sizeof(int), cudaMemcpyDeviceToHost);
      cudaMemcpy(&h_empty_count_1, d_empty_count_1, sizeof(int), cudaMemcpyDeviceToHost);*/

      interpTimer.start();
      /*interpolate<<<numBlocks_all_particle, blockSize>>>(num_particle,
                                                         d_particle[idx],
                                                         d_qes_winds_data,
                                                         qes_grid);*/
      interpolate<<<numBlocks_all_particle, blockSize>>>(num_particle,
                                                         d_particle[idx],
                                                         WGD->d_data,
                                                         qes_grid);
      // interpolate_1<<<numBlocks_all_particle, blockSize>>>(num_particle, d_particle[idx], d_qes_turb_data, qes_grid);
      // interpolate_2<<<numBlocks_all_particle, blockSize>>>(num_particle, d_particle[idx], d_qes_turb_data, qes_grid);
      // interpolate_3<<<numBlocks_all_particle, blockSize>>>(num_particle, d_particle[idx], d_qes_turb_data, qes_grid);
      interpolate<<<numBlocks_all_particle, blockSize>>>(num_particle,
                                                         d_particle[idx],
                                                         d_qes_turb_data,
                                                         qes_grid);
      cudaDeviceSynchronize();
      interpTimer.stop();


      advectTimer.start();
      /*advect_particle<<<numBlocks_all_particle, blockSize>>>(d_particle[idx],
                                                             d_RNG_vals,
                                                             bc_param,
                                                             num_particle);*/

      advect_particle<<<numBlocks_all_particle, blockSize>>>(d_particle[idx],
                                                             random->get("advect"),
                                                             bc_param,
                                                             num_particle);


      // this is slower that calling devive function bc in the kernel
      // boundary_conditions<<<numBlocks_all_particle, blockSize>>>(num_particle, d_particle[idx]);

      // cudaMemcpy(&h_lower_count, d_lower_count, sizeof(int), cudaMemcpyDeviceToHost);
      // cudaMemcpy(&h_upper_count, d_upper_count, sizeof(int), cudaMemcpyDeviceToHost);
      //  std::cout << k << " " << h_lower_count << " " << num_particle << std::endl;

      /*std::cout << k + 1 << " "
                << h_lower_count << " "
                << h_active_count_2 << " "
                << h_active_count_1 << " "
                << h_lower_count + new_particle - h_active_count_1 << " "
                << h_lower_count + h_upper_count << " "
                << h_active_count_1 + h_empty_count_1 << " "
                << h_active_count_2 + h_empty_count_2 << std::endl;*/

      cudaDeviceSynchronize();
      advectTimer.stop();

      if (k >= 1000) {
        concenTimer.start();
        collect<<<numBlocks_all_particle, blockSize>>>(num_particle,
                                                       d_particle[idx],
                                                       d_pBox,
                                                       param);
        cudaDeviceSynchronize();
        ongoingAveragingTime += timeStep;
        concenTimer.stop();
      }

      print_percentage((float)partition->active() / (float)length);
    }
    std::cout << std::endl;
    timeLoopTimer.stop();

    int numBlocks_buffer = (length + blockSize - 1) / blockSize;

    // cudaMemset(d_lower_count, 0, sizeof(int));
    // cudaMemset(d_upper_count, 0, sizeof(int));
    // check_buffer<<<numBlocks_buffer, blockSize>>>(d_particle[idx], d_lower_count, d_upper_count, length);
    // cudaMemcpy(&h_lower_count, d_lower_count, sizeof(int), cudaMemcpyDeviceToHost);
    // cudaMemcpy(&h_upper_count, d_upper_count, sizeof(int), cudaMemcpyDeviceToHost);

    int active_count = 0, empty_count = 0;
    partition->check(d_particle[idx], active_count, empty_count);

    // cudamemcpy back to host
    std::vector<int> particle_state(length);
    std::vector<uint32_t> particle_ID(length);
    // cudaMemcpy(isRogue.data(), d_particle.isRogue, length * sizeof(bool), cudaMemcpyDeviceToHost);
    // cudaMemcpy(isActive.data(), &d_particle.isActive, length * sizeof(bool), cudaMemcpyDeviceToHost);
    cudaMemcpy(particle_state.data(), d_particle[idx].state, length * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(particle_ID.data(), d_particle[idx].ID, length * sizeof(uint32_t), cudaMemcpyDeviceToHost);

    std::vector<vec3> pos(length);
    std::vector<vec3> velMean(length);
    std::vector<vec3> velFluct(length);
    // cudaMemcpy(CoEps.data(), d_particle.CoEps, length * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(pos.data(), d_particle[idx].pos, length * sizeof(vec3), cudaMemcpyDeviceToHost);
    cudaMemcpy(velMean.data(), d_particle[idx].velMean, length * sizeof(vec3), cudaMemcpyDeviceToHost);
    cudaMemcpy(velFluct.data(), d_particle[idx].velFluct, length * sizeof(vec3), cudaMemcpyDeviceToHost);

    cudaMemcpy(h_pBox.data(), d_pBox, param.nx * param.ny * param.nz * sizeof(int), cudaMemcpyDeviceToHost);

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
    partition->free_device_particle_list(d_particle[0]);
    partition->free_device_particle_list(d_particle[1]);
    partition->free_device_particle_list(d_new_particle);
    // cudaFree(d_sorting_index);
    cudaFree(d_pBox);
    cudaFree(d_RNG_vals);
    cudaFree(d_RNG_newvals);

    WGD->freeDevice();

    delete partition;
    delete random;

    gpuTimer.stop();

    int count = 0;
    for (auto &p : particle_list) {
      if (p.state == ACTIVE) { count++; }
    }
    std::cout << "-------------------------------------------------------------------" << std::endl;
    std::cout << "buffer status: " << active_count << " " << empty_count << " " << length << std::endl;
    // print_percentage((float)h_lower_count / (float)length);
    // std::cout << std::endl;
    // std::cout << "--------------------------------------" << std::endl;
    std::cout << "number of active particle: " << count << std::endl;

    // print_particle(particle_list[0]);
    // print_particle(particle_list[1]);

    // source info (hard coded because no way to access the source info here)
    float xS = 20;
    float yS = 50;
    float zS = 70;
    float Q = new_particle;// #par/s (source strength)
    float tRelease = ntest;// total time of release
    float Ntot = Q * tRelease;// total number of particles

    float CNorm = (uMean * H * H / Q);

    float dt = timeStep;
    float tAvg = ongoingAveragingTime;

    // normalization of particle count #particle -> time-averaged # particle/m3
    float CC = dt / tAvg / volume;

    std::cout << "total time for current on going average: " << tAvg << std::endl;
    std::cout << "normalization of particle count: " << CC << std::endl;
    std::cout << "normalization for source: " << CNorm << std::endl;

    // output concentration storage variables
    std::vector<float> xBoxCen(param.nx, 0.0);
    std::vector<float> yBoxCen(param.ny, 0.0);
    std::vector<float> zBoxCen(param.nz, 0.0);

    int zR = 0, yR = 0, xR = 0;
    for (int k = 0; k < param.nz; ++k) {
      zBoxCen.at(k) = param.lbndz + (zR * param.dz) + (param.dz / 2.0);
      zR++;
    }
    for (int j = 0; j < param.ny; ++j) {
      yBoxCen.at(j) = param.lbndy + (yR * param.dy) + (param.dy / 2.0);
      yR++;
    }
    for (int i = 0; i < param.nx; ++i) {
      xBoxCen.at(i) = param.lbndx + (xR * param.dx) + (param.dx / 2.0);
      xR++;
    }

    // model variances
    float sigV = 1.78f * uStar;
    float sigW = powf(1.0f / (3.0f * 0.4f * 0.4f), 1.0f / 3.0f) * uStar;

    // working variables to calculate statistics
    float SSres = 0.0, SStot = 0.0;
    std::vector<float> xoH, maxRelErr, relRMSE;

    for (int i = 7; i < param.nx; ++i) {
      // x-location of the profile
      float x = xBoxCen[i] - xS;

      // time of flight of the particle
      float t = x / uMean;

      // horizontal and vertical spread rate
      float Ti = powf(2.5f * uStar / zi, -1);
      float To = 1.001;
      float Fy = powf(1.0f + powf(t / Ti, 0.5), -1);
      // float Fz=(1+0.945*(t/To)^0.8)^(-1);
      float Fz = Fy;
      float sigY = sigV * t * Fy;
      float sigZ = sigW * t * Fz;

      // calculate the normalize concentration from Gaussian plume model
      std::vector<float> CStarModel;
      CStarModel.resize(param.ny * param.nz, 0.0);
      for (int k = 0; k < param.nz; ++k) {
        for (int j = 0; j < param.ny; ++j) {
          float y = yBoxCen[j] - yS;
          float z = zBoxCen[k] - zS;
          CStarModel[j + k * param.ny] = Q / (2 * M_PI * uMean * sigY * sigZ)
                                         * expf(-0.5f * powf(y, 2) / powf(sigY, 2))
                                         * expf(-0.5f * powf(z, 2) / powf(sigZ, 2))
                                         * CNorm;
        }
      }

      // calculate the normalize concentration from QES-plume
      std::vector<float> CStarQES;
      CStarQES.resize(param.ny * param.nz, 0.0);
      for (int k = 0; k < param.nz; ++k) {
        for (int j = 0; j < param.ny; ++j) {
          int id = i + j * param.nx + k * param.nx * param.ny;
          CStarQES[j + k * param.ny] = h_pBox[id] * CC * CNorm;
        }
      }

      auto maxModel = *max_element(std::begin(CStarModel), std::end(CStarModel));
      // auto maxQES = *max_element(std::begin(CStarQES), std::end(CStarQES));

      // root mean square error
      float RMSE = calcRMSE(CStarQES, CStarModel);

      // maximum relative error
      float maxAbsErr = calcMaxAbsErr(CStarQES, CStarModel);

      xoH.push_back(x / H);
      maxRelErr.push_back(maxAbsErr / maxModel);
      relRMSE.push_back(RMSE / maxModel);

      // calculate r2
      float CStarQESMean = 0.0;
      for (auto k = 0u; k < CStarQES.size(); ++k) {
        CStarQESMean += CStarQES[k];
      }
      CStarQESMean /= (float)CStarQES.size();

      for (auto k = 0u; k < CStarQES.size(); ++k) {
        SSres += powf(CStarModel[k] - CStarQES[k], 2);
        SStot += powf(CStarQES[k] - CStarQESMean, 2);
      }
    }
    float R2 = 1.0f - SSres / SStot;

    QESNetCDFOutput_v2 *outfile = nullptr;
    outfile = new QESNetCDFOutput_v2("test_plumeOut.nc");

    outfile->newDimension("x_c", "x-center collection box", "m", &xBoxCen);
    outfile->newDimension("y_c", "y-center collection box", "m", &yBoxCen);
    outfile->newDimension("z_c", "z-center collection box", "m", &zBoxCen);

    outfile->newDimensionSet("concentration", { "t", "z_c", "y_c", "x_c" });

    outfile->newField("t_avg", "Averaging time", "s", "time", &ongoingAveragingTime);
    outfile->newField("p_count", "number of particle per box", "#ofPar", "concentration", &h_pBox);
    std::vector<float> CStar(param.nx * param.ny * param.nz);
    for (size_t id = 0; id < CStar.size(); ++id) {
      CStar[id] = h_pBox[id] * CC * CNorm;
    }
    outfile->newField("c", "normailzed concentration", "--", "concentration", &CStar);

    QEStime time;
    outfile->pushAllFieldsToFile(time);

    delete outfile;
    delete WGD, TGD;

    // check the results
    std::cout << "-------------------------------------------------------------------" << std::endl;
    std::cout << "TEST RESULTS" << std::endl;
    std::cout << "-------------------------------------------------------------------" << std::endl;
    std::cout << "x / H \t maxRelErr \t relRMSE" << std::endl;
    for (auto k = 0u; k < xoH.size(); ++k) {
      printf("%.3f \t %.6f \t %.6f\n", xoH[k], maxRelErr[k], relRMSE[k]);
      // std::cout << xoH[k] << " " << maxRelErr[k] << "\t " << relRMSE[k] << std::endl;
    }
    std::cout << "-------------------------------------------------------------------" << std::endl;
    std::cout << "QES-Plume r2 coeff: r2 = " << R2 << std::endl;
    std::cout << "-------------------------------------------------------------------" << std::endl;
    std::cout << std::endl;

    std::cout << "-------------------------------------------------------------------" << std::endl;
    std::cout << "TEST PROFILING" << std::endl;
    std::cout << "-------------------------------------------------------------------" << std::endl;
    initTimer.show();
    gpuTimer.show();
    gpuInitTimer.show();
    std::cout << "-------------------------------------------------------------------" << std::endl;
    timeLoopTimer.show();
    partitonTimer.show();
    interpTimer.show();
    advectTimer.show();
    concenTimer.show();
    std::cout << "-------------------------------------------------------------------" << std::endl;

    // std::cout << "interpolation elapsed time: " << interpElapsed.count() << " s\n";
    // std::cout << "advection elapsed time: " << advectElapsed.count() << " s\n";
    // std::cout << "concentration elapsed time: " << concenElapsed.count() << " s\n";
    // std::cout << "kernel elapsed time: " << kernelElapsed.count() << " s\n";
    // std::cout << "GPU elapsed time:    " << gpuElapsed.count() << " s\n";
  } else {
    printf("CUDA ERROR!\n");
  }
}
