
#include "CUDA_plume_testkernel.h"

#include "CUDA_boundary_conditions.cuh"
#include "CUDA_particle_partition.cuh"
#include "CUDA_interpolation.cuh"
#include "CUDA_advection.cuh"
#include "CUDA_concentration.cuh"

#define PBSTR "||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||"
#define PBWIDTH 60

class test_WINDSGeneralData : public WINDSGeneralData
{
public:
  test_WINDSGeneralData()
  {}
  test_WINDSGeneralData(const int gridSize[3], const float gridRes[3])
  {

    nx = gridSize[0];
    ny = gridSize[1];
    nz = gridSize[2];

    // Modify the domain size to fit the Staggered Grid used in the solver
    nx += 1;// +1 for Staggered grid
    ny += 1;// +1 for Staggered grid
    nz += 2;// +2 for staggered grid and ghost cell

    dx = gridRes[0];// Grid resolution in x-direction
    dy = gridRes[1];// Grid resolution in y-direction
    dz = gridRes[2];// Grid resolution in z-direction
    dxy = MIN_S(dx, dy);

    defineVerticalStretching(dz);
    defineVerticalGrid();
    defineHorizontalGrid();

    timestamp.emplace_back("2020-01-01T00:00:00");

    allocateMemory();
  }

  virtual ~test_WINDSGeneralData()
  {}

private:
};

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
  float uMean = 2.0;
  float uStar = 0.174;
  float H = 70;
  float C0 = 5.7;
  float zi = 1000;

  // set QES grid
  qes_grid.dx = 1;
  qes_grid.dy = 1;
  qes_grid.dz = 1;

  qes_grid.nx = 202;
  qes_grid.ny = 102;
  qes_grid.nz = 141;

  int gridSize[3] = { qes_grid.nx, qes_grid.ny, qes_grid.nz };
  float gridRes[3] = { qes_grid.dx, qes_grid.dy, qes_grid.dz };

  WINDSGeneralData *WGD = new test_WINDSGeneralData(gridSize, gridRes);
  TURBGeneralData *TGD = new TURBGeneralData(WGD);

  for (int k = 0; k < WGD->nz; ++k) {
    for (int j = 0; j < WGD->ny; ++j) {
      for (int i = 0; i < WGD->nx; ++i) {
        int faceID = i + j * (WGD->nx) + k * (WGD->nx) * (WGD->ny);
        WGD->u[faceID] = uMean;
      }
    }
  }

  for (int k = 1; k < WGD->nz - 1; ++k) {
    for (int j = 0; j < WGD->ny - 1; ++j) {
      for (int i = 0; i < WGD->nx - 1; ++i) {
        int cellID = i + j * (WGD->nx - 1) + k * (WGD->nx - 1) * (WGD->ny - 1);
        TGD->txx[cellID] = pow(2.50 * uStar, 2) * pow(1 - WGD->z[k] / zi, 3. / 2.);
        TGD->tyy[cellID] = pow(1.78 * uStar, 2) * pow(1 - WGD->z[k] / zi, 3. / 2.);
        TGD->tzz[cellID] = pow(1.27 * uStar, 2) * pow(1 - WGD->z[k] / zi, 3. / 2.);
        TGD->txz[cellID] = -pow(uStar, 2) * pow(1 - WGD->z[k] / zi, 3. / 2.);

        TGD->tke[cellID] = pow(uStar / 0.55, 2.0);
        TGD->CoEps[cellID] = C0 * pow(uStar, 3)
                             / (0.4 * WGD->z[k]) * pow(1 - 0.85 * WGD->z[k] / zi, 3.0 / 2.0);
      }
    }
  }
  TGD->divergenceStress();

  int gpuID = 0;
  cudaError_t errorCheck = cudaGetDevice(&gpuID);
  if (errorCheck == cudaSuccess) {
    auto gpuStartTime = std::chrono::high_resolution_clock::now();

    int blockCount = 1;
    cudaDeviceGetAttribute(&blockCount, cudaDevAttrMultiProcessorCount, gpuID);
    // std::cout << blockCount << std::endl;

    int threadsPerBlock = 128;
    cudaDeviceGetAttribute(&threadsPerBlock, cudaDevAttrMaxThreadsPerBlock, gpuID);
    // std::cout << threadsPerBlock << std::endl;

    curandGenerator_t gen;
    float *d_RNG_vals, *d_RNG_newvals;

    // Create pseudo-random number generator
    // CURAND_CALL(
    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);

    // Set the seed --- not sure how we'll do this yet in general
    // CURAND_CALL(
    curandSetPseudoRandomGeneratorSeed(gen, 1234ULL);

    IDGenerator *id_gen;
    id_gen = IDGenerator::getInstance();

    QESWindsData d_qes_winds_data;
    copy_data_gpu(WGD, d_qes_winds_data);

    QESTurbData d_qes_turb_data;
    copy_data_gpu(TGD, d_qes_turb_data);

    // set boundary condition
    bc_param.xStartDomain = 0 + 0.5 * qes_grid.dx;
    bc_param.yStartDomain = 0 + 0.5 * qes_grid.dy;
    bc_param.zStartDomain = 0;

    bc_param.xEndDomain = 200 - 0.5 * qes_grid.dx;
    bc_param.yEndDomain = 100 - 0.5 * qes_grid.dy;
    bc_param.zEndDomain = 140;

    // concnetration calculation
    param.lbndx = 0.0;
    param.lbndy = 1.0;
    param.lbndz = 1.0;

    param.ubndx = 200.0;
    param.ubndy = 99.0;
    param.ubndz = 139.0;

    param.nx = 40;
    param.ny = 49;
    param.nz = 69;

    param.dx = (param.ubndx - param.lbndx) / (param.nx);
    param.dy = (param.ubndy - param.lbndy) / (param.ny);
    param.dz = (param.ubndz - param.lbndz) / (param.nz);

    std::vector<int> h_pBox(param.nx * param.ny * param.nz, 0.0);

    int *d_pBox;
    cudaMalloc(&d_pBox, param.nx * param.ny * param.nz * sizeof(int));

    int h_lower_count = 0, h_upper_count;

    std::chrono::duration<double> interpElapsed(0.0);
    std::chrono::duration<double> advectElapsed(0.0);
    std::chrono::duration<double> concenElapsed(0.0);

    // Allocate particle array on the device ONLY
    particle_array d_particle[2];
    allocate_device_particle_list(d_particle[0], length);
    allocate_device_particle_list(d_particle[1], length);

    particle_array d_new_particle;
    allocate_device_particle_list(d_new_particle, new_particle);

    // initialize on the device
    cudaMemset(d_particle[0].state, INACTIVE, length * sizeof(int));
    cudaMemset(d_particle[0].ID, 0, length * sizeof(uint32_t));

    cudaMemset(d_particle[1].state, INACTIVE, length * sizeof(int));
    cudaMemset(d_particle[1].ID, 0, length * sizeof(uint32_t));

    // Allocate n floats on device to hold random numbers
    // Allocate numParticle * 3 floats on host
    cudaMalloc((void **)&d_RNG_vals, 3 * length * sizeof(float));
    cudaMalloc((void **)&d_RNG_newvals, 3 * new_particle * sizeof(float));

    int *d_lower_count, *d_upper_count;
    cudaMalloc(&d_lower_count, sizeof(int));
    cudaMalloc(&d_upper_count, sizeof(int));

    int blockSize = 256;

    int idx = 0, alt_idx = 1;

    float ongoingAveragingTime = 0.0;
    float timeStep = 1.0;
    float volume = param.dx * param.dy * param.dz;

    // call kernel
    auto kernelStartTime = std::chrono::high_resolution_clock::now();
    std::cout << "buffer usage: " << std::endl;
    for (int k = 0; k < ntest; ++k) {

      cudaMemset(d_lower_count, 0, sizeof(int));
      cudaMemset(d_upper_count, 0, sizeof(int));

      cudaMemset(d_new_particle.state, ACTIVE, new_particle * sizeof(int));
      std::vector<uint32_t> new_ID(new_particle);
      id_gen->get(new_ID);
      cudaMemcpy(d_new_particle.ID, new_ID.data(), new_particle * sizeof(uint32_t), cudaMemcpyHostToDevice);
      std::vector<vec3> new_pos(new_particle, { 20.0, 50.0, 70.0 });
      cudaMemcpy(d_new_particle.pos, new_pos.data(), new_particle * sizeof(vec3), cudaMemcpyHostToDevice);
      // std::vector<vec3> new_sig(new_particle, { 0.2, 0.2, 0.2 });
      // cudaMemcpy(d_new_particle.velFluct_old, new_sig.data(), new_particle * sizeof(vec3), cudaMemcpyHostToDevice);


      // int num_particle = length;
      int num_particle = h_lower_count + new_particle;
      // std::cout << num_particle << std::endl;

      int numBlocks_buffer = (length + blockSize - 1) / blockSize;
      int numBlocks_all_particle = (num_particle + blockSize - 1) / blockSize;
      int numBlocks_new_particle = (new_particle + blockSize - 1) / blockSize;

      // these indeces are used to leap-frog the lists of the particles.
      idx = k % 2;
      alt_idx = (k + 1) % 2;

      curandGenerateNormal(gen, d_RNG_vals, 3 * length, 0.0, 1.0);
      curandGenerateNormal(gen, d_RNG_newvals, 3 * new_particle, 0.0, 1.0);

      partition_particle<<<numBlocks_buffer, blockSize>>>(d_particle[idx],
                                                          d_particle[alt_idx],
                                                          d_lower_count,
                                                          d_upper_count,
                                                          length);
      cudaDeviceSynchronize();

      interpolate<<<numBlocks_new_particle, blockSize>>>(new_particle,
                                                         d_new_particle.pos,
                                                         d_new_particle.tau,
                                                         d_new_particle.velFluct_old,
                                                         d_qes_turb_data,
                                                         qes_grid);
      set_new_particle<<<numBlocks_new_particle, blockSize>>>(new_particle,
                                                              d_new_particle,
                                                              d_RNG_newvals);
      cudaDeviceSynchronize();

      insert_particle<<<numBlocks_new_particle, blockSize>>>(length,
                                                             new_particle,
                                                             d_lower_count,
                                                             d_new_particle,
                                                             d_particle[idx]);
      cudaDeviceSynchronize();

      auto interpStartTime = std::chrono::high_resolution_clock::now();

      interpolate<<<numBlocks_all_particle, blockSize>>>(num_particle,
                                                         d_particle[idx],
                                                         d_qes_winds_data,
                                                         qes_grid);
      // interpolate_1<<<numBlocks_all_particle, blockSize>>>(num_particle, d_particle[idx], d_qes_turb_data, qes_grid);
      // interpolate_2<<<numBlocks_all_particle, blockSize>>>(num_particle, d_particle[idx], d_qes_turb_data, qes_grid);
      // interpolate_3<<<numBlocks_all_particle, blockSize>>>(num_particle, d_particle[idx], d_qes_turb_data, qes_grid);
      interpolate<<<numBlocks_all_particle, blockSize>>>(num_particle,
                                                         d_particle[idx],
                                                         d_qes_turb_data,
                                                         qes_grid);
      cudaDeviceSynchronize();

      auto interpEndTime = std::chrono::high_resolution_clock::now();
      interpElapsed += interpEndTime - interpStartTime;

      auto advectStartTime = std::chrono::high_resolution_clock::now();

      advect_particle<<<numBlocks_all_particle, blockSize>>>(num_particle,
                                                             d_particle[idx],
                                                             d_RNG_vals,
                                                             bc_param);

      // this is slower that calling devive function bc in the kernel
      // boundary_conditions<<<numBlocks_all_particle, blockSize>>>(num_particle, d_particle[idx]);

      cudaMemcpy(&h_lower_count, d_lower_count, sizeof(int), cudaMemcpyDeviceToHost);
      cudaMemcpy(&h_upper_count, d_upper_count, sizeof(int), cudaMemcpyDeviceToHost);
      // std::cout << k << " " << h_lower_count << " " << h_upper_count << std::endl;

      cudaDeviceSynchronize();

      auto advectEndTime = std::chrono::high_resolution_clock::now();
      advectElapsed += advectEndTime - advectStartTime;


      if (k >= 1000) {
        auto concenStartTime = std::chrono::high_resolution_clock::now();
        collect<<<numBlocks_all_particle, blockSize>>>(num_particle,
                                                       d_particle[idx],
                                                       d_pBox,
                                                       param);
        cudaDeviceSynchronize();
        ongoingAveragingTime += timeStep;
        auto concenEndTime = std::chrono::high_resolution_clock::now();
        concenElapsed += concenEndTime - concenStartTime;
      }

      print_percentage((float)h_lower_count / (float)length);
    }
    std::cout << std::endl;


    int numBlocks_buffer = (length + blockSize - 1) / blockSize;
    cudaMemset(d_lower_count, 0, sizeof(int));
    cudaMemset(d_upper_count, 0, sizeof(int));
    check_buffer<<<numBlocks_buffer, blockSize>>>(d_particle[idx], d_lower_count, d_upper_count, length);

    auto kernelEndTime = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> kernelElapsed = kernelEndTime - kernelStartTime;

    cudaMemcpy(&h_lower_count, d_lower_count, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_upper_count, d_upper_count, sizeof(int), cudaMemcpyDeviceToHost);

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

    auto gpuEndTime = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> gpuElapsed = gpuEndTime - gpuStartTime;

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
    free_device_particle_list(d_particle[0]);
    free_device_particle_list(d_particle[1]);
    free_device_particle_list(d_new_particle);
    cudaFree(d_pBox);
    cudaFree(d_RNG_vals);

    int count = 0;
    for (auto &p : particle_list) {
      if (p.state == ACTIVE) { count++; }
    }
    std::cout << "--------------------------------------" << std::endl;
    std::cout << "buffer status: " << h_lower_count << " " << h_upper_count << " " << length << std::endl;
    print_percentage((float)h_lower_count / (float)length);
    std::cout << std::endl;
    std::cout << "--------------------------------------" << std::endl;
    std::cout << "number of active particle: " << count << std::endl;

    print_particle(particle_list[0]);
    print_particle(particle_list[1]);

    // source info (hard coded because no way to access the source info here)
    float xS = 20;
    float yS = 50;
    float zS = 70;
    float Q = new_particle;// #par/s (source strength)
    float tRelease = 2100;// total time of release
    float Ntot = Q * tRelease;// total number of particles

    float CNorm = (uMean * H * H / Q);

    float dt = timeStep;
    float tAvg = ongoingAveragingTime;

    // normalization of particle count #particle -> time-averaged # particle/m3
    float CC = dt / tAvg / volume;

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

    // check the results
    std::cout << "--------------------------------------------------------------" << std::endl;
    std::cout << "TEST RESULTS" << std::endl;
    std::cout << "--------------------------------------------------------------" << std::endl;
    std::cout << "x / H \t maxRelErr \t relRMSE" << std::endl;
    for (auto k = 0u; k < xoH.size(); ++k) {
      printf("%.3f \t %.6f \t %.6f\n", xoH[k], maxRelErr[k], relRMSE[k]);
      // std::cout << xoH[k] << " " << maxRelErr[k] << "\t " << relRMSE[k] << std::endl;
    }
    std::cout << std::endl;
    std::cout << "QES-Plume r2 coeff: r2 = " << R2 << std::endl;
    std::cout << "--------------------------------------------------------------" << std::endl;

    std::cout << "--------------------------------------" << std::endl;
    std::cout << "interpolation elapsed time: " << interpElapsed.count() << " s\n";
    std::cout << "advection elapsed time: " << advectElapsed.count() << " s\n";
    std::cout << "concentration elapsed time: " << concenElapsed.count() << " s\n";
    std::cout << "kernel elapsed time: " << kernelElapsed.count() << " s\n";
    std::cout << "GPU elapsed time:    " << gpuElapsed.count() << " s\n";
  } else {
    printf("CUDA ERROR!\n");
  }
}
