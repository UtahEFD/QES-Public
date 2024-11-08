
#include "CUDA_plume_testkernel.h"

#include "plume/cuda/QES_data.h"
#include "plume/cuda/Interpolation.h"
#include "plume/cuda/Partition.h"
#include "plume/cuda/RandomGenerator.h"
#include "plume/cuda/Model.h"
#include "plume/cuda/Concentration.h"

#define PBSTR "||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||"
#define PBWIDTH 60

//__device__ __managed__ QESgrid qes_grid;
//__device__ __managed__ BC_Params bc_param;
//__device__ __managed__ ConcentrationParam param;

void copy_data_gpu(const WINDSGeneralData *WGD, QESWindsData &d_qes_winds_data)
{
  // velocity field components
  long numcell_face = WGD->domain.numFaceCentered();
  cudaMalloc((void **)&d_qes_winds_data.u, numcell_face * sizeof(float));
  cudaMemcpy(d_qes_winds_data.u, WGD->u.data(), numcell_face * sizeof(float), cudaMemcpyHostToDevice);
  cudaMalloc((void **)&d_qes_winds_data.v, numcell_face * sizeof(float));
  cudaMemcpy(d_qes_winds_data.v, WGD->v.data(), numcell_face * sizeof(float), cudaMemcpyHostToDevice);
  cudaMalloc((void **)&d_qes_winds_data.w, numcell_face * sizeof(float));
  cudaMemcpy(d_qes_winds_data.w, WGD->w.data(), numcell_face * sizeof(float), cudaMemcpyHostToDevice);
}

void copy_data_gpu(const TURBGeneralData *TGD, QESTurbData &d_qes_turb_data)
{
  // stress tensor
  long numcell_cent = TGD->domain.numCellCentered();
  cudaMalloc((void **)&d_qes_turb_data.txx, numcell_cent * sizeof(float));
  cudaMemcpy(d_qes_turb_data.txx, TGD->txx.data(), numcell_cent * sizeof(float), cudaMemcpyHostToDevice);
  cudaMalloc((void **)&d_qes_turb_data.txy, numcell_cent * sizeof(float));
  cudaMemcpy(d_qes_turb_data.txy, TGD->txy.data(), numcell_cent * sizeof(float), cudaMemcpyHostToDevice);
  cudaMalloc((void **)&d_qes_turb_data.txz, numcell_cent * sizeof(float));
  cudaMemcpy(d_qes_turb_data.txz, TGD->txz.data(), numcell_cent * sizeof(float), cudaMemcpyHostToDevice);
  cudaMalloc((void **)&d_qes_turb_data.tyy, numcell_cent * sizeof(float));
  cudaMemcpy(d_qes_turb_data.tyy, TGD->tyy.data(), numcell_cent * sizeof(float), cudaMemcpyHostToDevice);
  cudaMalloc((void **)&d_qes_turb_data.tyz, numcell_cent * sizeof(float));
  cudaMemcpy(d_qes_turb_data.tyz, TGD->tyz.data(), numcell_cent * sizeof(float), cudaMemcpyHostToDevice);
  cudaMalloc((void **)&d_qes_turb_data.tzz, numcell_cent * sizeof(float));
  cudaMemcpy(d_qes_turb_data.tzz, TGD->tzz.data(), numcell_cent * sizeof(float), cudaMemcpyHostToDevice);

  // divergence of stress tensor
  cudaMalloc((void **)&d_qes_turb_data.div_tau_x, numcell_cent * sizeof(float));
  cudaMemcpy(d_qes_turb_data.div_tau_x, TGD->div_tau_x.data(), numcell_cent * sizeof(float), cudaMemcpyHostToDevice);
  cudaMalloc((void **)&d_qes_turb_data.div_tau_y, numcell_cent * sizeof(float));
  cudaMemcpy(d_qes_turb_data.div_tau_y, TGD->div_tau_y.data(), numcell_cent * sizeof(float), cudaMemcpyHostToDevice);
  cudaMalloc((void **)&d_qes_turb_data.div_tau_z, numcell_cent * sizeof(float));
  cudaMemcpy(d_qes_turb_data.div_tau_z, TGD->div_tau_z.data(), numcell_cent * sizeof(float), cudaMemcpyHostToDevice);

  // dissipation rate
  cudaMalloc((void **)&d_qes_turb_data.CoEps, numcell_cent * sizeof(float));
  cudaMemcpy(d_qes_turb_data.CoEps, TGD->CoEps.data(), numcell_cent * sizeof(float), cudaMemcpyHostToDevice);

  // turbulent viscosity
  cudaMalloc((void **)&d_qes_turb_data.nuT, numcell_cent * sizeof(float));
  cudaMemcpy(d_qes_turb_data.nuT, TGD->nuT.data(), numcell_cent * sizeof(float), cudaMemcpyHostToDevice);

  // turbulence kinetic energy
  cudaMalloc((void **)&d_qes_turb_data.tke, numcell_cent * sizeof(float));
  cudaMemcpy(d_qes_turb_data.tke, TGD->tke.data(), numcell_cent * sizeof(float), cudaMemcpyHostToDevice);
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
  Timer sourceTimer("new particles creation");
  Timer partitonTimer("particle partitioning");
  Timer interpTimer("interpolation");
  Timer advectTimer("advection");
  Timer concenTimer("concentration");

  float uMean = 2.0;
  float uStar = 0.174;
  float H = 70;
  float C0 = 5.7;
  float zi = 1000;

  QESgrid qes_grid;
  BC_Params bc_param;
  ConcentrationParam param;

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


    auto *random = new RandomGenerator();
    auto *interpolation = new Interpolation();
    auto *model = new Model();

    // IDGenerator *id_gen;
    // id_gen = IDGenerator::getInstance();

    QESWindsData d_qes_winds_data;
    copy_data_gpu(WGD, d_qes_winds_data);
    // WGD->allocateDevice();
    // WGD->copyDataToDevice();

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

    auto *concentration = new Concentration(param);
    auto *partition = new Partition(length);

    // Allocate particle array on the device ONLY
    particle_array d_particle[2];
    partition->allocate_device_particle_list(d_particle[0], length);
    partition->allocate_device_particle_list(d_particle[1], length);

    random->create("advect", 3 * length);

    gpuInitTimer.stop();

    float timeStep = 1.0;
    int idx = 0;

    timeLoopTimer.start();
    // call kernel
    std::cout << "-------------------------------------------------------------------" << std::endl;
    std::cout << "buffer usage: " << std::endl;
    for (int k = 0; k < ntest; ++k) {

      // int num_particle = length;
      int num_particle = partition->active() + new_particle;
      // std::cout << num_particle << std::endl;

      partitonTimer.start();
      idx = partition->run(k, d_particle);
      partitonTimer.stop();


      sourceTimer.start();
      model->getNewParticle(new_particle, d_particle[idx], d_qes_turb_data, qes_grid, random, interpolation, partition);
      sourceTimer.stop();

      interpTimer.start();
      interpolation->get(d_particle[idx], d_qes_winds_data, d_qes_turb_data, qes_grid, num_particle);
      interpTimer.stop();

      advectTimer.start();
      model->advectParticle(d_particle[idx], num_particle, bc_param, random);
      advectTimer.stop();


      if (k >= 1000) {
        concenTimer.start();
        concentration->collect(timeStep, d_particle[idx], num_particle);
        concenTimer.stop();
      }

      // print buffer status
      print_percentage((float)partition->active() / (float)length);
    }
    std::cout << std::endl;
    timeLoopTimer.stop();

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

    concentration->copyback();

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

    WGD->freeDevice();

    delete partition;
    delete random;
    delete interpolation;
    delete model;

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
    float tAvg = concentration->ongoingAveragingTime;
    float volume = concentration->volume;

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
      // float To = 1.001;
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
          CStarQES[j + k * param.ny] = concentration->h_pBox[id] * CC * CNorm;
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

    outfile->newField("t_avg", "Averaging time", "s", "time", &concentration->ongoingAveragingTime);
    outfile->newField("p_count", "number of particle per box", "#ofPar", "concentration", &concentration->h_pBox);
    std::vector<float> CStar(param.nx * param.ny * param.nz);
    for (size_t id = 0; id < CStar.size(); ++id) {
      CStar[id] = concentration->h_pBox[id] * CC * CNorm;
    }
    outfile->newField("c", "normailzed concentration", "--", "concentration", &CStar);

    QEStime time;
    outfile->pushAllFieldsToFile(time);

    delete outfile;
    delete concentration;
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
    sourceTimer.show();
    interpTimer.show();
    advectTimer.show();
    concenTimer.show();
    std::cout << "-------------------------------------------------------------------" << std::endl;
  } else {
    printf("CUDA ERROR!\n");
  }
}
