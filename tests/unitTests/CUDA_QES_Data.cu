#include "CUDA_QES_Data.h"


void copy_data_gpu(const int &num_face, QESWindsData &d_qes_winds_data)
{
  std::vector<float> data;
  data.resize(num_face, 1.0);

  std::fill(data.begin(), data.end(), 1.0);
  cudaMalloc((void **)&d_qes_winds_data.u, num_face * sizeof(float));
  cudaMemcpy(d_qes_winds_data.u, data.data(), num_face * sizeof(float), cudaMemcpyHostToDevice);
  std::fill(data.begin(), data.end(), 0.0);
  cudaMalloc((void **)&d_qes_winds_data.v, num_face * sizeof(float));
  cudaMemcpy(d_qes_winds_data.v, data.data(), num_face * sizeof(float), cudaMemcpyHostToDevice);
  std::fill(data.begin(), data.end(), 0.0);
  cudaMalloc((void **)&d_qes_winds_data.w, num_face * sizeof(float));
  cudaMemcpy(d_qes_winds_data.w, data.data(), num_face * sizeof(float), cudaMemcpyHostToDevice);
}

void copy_data_gpu(const int &num_cell, QESTurbData &d_qes_turb_data)
{
  std::vector<float> data;
  data.resize(num_cell, 1.0);

  std::fill(data.begin(), data.end(), 1.0);
  cudaMalloc((void **)&d_qes_turb_data.txx, num_cell * sizeof(float));
  cudaMemcpy(d_qes_turb_data.txx, data.data(), num_cell * sizeof(float), cudaMemcpyHostToDevice);

  std::fill(data.begin(), data.end(), 0.0);
  cudaMalloc((void **)&d_qes_turb_data.txy, num_cell * sizeof(float));
  cudaMemcpy(d_qes_turb_data.txy, data.data(), num_cell * sizeof(float), cudaMemcpyHostToDevice);

  std::fill(data.begin(), data.end(), 0.0);
  cudaMalloc((void **)&d_qes_turb_data.txz, num_cell * sizeof(float));
  cudaMemcpy(d_qes_turb_data.txz, data.data(), num_cell * sizeof(float), cudaMemcpyHostToDevice);

  std::fill(data.begin(), data.end(), 1.0);
  cudaMalloc((void **)&d_qes_turb_data.tyy, num_cell * sizeof(float));
  cudaMemcpy(d_qes_turb_data.tyy, data.data(), num_cell * sizeof(float), cudaMemcpyHostToDevice);

  std::fill(data.begin(), data.end(), 0.0);
  cudaMalloc((void **)&d_qes_turb_data.tyz, num_cell * sizeof(float));
  cudaMemcpy(d_qes_turb_data.tyz, data.data(), num_cell * sizeof(float), cudaMemcpyHostToDevice);

  std::fill(data.begin(), data.end(), 1.0);
  cudaMalloc((void **)&d_qes_turb_data.tzz, num_cell * sizeof(float));
  cudaMemcpy(d_qes_turb_data.tzz, data.data(), num_cell * sizeof(float), cudaMemcpyHostToDevice);

  std::fill(data.begin(), data.end(), 0.0);
  cudaMalloc((void **)&d_qes_turb_data.div_tau_x, num_cell * sizeof(float));
  cudaMemcpy(d_qes_turb_data.div_tau_x, data.data(), num_cell * sizeof(float), cudaMemcpyHostToDevice);
  cudaMalloc((void **)&d_qes_turb_data.div_tau_y, num_cell * sizeof(float));
  cudaMemcpy(d_qes_turb_data.div_tau_y, data.data(), num_cell * sizeof(float), cudaMemcpyHostToDevice);
  cudaMalloc((void **)&d_qes_turb_data.div_tau_z, num_cell * sizeof(float));
  cudaMemcpy(d_qes_turb_data.div_tau_z, data.data(), num_cell * sizeof(float), cudaMemcpyHostToDevice);

  std::fill(data.begin(), data.end(), 0.1);
  cudaMalloc((void **)&d_qes_turb_data.CoEps, num_cell * sizeof(float));
  cudaMemcpy(d_qes_turb_data.CoEps, data.data(), num_cell * sizeof(float), cudaMemcpyHostToDevice);
  cudaMalloc((void **)&d_qes_turb_data.nuT, num_cell * sizeof(float));
  cudaMemcpy(d_qes_turb_data.nuT, data.data(), num_cell * sizeof(float), cudaMemcpyHostToDevice);
}

void copy_data_gpu(const WINDSGeneralData *WGD, QESWindsData &d_qes_winds_data)
{
  // velocity field components
  cudaMalloc((void **)&d_qes_winds_data.u, WGD->numcell_face * sizeof(float));
  cudaMemcpy(d_qes_winds_data.u, WGD->u.data(), WGD->numcell_face * sizeof(float), cudaMemcpyHostToDevice);
  cudaMalloc((void **)&d_qes_winds_data.v, WGD->numcell_face * sizeof(float));
  cudaMemcpy(d_qes_winds_data.v, WGD->v.data(), WGD->numcell_face * sizeof(float), cudaMemcpyHostToDevice);
  cudaMalloc((void **)&d_qes_winds_data.w, WGD->numcell_face * sizeof(float));
  cudaMemcpy(d_qes_winds_data.w, WGD->w.data(), WGD->numcell_face * sizeof(float), cudaMemcpyHostToDevice);
}

void copy_data_gpu(const TURBGeneralData *TGD, QESTurbData &d_qes_turb_data)
{

  // stress tensor
  cudaMalloc((void **)&d_qes_turb_data.txx, TGD->numcell_cent * sizeof(float));
  cudaMemcpy(d_qes_turb_data.txx, TGD->txx.data(), TGD->numcell_cent * sizeof(float), cudaMemcpyHostToDevice);
  cudaMalloc((void **)&d_qes_turb_data.txy, TGD->numcell_cent * sizeof(float));
  cudaMemcpy(d_qes_turb_data.txy, TGD->txy.data(), TGD->numcell_cent * sizeof(float), cudaMemcpyHostToDevice);
  cudaMalloc((void **)&d_qes_turb_data.txz, TGD->numcell_cent * sizeof(float));
  cudaMemcpy(d_qes_turb_data.txz, TGD->txz.data(), TGD->numcell_cent * sizeof(float), cudaMemcpyHostToDevice);
  cudaMalloc((void **)&d_qes_turb_data.tyy, TGD->numcell_cent * sizeof(float));
  cudaMemcpy(d_qes_turb_data.tyy, TGD->tyy.data(), TGD->numcell_cent * sizeof(float), cudaMemcpyHostToDevice);
  cudaMalloc((void **)&d_qes_turb_data.tyz, TGD->numcell_cent * sizeof(float));
  cudaMemcpy(d_qes_turb_data.tyz, TGD->tyz.data(), TGD->numcell_cent * sizeof(float), cudaMemcpyHostToDevice);
  cudaMalloc((void **)&d_qes_turb_data.tzz, TGD->numcell_cent * sizeof(float));
  cudaMemcpy(d_qes_turb_data.tzz, TGD->tzz.data(), TGD->numcell_cent * sizeof(float), cudaMemcpyHostToDevice);

  // divergence of stress tensor
  cudaMalloc((void **)&d_qes_turb_data.div_tau_x, TGD->numcell_cent * sizeof(float));
  cudaMemcpy(d_qes_turb_data.div_tau_x, TGD->div_tau_x.data(), TGD->numcell_cent * sizeof(float), cudaMemcpyHostToDevice);
  cudaMalloc((void **)&d_qes_turb_data.div_tau_y, TGD->numcell_cent * sizeof(float));
  cudaMemcpy(d_qes_turb_data.div_tau_y, TGD->div_tau_y.data(), TGD->numcell_cent * sizeof(float), cudaMemcpyHostToDevice);
  cudaMalloc((void **)&d_qes_turb_data.div_tau_z, TGD->numcell_cent * sizeof(float));
  cudaMemcpy(d_qes_turb_data.div_tau_z, TGD->div_tau_z.data(), TGD->numcell_cent * sizeof(float), cudaMemcpyHostToDevice);

  // dissipation rate
  cudaMalloc((void **)&d_qes_turb_data.CoEps, TGD->numcell_cent * sizeof(float));
  cudaMemcpy(d_qes_turb_data.CoEps, TGD->CoEps.data(), TGD->numcell_cent * sizeof(float), cudaMemcpyHostToDevice);

  // turbulent viscosity
  cudaMalloc((void **)&d_qes_turb_data.nuT, TGD->numcell_cent * sizeof(float));
  cudaMemcpy(d_qes_turb_data.nuT, TGD->nuT.data(), TGD->numcell_cent * sizeof(float), cudaMemcpyHostToDevice);

  // turbulence kinetic energy
  cudaMalloc((void **)&d_qes_turb_data.tke, TGD->numcell_cent * sizeof(float));
  cudaMemcpy(d_qes_turb_data.tke, TGD->tke.data(), TGD->numcell_cent * sizeof(float), cudaMemcpyHostToDevice);
}
