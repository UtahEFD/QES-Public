#include "CUDA_QES_Data.h"


void copy_data_gpu(const int &num_cell, QESWindsData &d_qes_winds_data)
{
  std::vector<float> data;
  data.resize(num_cell, 1.0);

  std::fill(data.begin(), data.end(), 1.0);
  cudaMalloc((void **)&d_qes_winds_data.u, num_cell * sizeof(float));
  cudaMemcpy(d_qes_winds_data.u, data.data(), num_cell * sizeof(float), cudaMemcpyHostToDevice);
  std::fill(data.begin(), data.end(), 0.0);
  cudaMalloc((void **)&d_qes_winds_data.v, num_cell * sizeof(float));
  cudaMemcpy(d_qes_winds_data.v, data.data(), num_cell * sizeof(float), cudaMemcpyHostToDevice);
  std::fill(data.begin(), data.end(), 0.0);
  cudaMalloc((void **)&d_qes_winds_data.w, num_cell * sizeof(float));
  cudaMemcpy(d_qes_winds_data.w, data.data(), num_cell * sizeof(float), cudaMemcpyHostToDevice);
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
