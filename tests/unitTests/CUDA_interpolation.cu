
#include "CUDA_interpolation.cuh"

__device__ void setInterp3Dindex_uFace(vec3 &pos, interpWeight &wgt, const QESgrid &qes_grid)
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


__device__ void setInterp3Dindex_vFace(vec3 &pos, interpWeight &wgt, const QESgrid &qes_grid)
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

__device__ void setInterp3Dindex_wFace(vec3 &pos, interpWeight &wgt, const QESgrid &qes_grid)
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
__device__ void interp3D_faceVar(float &out, const float *data, const interpWeight &wgt, const QESgrid &qes_grid)
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

__device__ void setInterp3Dindex_cellVar(const vec3 &pos, interpWeight &wgt, const QESgrid &qes_grid)
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
__device__ void interp3D_cellVar(float &out, const float *data, const interpWeight &wgt, const QESgrid &qes_grid)
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

__global__ void interpolate(int length, particle_array d_particle_list, const QESWindsData data, const QESgrid &qes_grid)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < length) {
    if (d_particle_list.state[idx] == ACTIVE) {
      interpWeight wgt{ 0, 0, 0, 0.0, 0.0, 0.0 };

      float u, v, w;

      // set interpolation indexing variables for uFace variables
      setInterp3Dindex_uFace(d_particle_list.pos[idx], wgt, qes_grid);
      // interpolation of variables on uFace
      interp3D_faceVar(u, data.u, wgt, qes_grid);

      // set interpolation indexing variables for vFace variables
      setInterp3Dindex_vFace(d_particle_list.pos[idx], wgt, qes_grid);
      // interpolation of variables on vFace
      interp3D_faceVar(v, data.v, wgt, qes_grid);

      // set interpolation indexing variables for wFace variables
      setInterp3Dindex_wFace(d_particle_list.pos[idx], wgt, qes_grid);
      // interpolation of variables on wFace
      interp3D_faceVar(w, data.w, wgt, qes_grid);

      d_particle_list.velMean[idx] = { u, v, w };
    }
  }
}

__global__ void interpolate(int length, const vec3 *pos, mat3sym *tau, vec3 *sigma, const QESTurbData data, const QESgrid &qes_grid)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < length) {
    // these are the current interp3D variables, as they are used for multiple interpolations for each particle
    interpWeight wgt{ 0, 0, 0, 0.0, 0.0, 0.0 };

    setInterp3Dindex_cellVar(pos[idx], wgt, qes_grid);

    // this is the current reynolds stress tensor
    float txx_out, txy_out, txz_out, tyy_out, tyz_out, tzz_out;
    interp3D_cellVar(txx_out, data.txx, wgt, qes_grid);
    interp3D_cellVar(txy_out, data.txy, wgt, qes_grid);
    interp3D_cellVar(txz_out, data.txz, wgt, qes_grid);
    interp3D_cellVar(tyy_out, data.tyy, wgt, qes_grid);
    interp3D_cellVar(tyz_out, data.tyz, wgt, qes_grid);
    interp3D_cellVar(tzz_out, data.tzz, wgt, qes_grid);
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

__global__ void interpolate(int length, particle_array d_particle_list, const QESTurbData data, const QESgrid &qes_grid)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < length) {
    if (d_particle_list.state[idx] == ACTIVE) {
      // these are the current interp3D variables, as they are used for multiple interpolations for each particle
      interpWeight wgt{ 0, 0, 0, 0.0, 0.0, 0.0 };

      setInterp3Dindex_cellVar(d_particle_list.pos[idx], wgt, qes_grid);

      // this is the CoEps for the particle
      float CoEps_out;
      interp3D_cellVar(CoEps_out, data.CoEps, wgt, qes_grid);
      // make sure CoEps is always bigger than zero
      if (CoEps_out <= 1e-6) {
        CoEps_out = 1e-6;
      }
      d_particle_list.CoEps[idx] = CoEps_out;

      // this is the current reynolds stress tensor
      float txx_out, txy_out, txz_out, tyy_out, tyz_out, tzz_out;
      interp3D_cellVar(txx_out, data.txx, wgt, qes_grid);
      interp3D_cellVar(txy_out, data.txy, wgt, qes_grid);
      interp3D_cellVar(txz_out, data.txz, wgt, qes_grid);
      interp3D_cellVar(tyy_out, data.tyy, wgt, qes_grid);
      interp3D_cellVar(tyz_out, data.tyz, wgt, qes_grid);
      interp3D_cellVar(tzz_out, data.tzz, wgt, qes_grid);
      d_particle_list.tau[idx] = { txx_out, txy_out, txz_out, tyy_out, tyz_out, tzz_out };

      float flux_div_x_out, flux_div_y_out, flux_div_z_out;
      interp3D_cellVar(flux_div_x_out, data.div_tau_x, wgt, qes_grid);
      interp3D_cellVar(flux_div_y_out, data.div_tau_y, wgt, qes_grid);
      interp3D_cellVar(flux_div_z_out, data.div_tau_z, wgt, qes_grid);
      d_particle_list.flux_div[idx] = { flux_div_x_out, flux_div_y_out, flux_div_z_out };

      float nuT_out;
      interp3D_cellVar(nuT_out, data.nuT, wgt, qes_grid);
      d_particle_list.nuT[idx] = nuT_out;
    }
  }
}


__global__ void interpolate_1(int length, particle_array d_particle_list, const QESTurbData data, const QESgrid &qes_grid)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < length) {
    if (d_particle_list.state[idx] == ACTIVE) {
      // these are the current interp3D variables, as they are used for multiple interpolations for each particle
      interpWeight wgt{ 0, 0, 0, 0.0, 0.0, 0.0 };

      setInterp3Dindex_cellVar(d_particle_list.pos[idx], wgt, qes_grid);

      // this is the CoEps for the particle
      float CoEps_out;
      interp3D_cellVar(CoEps_out, data.CoEps, wgt, qes_grid);
      // make sure CoEps is always bigger than zero
      if (CoEps_out <= 1e-6) {
        CoEps_out = 1e-6;
      }
      d_particle_list.CoEps[idx] = CoEps_out;

      float nuT_out;
      interp3D_cellVar(nuT_out, data.nuT, wgt, qes_grid);
      d_particle_list.nuT[idx] = nuT_out;
    }
  }
}

__global__ void interpolate_2(int length, particle_array d_particle_list, const QESTurbData data, const QESgrid &qes_grid)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < length) {
    if (d_particle_list.state[idx] == ACTIVE) {
      // these are the current interp3D variables, as they are used for multiple interpolations for each particle
      interpWeight wgt{ 0, 0, 0, 0.0, 0.0, 0.0 };

      setInterp3Dindex_cellVar(d_particle_list.pos[idx], wgt, qes_grid);

      // this is the current reynolds stress tensor
      float txx_out, txy_out, txz_out, tyy_out, tyz_out, tzz_out;
      interp3D_cellVar(txx_out, data.txx, wgt, qes_grid);
      interp3D_cellVar(txy_out, data.txy, wgt, qes_grid);
      interp3D_cellVar(txz_out, data.txz, wgt, qes_grid);
      interp3D_cellVar(tyy_out, data.tyy, wgt, qes_grid);
      interp3D_cellVar(tyz_out, data.tyz, wgt, qes_grid);
      interp3D_cellVar(tzz_out, data.tzz, wgt, qes_grid);
      d_particle_list.tau[idx] = { txx_out, txy_out, txz_out, tyy_out, tyz_out, tzz_out };
    }
  }
}


__global__ void interpolate_3(int length, particle_array d_particle_list, const QESTurbData data, const QESgrid &qes_grid)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < length) {
    if (d_particle_list.state[idx] == ACTIVE) {
      // these are the current interp3D variables, as they are used for multiple interpolations for each particle
      interpWeight wgt{ 0, 0, 0, 0.0, 0.0, 0.0 };

      setInterp3Dindex_cellVar(d_particle_list.pos[idx], wgt, qes_grid);

      float flux_div_x_out, flux_div_y_out, flux_div_z_out;
      interp3D_cellVar(flux_div_x_out, data.div_tau_x, wgt, qes_grid);
      interp3D_cellVar(flux_div_y_out, data.div_tau_y, wgt, qes_grid);
      interp3D_cellVar(flux_div_z_out, data.div_tau_z, wgt, qes_grid);
      d_particle_list.flux_div[idx] = { flux_div_x_out, flux_div_y_out, flux_div_z_out };
    }
  }
}
