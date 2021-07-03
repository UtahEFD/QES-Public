
#include "Eulerian.h"


Eulerian::Eulerian(PlumeInputData *PID, WINDSGeneralData *WGD, TURBGeneralData *TGD, const bool &debug_val)
{

  std::cout << "[Eulerian] \t Setting Eulerian fields " << std::endl;

  // copy debug information
  debug = debug_val;

  // copy WGD grid information
  nz = WGD->nz;
  ny = WGD->ny;
  nx = WGD->nx;

  dz = WGD->dz;
  dy = WGD->dy;
  dx = WGD->dx;

  // domain beginning for interpolation in each direction
  // in x-direction (halo cell to account for TURB variables)
  iStart = 1;
  iEnd = nx - 3;
  // in y-direction (halo cell to account for TURB variables)
  jStart = 1;
  jEnd = ny - 3;
  // in z-direction (ghost cell at bottom and halo cell at top)
  kStart = 1;
  kEnd = nz - 2;

  // get the TGD domain start and end values, other TGD grid information
  // in x-direction (face)
  xStart = WGD->x[iStart] - 0.5 * dx;
  xEnd = WGD->x[iEnd] + 0.5 * dx;

  // in y-direction (face)
  yStart = WGD->y[jStart] - 0.5 * dy;
  yEnd = WGD->y[jEnd] + 0.5 * dy;
  // in z-direction (face)
  zStart = WGD->z_face[kStart - 1];// z_face does not have a ghost cell under the terrain.
  zEnd = WGD->z_face[kEnd - 1];// z_face does not have a ghost cell under the terrain.


  if (debug == true) {
    std::cout << "[Eulerian] \t DEBUG - Domain boundary" << std::endl;
    std::cout << "\t\t xStart=" << xStart << " xEnd=" << xEnd << std::endl;
    std::cout << "\t\t yStart=" << yStart << " yEnd=" << yEnd << std::endl;
    std::cout << "\t\t zStart=" << zStart << " zEnd=" << zEnd << std::endl;
  }

  // set additional values from the input
  C_0 = PID->simParams->C_0;

  // set the tau gradient sizes
  dtxxdx.resize(WGD->numcell_face, 0.0);
  dtxydy.resize(WGD->numcell_face, 0.0);
  dtxzdz.resize(WGD->numcell_face, 0.0);

  dtxydx.resize(WGD->numcell_face, 0.0);
  dtyydy.resize(WGD->numcell_face, 0.0);
  dtyzdz.resize(WGD->numcell_face, 0.0);

  dtxzdx.resize(WGD->numcell_face, 0.0);
  dtyzdy.resize(WGD->numcell_face, 0.0);
  dtzzdz.resize(WGD->numcell_face, 0.0);

  // temp storage of sigma's
  sig_x.resize(WGD->numcell_cent, 0.0);
  sig_y.resize(WGD->numcell_cent, 0.0);
  sig_z.resize(WGD->numcell_cent, 0.0);
}

void Eulerian::setData(WINDSGeneralData *WGD, TURBGeneralData *TGD)
{
  // set BC
  setBC(WGD, TGD);

  // compute stress gradients
  setStressGradient(TGD);

  // temporary copy of sigma's
  setSigmas(TGD);

  // calculate the threshold velocity
  vel_threshold = 10.0 * std::sqrt(getMaxVariance(sig_x, sig_y, sig_z));
}

void Eulerian::setBC(WINDSGeneralData *WGD, TURBGeneralData *TGD)
{
  std::cout << "[Eulerian] \t Correction QES-winds fields for BC" << std::endl;

  // verical surface (wall right => j-1)
  for (size_t id; id < WGD->wall_right_indices.size(); ++id) {
    int idface = WGD->wall_right_indices[id];
    // u(i,j-1,k)=-u(i,j,k)
    WGD->u[idface - nx] = -WGD->u[idface];
    // u(i+1,j-1,k)=-u(i+1,j,k)
    WGD->u[idface + 1 - nx] = -WGD->u[idface + 1];

    // w(i,j-1,k)=-w(i,j,k)
    WGD->w[idface - nx] = -WGD->w[idface];
    // w(i,j-1,k+1)=-w(i,j,k+1)
    WGD->w[idface + nx * ny - nx] = -WGD->w[idface + nx * ny];
  }

  // verical surface (wall left => j+1)
  for (size_t id; id < WGD->wall_left_indices.size(); ++id) {
    int idface = WGD->wall_left_indices[id];
    // u(i,j+1,k)=-u(i,j,k)
    WGD->u[idface + nx] = -WGD->u[idface];
    // u(i+1,j+1,k)=-u(i+1,j,k)
    WGD->u[idface + 1 + nx] = -WGD->u[idface + 1];

    // w(i,j+1,k)=-w(i,j,k)
    WGD->w[idface + nx] = -WGD->w[idface];
    // w(i,j+1,k+1)=-w(i,j,k+1)
    WGD->w[idface + nx * ny + nx] = -WGD->w[idface + nx * ny];
  }

  // horizontal surface (wall above => k+1)
  for (size_t id; id < WGD->wall_above_indices.size(); ++id) {
    int idface = WGD->wall_above_indices[id];
    // u(i,j,k+1)=-u(i,j,k)
    WGD->u[idface + nx * ny] = -WGD->u[idface];
    // u(i+1,j,k+1)=-u(i+1,j,k)
    WGD->u[idface + 1 + nx * ny] = -WGD->u[idface + 1];

    // v(i,j,k+1)=-v(i,j,k)
    WGD->v[idface + nx * ny] = -WGD->v[idface];
    // v(i,j+1,k+1)=-v(i,j+1,k)
    WGD->v[idface + nx + nx * ny] = -WGD->v[idface + nx];
  }

  // horizontal surface (wall below => k-1)
  for (size_t id; id < WGD->wall_below_indices.size(); ++id) {
    int idface = WGD->wall_below_indices[id];
    // u(i,j,k-1)=-u(i,j,k)
    WGD->u[idface - nx * ny] = -WGD->u[idface];
    // u(i+1,j,k+1)=-u(i+1,j,k)
    WGD->u[idface + 1 - nx * ny] = -WGD->u[idface + 1];

    // v(i,j,k-1)=-v(i,j,k)
    WGD->v[idface - nx * ny] = -WGD->v[idface];
    // v(i,j+1,k-1)=-v(i,j+1,k)
    WGD->v[idface + nx - nx * ny] = -WGD->v[idface + nx];
  }


  // verical surface (wall back => i-1)
  for (size_t id; id < WGD->wall_back_indices.size(); ++id) {
    int idface = WGD->wall_back_indices[id];
    // v(i-1,j,k)=-v(i,j,k)
    WGD->v[idface - 1] = -WGD->v[idface];
    // v(i-1,j+1,k)=-v(i,j+1,k)
    WGD->v[idface + nx - 1] = -WGD->v[idface + nx];

    // w(i-1,j,k)=-w(i,j,k)
    WGD->w[idface - 1] = -WGD->w[idface];
    // w(i-1,j,k+1)=-w(i,j,k+1)
    WGD->w[idface + nx * ny - 1] = -WGD->w[idface + nx * ny];
  }

  // verical surface (wall front => i+1)
  for (size_t id; id < WGD->wall_front_indices.size(); ++id) {
    int idface = WGD->wall_front_indices[id];
    // v(i+1,j,k)=-v(i,j,k)
    WGD->v[idface + 1] = -WGD->v[idface];
    // v(i+1,j+1,k)=-v(i,j+1,k)
    WGD->v[idface + nx + 1] = -WGD->v[idface + nx];

    // w(i+1,j,k)=-w(i,j,k)
    WGD->w[idface + 1] = -WGD->w[idface];
    // w(i+1,j,k+1)=-w(i,j,k+1)
    WGD->w[idface + nx * ny + 1] = -WGD->w[idface + nx * ny];
  }

  std::cout << "[Eulerian] \t Correction QES-turb fields for BC" << std::endl;

  // verical surface (wall right => j-1)
  for (size_t id; id < WGD->wall_right_indices.size(); ++id) {
    int idface = WGD->wall_right_indices[id];
    // i,j,k -> inverted linearized index
    int k = (int)(idface / ((nx * nx)));
    int j = (int)((idface - k * (nx * ny)) / (nx));
    int i = idface - j * (nx)-k * (nx * ny);
    // id of cell
    int idcell = i + j * (nx - 1) + k * (nx - 1) * (ny - 1);
    int idcell_m = idcell - (nx - 1);

    TGD->CoEps[idcell_m] = TGD->CoEps[idcell];

    TGD->txx[idcell_m] = TGD->txx[idcell];
    TGD->txy[idcell_m] = TGD->txy[idcell];
    TGD->txz[idcell_m] = TGD->txz[idcell];
    TGD->tyy[idcell_m] = TGD->tyy[idcell];
    TGD->tyz[idcell_m] = TGD->tyz[idcell];
    TGD->tzz[idcell_m] = TGD->tzz[idcell];
  }

  // verical surface (wall left => j+1)
  for (size_t id; id < WGD->wall_left_indices.size(); ++id) {
    int idface = WGD->wall_left_indices[id];
    // i,j,k -> inverted linearized index
    int k = (int)(idface / ((nx * nx)));
    int j = (int)((idface - k * (nx * ny)) / (nx));
    int i = idface - j * (nx)-k * (nx * ny);
    // id of cell
    int idcell = i + j * (nx - 1) + k * (nx - 1) * (ny - 1);
    int idcell_m = idcell + (nx - 1);

    TGD->CoEps[idcell_m] = TGD->CoEps[idcell];

    TGD->txx[idcell_m] = TGD->txx[idcell];
    TGD->txy[idcell_m] = TGD->txy[idcell];
    TGD->txz[idcell_m] = TGD->txz[idcell];
    TGD->tyy[idcell_m] = TGD->tyy[idcell];
    TGD->tyz[idcell_m] = TGD->tyz[idcell];
    TGD->tzz[idcell_m] = TGD->tzz[idcell];
  }

  // horizontal surface (wall above => k+1)
  for (size_t id; id < WGD->wall_above_indices.size(); ++id) {
    int idface = WGD->wall_above_indices[id];
    // i,j,k -> inverted linearized index
    int k = (int)(idface / ((nx * nx)));
    int j = (int)((idface - k * (nx * ny)) / (nx));
    int i = idface - j * (nx)-k * (nx * ny);
    // id of cell
    int idcell = i + j * (nx - 1) + k * (nx - 1) * (ny - 1);
    int idcell_m = idcell + (nx - 1) * (ny - 1);

    TGD->CoEps[idcell_m] = TGD->CoEps[idcell];

    TGD->txx[idcell_m] = TGD->txx[idcell];
    TGD->txy[idcell_m] = TGD->txy[idcell];
    TGD->txz[idcell_m] = TGD->txz[idcell];
    TGD->tyy[idcell_m] = TGD->tyy[idcell];
    TGD->tyz[idcell_m] = TGD->tyz[idcell];
    TGD->tzz[idcell_m] = TGD->tzz[idcell];
  }

  // horizontal surface (wall below => k-1)
  for (size_t id; id < WGD->wall_below_indices.size(); ++id) {
    int idface = WGD->wall_below_indices[id];
    // i,j,k -> inverted linearized index
    int k = (int)(idface / ((nx * nx)));
    int j = (int)((idface - k * (nx * ny)) / (nx));
    int i = idface - j * (nx)-k * (nx * ny);
    // id of cell
    int idcell = i + j * (nx - 1) + k * (nx - 1) * (ny - 1);
    int idcell_m = idcell - (nx - 1) * (ny - 1);

    TGD->CoEps[idcell_m] = TGD->CoEps[idcell];

    TGD->txx[idcell_m] = TGD->txx[idcell];
    TGD->txy[idcell_m] = TGD->txy[idcell];
    TGD->txz[idcell_m] = TGD->txz[idcell];
    TGD->tyy[idcell_m] = TGD->tyy[idcell];
    TGD->tyz[idcell_m] = TGD->tyz[idcell];
    TGD->tzz[idcell_m] = TGD->tzz[idcell];
  }


  // verical surface (wall back => i-1)
  for (size_t id; id < WGD->wall_back_indices.size(); ++id) {
    int idface = WGD->wall_back_indices[id];
    // i,j,k -> inverted linearized index
    int k = (int)(idface / ((nx * nx)));
    int j = (int)((idface - k * (nx * ny)) / (nx));
    int i = idface - j * (nx)-k * (nx * ny);
    // id of cell
    int idcell = i + j * (nx - 1) + k * (nx - 1) * (ny - 1);
    int idcell_m = idcell - 1;

    TGD->CoEps[idcell_m] = TGD->CoEps[idcell];

    TGD->txx[idcell_m] = TGD->txx[idcell];
    TGD->txy[idcell_m] = TGD->txy[idcell];
    TGD->txz[idcell_m] = TGD->txz[idcell];
    TGD->tyy[idcell_m] = TGD->tyy[idcell];
    TGD->tyz[idcell_m] = TGD->tyz[idcell];
    TGD->tzz[idcell_m] = TGD->tzz[idcell];
  }

  // verical surface (wall front => i+1)
  for (size_t id; id < WGD->wall_front_indices.size(); ++id) {
    int idface = WGD->wall_front_indices[id];
    // i,j,k -> inverted linearized index
    int k = (int)(idface / ((nx * nx)));
    int j = (int)((idface - k * (nx * ny)) / (nx));
    int i = idface - j * (nx)-k * (nx * ny);
    // id of cell
    int idcell = i + j * (nx - 1) + k * (nx - 1) * (ny - 1);
    int idcell_m = idcell + 1;

    TGD->CoEps[idcell_m] = TGD->CoEps[idcell];

    TGD->txx[idcell_m] = TGD->txx[idcell];
    TGD->txy[idcell_m] = TGD->txy[idcell];
    TGD->txz[idcell_m] = TGD->txz[idcell];
    TGD->tyy[idcell_m] = TGD->tyy[idcell];
    TGD->tyz[idcell_m] = TGD->tyz[idcell];
    TGD->tzz[idcell_m] = TGD->tzz[idcell];
  }

  return;
}

void Eulerian::setStressGradient(TURBGeneralData *TGD)
{
  std::cout << "[Eulerian] \t Computing stress gradients on face" << std::endl;

  for (int k = kStart; k < kEnd + 1; ++k) {
    for (int j = jStart; j < jEnd + 1; ++j) {
      for (int i = iStart; i < iEnd + 1; ++i) {
        // Provides a linear index based on the 3D (i, j, k)
        int cellid = k * (ny - 1) * (nx - 1) + j * (nx - 1) + i;
        int faceid = k * (ny * nx) + j * (nx) + i;

        dtxxdx[faceid] = (TGD->txx[cellid] - TGD->txx[cellid - 1]) / dx;
        dtxydx[faceid] = (TGD->txy[cellid] - TGD->txy[cellid - 1]) / dx;
        dtxzdx[faceid] = (TGD->txz[cellid] - TGD->txz[cellid - 1]) / dx;
      }
    }
  }

  for (int k = kStart; k < kEnd + 1; ++k) {
    for (int j = jStart; j < jEnd + 1; ++j) {
      for (int i = iStart; i < iEnd + 1; ++i) {
        // Provides a linear index based on the 3D (i, j, k)
        int cellid = k * (ny - 1) * (nx - 1) + j * (nx - 1) + i;
        int faceid = k * (ny * nx) + j * (nx) + i;

        dtxydy[faceid] = (TGD->txy[cellid] - TGD->txy[cellid - (nx - 1)]) / dy;
        dtyydy[faceid] = (TGD->tyy[cellid] - TGD->tyy[cellid - (nx - 1)]) / dy;
        dtyzdy[faceid] = (TGD->tyz[cellid] - TGD->tyz[cellid - (nx - 1)]) / dy;
      }
    }
  }

  for (int k = kStart; k < kEnd + 1; ++k) {
    for (int j = jStart; j < jEnd + 1; ++j) {
      for (int i = iStart; i < iEnd + 1; ++i) {
        // Provides a linear index based on the 3D (i, j, k)
        int cellid = k * (ny - 1) * (nx - 1) + j * (nx - 1) + i;
        int faceid = k * (ny * nx) + j * (nx) + i;

        dtxzdz[faceid] = (TGD->txz[cellid] - TGD->txz[cellid - (ny - 1) * (nx - 1)]) / dz;
        dtyzdz[faceid] = (TGD->tyz[cellid] - TGD->tyz[cellid - (ny - 1) * (nx - 1)]) / dz;
        dtzzdz[faceid] = (TGD->tzz[cellid] - TGD->tzz[cellid - (ny - 1) * (nx - 1)]) / dz;
      }
    }
  }

  return;
}

void Eulerian::setStressGrads(TURBGeneralData *TGD)
{
  std::cout << "[Eulerian] \t Computing stress gradients " << std::endl;

  // start recording execution time
  if (debug == true) {
    timers.startNewTimer("calcGradient");
  }

  // 2nd order Forward differencing up to 2 in from the edge in the direction of the gradient,
  // 2nd order Backward differencing for the last two in the direction of the gradient,
  // all this over all cells in the other two directions

  // DX forward differencing
  for (int k = kStart; k < kEnd; ++k) {
    for (int j = jStart; j < jEnd; ++j) {
      for (int i = iStart; i < iEnd - 2; ++i) {
        // Provides a linear index based on the 3D (i, j, k)
        int idx = k * (ny - 1) * (nx - 1) + j * (nx - 1) + i;
        setDX_Forward(TGD, idx);
      }
    }
  }

  // DX backward differencing
  for (int k = kStart; k < kEnd; ++k) {
    for (int j = jStart; j < jEnd; ++j) {
      for (int i = iEnd - 2; i < iEnd; ++i) {
        // Provides a linear index based on the 3D (i, j, k)
        int idx = k * (ny - 1) * (nx - 1) + j * (nx - 1) + i;
        setDX_Backward(TGD, idx);
      }
    }
  }


  // DY forward differencing
  for (int k = kStart; k < kEnd; ++k) {
    for (int j = jStart; j < jEnd - 2; ++j) {
      for (int i = iStart; i < iEnd; ++i) {
        // Provides a linear index based on the 3D (i, j, k)
        int idx = k * (ny - 1) * (nx - 1) + j * (nx - 1) + i;
        setDY_Forward(TGD, idx);
      }
    }
  }

  // DY backward differencing
  for (int k = kStart; k < kEnd; ++k) {
    for (int j = jEnd - 2; j < jEnd; ++j) {
      for (int i = iStart; i < iEnd; ++i) {
        // Provides a linear index based on the 3D (i, j, k)
        int idx = k * (ny - 1) * (nx - 1) + j * (nx - 1) + i;
        setDY_Backward(TGD, idx);
      }
    }
  }

  // DZ forward differencing
  for (int k = kStart; k < kEnd - 2; ++k) {
    for (int j = jStart; j < jEnd; ++j) {
      for (int i = iStart; i < iEnd; ++i) {
        // Provides a linear index based on the 3D (i, j, k)
        int idx = k * (ny - 1) * (nx - 1) + j * (nx - 1) + i;
        setDZ_Forward(TGD, idx);
      }
    }
  }

  // DZ backward differencing
  for (int k = kEnd - 2; k < kEnd; ++k) {
    for (int j = jStart; j < jEnd; ++j) {
      for (int i = iEnd; i < iEnd; ++i) {
        // Provides a  linear index based on the 3D (i, j, k)
        int idx = k * (ny - 1) * (nx - 1) + j * (nx - 1) + i;
        setDZ_Backward(TGD, idx);
      }
    }
  }

  // print out elapsed execution time
  if (debug == true) {
    timers.printStoredTime("calcGradient");
  }
}

void Eulerian::setSigmas(TURBGeneralData *TGD)
{
  for (int idx = 0; idx < (nx - 1) * (ny - 1) * (nz - 1); idx++) {
    sig_x.at(idx) = std::abs(TGD->txx.at(idx));
    sig_y.at(idx) = std::abs(TGD->tyy.at(idx));
    sig_z.at(idx) = std::abs(TGD->tzz.at(idx));
  }
  return;
}

double Eulerian::getMaxVariance(const std::vector<double> &sigma_x_vals, const std::vector<double> &sigma_y_vals, const std::vector<double> &sigma_z_vals)
{
  // set thoe initial maximum value to a very small number. The idea is to go through each value of the data,
  // setting the current value to the max value each time the current value is bigger than the old maximum value
  double maximumVal = -10e-10;

  // go through each vector to find the maximum value
  // each one could potentially be different sizes if the grid is not 3D
  for (size_t idx = 0; idx < sigma_x_vals.size(); idx++) {
    if (sigma_x_vals.at(idx) > maximumVal) {
      maximumVal = sigma_x_vals.at(idx);
    }
  }

  for (size_t idx = 0; idx < sigma_y_vals.size(); idx++) {
    if (sigma_y_vals.at(idx) > maximumVal) {
      maximumVal = sigma_y_vals.at(idx);
    }
  }

  for (size_t idx = 0; idx < sigma_z_vals.size(); idx++) {
    if (sigma_z_vals.at(idx) > maximumVal) {
      maximumVal = sigma_z_vals.at(idx);
    }
  }

  return maximumVal;
}


void Eulerian::setInterp3Dindex_uFace(const double &par_xPos, const double &par_yPos, const double &par_zPos)
{

  // set a particle position corrected by the start of the domain in each direction
  //double par_x = par_xPos - xStart + 1.0*dx;
  //double par_y = par_yPos - yStart + 0.5*dy;
  //double par_z = par_zPos - zStart + 0.5*dz;
  double par_x = par_xPos - 0.0 * dx;
  double par_y = par_yPos - 0.5 * dy;
  double par_z = par_zPos + 0.5 * dz;

  ii = floor(par_x / (dx + 1e-9));
  jj = floor(par_y / (dy + 1e-9));
  kk = floor(par_z / (dz + 1e-9));

  // fractional distance between nearest nodes
  iw = (par_x / dx - floor(par_x / dx));
  jw = (par_y / dy - floor(par_y / dy));
  kw = (par_z / dz - floor(par_z / dz));

  return;
}

void Eulerian::setInterp3Dindex_vFace(const double &par_xPos, const double &par_yPos, const double &par_zPos)
{

  // set a particle position corrected by the start of the domain in each direction
  //double par_x = par_xPos - xStart + 0.5*dx;
  //double par_y = par_yPos - yStart + 1.0*dy;
  //double par_z = par_zPos - zStart + 0.5*dz;
  double par_x = par_xPos - 0.5 * dx;
  double par_y = par_yPos - 0.0 * dy;
  double par_z = par_zPos + 0.5 * dz;

  ii = floor(par_x / (dx + 1e-9));
  jj = floor(par_y / (dy + 1e-9));
  kk = floor(par_z / (dz + 1e-9));

  // fractional distance between nearest nodes
  iw = (par_x / dx - floor(par_x / dx));
  jw = (par_y / dy - floor(par_y / dy));
  kw = (par_z / dz - floor(par_z / dz));

  return;
}

void Eulerian::setInterp3Dindex_wFace(const double &par_xPos, const double &par_yPos, const double &par_zPos)
{

  // set a particle position corrected by the start of the domain in each direction
  //double par_x = par_xPos - xStart + 0.5*dx;
  //double par_y = par_yPos - yStart + 0.5*dy;
  //double par_z = par_zPos - zStart + 1.0*dz;
  double par_x = par_xPos - 0.5 * dx;
  double par_y = par_yPos - 0.5 * dy;
  double par_z = par_zPos + 1.0 * dz;

  ii = floor(par_x / (dx + 1e-9));
  jj = floor(par_y / (dy + 1e-9));
  kk = floor(par_z / (dz + 1e-9));

  // fractional distance between nearest nodes
  iw = (par_x / dx - floor(par_x / dx));
  jw = (par_y / dy - floor(par_y / dy));
  kw = (par_z / dz - floor(par_z / dz));

  return;
}

// always call this after setting the interpolation indices with the setInterp3Dindex_u/v/wFace() function!
double Eulerian::interp3D_faceVar(const std::vector<float> &EulerData)
{

  double cube[2][2][2] = { 0.0 };

  // now set the cube values
  for (int kkk = 0; kkk <= 1; kkk++) {
    for (int jjj = 0; jjj <= 1; jjj++) {
      for (int iii = 0; iii <= 1; iii++) {
        // set the actual indices to use for the linearized Euler data
        int idx = (kk + kkk) * (ny * nx) + (jj + jjj) * (nx) + (ii + iii);
        cube[iii][jjj][kkk] = EulerData[idx];
      }
    }
  }

  // now do the interpolation, with the cube, the counters from the indices,
  // and the normalized width between the point locations and the closest cell left walls
  double u_low = (1 - iw) * (1 - jw) * cube[0][0][0] + iw * (1 - jw) * cube[1][0][0] + iw * jw * cube[1][1][0] + (1 - iw) * jw * cube[0][1][0];
  double u_high = (1 - iw) * (1 - jw) * cube[0][0][1] + iw * (1 - jw) * cube[1][0][1] + iw * jw * cube[1][1][1] + (1 - iw) * jw * cube[0][1][1];

  return (u_high - u_low) * kw + u_low;
}

// always call this after setting the interpolation indices with the setInterp3Dindex_u/v/wFace() function!
double Eulerian::interp3D_faceVar(const std::vector<double> &EulerData)
{

  double cube[2][2][2] = { 0.0 };

  // now set the cube values
  for (int kkk = 0; kkk <= 1; kkk++) {
    for (int jjj = 0; jjj <= 1; jjj++) {
      for (int iii = 0; iii <= 1; iii++) {
        // set the actual indices to use for the linearized Euler data
        int idx = (kk + kkk) * (ny * nx) + (jj + jjj) * (nx) + (ii + iii);
        cube[iii][jjj][kkk] = EulerData[idx];
      }
    }
  }

  // now do the interpolation, with the cube, the counters from the indices,
  // and the normalized width between the point locations and the closest cell left walls
  double u_low = (1 - iw) * (1 - jw) * cube[0][0][0] + iw * (1 - jw) * cube[1][0][0] + iw * jw * cube[1][1][0] + (1 - iw) * jw * cube[0][1][0];
  double u_high = (1 - iw) * (1 - jw) * cube[0][0][1] + iw * (1 - jw) * cube[1][0][1] + iw * jw * cube[1][1][1] + (1 - iw) * jw * cube[0][1][1];

  return (u_high - u_low) * kw + u_low;
}

// this gets around the problem of repeated or not repeated information, just needs called once before each interpolation,
// then intepolation on all kinds of datatypes can be done
void Eulerian::setInterp3Dindex_cellVar(const double &par_xPos, const double &par_yPos, const double &par_zPos)
{

  // the next steps are to figure out the right indices to grab the values for cube from the data,
  // where indices are forced to be special if nx, ny, or nz are zero.
  // This allows the interpolation to multiply by zero any 2nd values that are set to zero in cube.

  // so this is called once before calling the interp3D function on many different datatypes
  // sets the current indices for grabbing the cube values and for interpolating with the cube,
  // but importantly sets ip,jp, and kp to zero if the number of cells in a dimension is 1
  // as this avoids referencing outside of array problems in an efficient manner
  // it also causes some stuff to be multiplied by zero so that interpolation works on any size of data without lots of if statements


  // set a particle position corrected by the start of the domain in each direction
  // the algorythm assumes the list starts at x = 0.

  double par_x = par_xPos - 0.5 * dx;
  double par_y = par_yPos - 0.5 * dy;
  double par_z = par_zPos + 0.5 * dz;

  //double par_x = par_xPos - xStart + 0.5*dx;
  //double par_y = par_yPos - yStart + 0.5*dy;
  //double par_z = par_zPos - zStart + 0.5*dz;

  // index of nearest node in negative direction
  // by adding a really small number to dx, it stops it from putting
  // the stuff on the right wall of the cell into the next cell, and
  // puts everything from the left wall to the right wall of a cell
  // into the left cell. Makes it simpler for interpolation, as without this,
  // the interpolation would reference outside the array if the input position was exactly on
  // nx, ny, or nz.
  // basically adding a small number to dx shifts the indices so that instead of going
  // from 0 to nx - 1, they go from 0 to nx - 2. This means that ii + ip can at most be nx - 1
  // and only if a particle lands directly on the far boundary condition edge
  ii = floor(par_x / (dx + 1e-9));
  jj = floor(par_y / (dy + 1e-9));
  kk = floor(par_z / (dz + 1e-9));

  // fractional distance between nearest nodes
  iw = (par_x / dx) - floor(par_x / dx);
  jw = (par_y / dy) - floor(par_y / dy);
  kw = (par_z / dz) - floor(par_z / dz);

  /* FM -> OBSOLETE (this is insure at the boundary condition level)
    // now check to make sure that the indices are within the Eulerian grid domain
    // Notice that this no longer includes throwing an error if particles touch the far walls
    // because adding a small number to dx in the index calculation forces the index to be completely left side biased
    
    if( ii < 0 || ii+ip > nx-1 ) {
    std::cerr << "ERROR (Eulerian::setInterp3Dindexing): particle x position is out of range! x = \"" << par_xPos 
         << "\" ii+ip = \"" << ii << "\"+\"" << ip << "\",   nx-1 = \"" << nx-1 << "\"" << std::endl;
        exit(1);
    }
    if( jj < 0 || jj+jp > ny-1 ) {
        std::cerr << "ERROR (Eulerian::setInterp3Dindexing): particle y position is out of range! y = \"" << par_yPos 
            << "\" jj+jp = \"" << jj << "\"+\"" << jp << "\",   ny-1 = \"" << ny-1 << "\"" << std::endl;
        exit(1);
    }
    if( kk < 0 || kk+kp > nz-1 ) {
        std::cerr << "ERROR (Eulerian::setInterp3Dindexing): particle z position is out of range! z = \"" << par_zPos 
            << "\" kk+kp = \"" << kk << "\"+\"" << kp << "\",   nz-1 = \"" << nz-1 << "\"" << std::endl;
        exit(1);
    }
    */
}


// always call this after setting the interpolation indices with the setInterp3Dindexing() function!
double Eulerian::interp3D_cellVar(const std::vector<float> &EulerData)
{

  // first set a cube of size two to zero.
  // This is important because if nx, ny, or nz are only size 1, referencing two spots in cube won't reference outside the array.
  // the next steps are to figure out the right indices to grab the values for cube from the data,
  // where indices are forced to be special if nx, ny, or nz are zero.
  // This allows the interpolation to multiply by zero any 2nd values that are set to zero in cube.

  // now set the cube to zero, then fill it using the indices and the counters from the indices
  double cube[2][2][2] = { 0.0 };

  // now set the cube values
  for (int kkk = 0; kkk <= 1; kkk++) {
    for (int jjj = 0; jjj <= 1; jjj++) {
      for (int iii = 0; iii <= 1; iii++) {
        // set the actual indices to use for the linearized Euler data
        int idx = (kk + kkk) * (ny - 1) * (nx - 1) + (jj + jjj) * (nx - 1) + (ii + iii);
        cube[iii][jjj][kkk] = EulerData[idx];
      }
    }
  }

  // now do the interpolation, with the cube, the counters from the indices,
  // and the normalized width between the point locations and the closest cell left walls
  double u_low = (1 - iw) * (1 - jw) * cube[0][0][0] + iw * (1 - jw) * cube[1][0][0] + iw * jw * cube[1][1][0] + (1 - iw) * jw * cube[0][1][0];
  double u_high = (1 - iw) * (1 - jw) * cube[0][0][1] + iw * (1 - jw) * cube[1][0][1] + iw * jw * cube[1][1][1] + (1 - iw) * jw * cube[0][1][1];

  return (u_high - u_low) * kw + u_low;
}

// always call this after setting the interpolation indices with the setInterp3Dindexing() function!
double Eulerian::interp3D_cellVar(const std::vector<double> &EulerData)
{

  // first set a cube of size two to zero.
  // This is important because if nx, ny, or nz are only size 1, referencing two spots in cube won't reference outside the array.
  // the next steps are to figure out the right indices to grab the values for cube from the data,
  // where indices are forced to be special if nx, ny, or nz are zero.
  // This allows the interpolation to multiply by zero any 2nd values that are set to zero in cube.

  // now set the cube to zero, then fill it using the indices and the counters from the indices
  double cube[2][2][2] = { 0.0 };

  // now set the cube values
  for (int kkk = 0; kkk <= 1; kkk++) {
    for (int jjj = 0; jjj <= 1; jjj++) {
      for (int iii = 0; iii <= 1; iii++) {
        // set the actual indices to use for the linearized Euler data
        int idx = (kk + kkk) * (ny - 1) * (nx - 1) + (jj + jjj) * (nx - 1) + (ii + iii);
        cube[iii][jjj][kkk] = EulerData[idx];
      }
    }
  }

  // now do the interpolation, with the cube, the counters from the indices,
  // and the normalized width between the point locations and the closest cell left walls
  double u_low = (1 - iw) * (1 - jw) * cube[0][0][0] + iw * (1 - jw) * cube[1][0][0] + iw * jw * cube[1][1][0] + (1 - iw) * jw * cube[0][1][0];
  double u_high = (1 - iw) * (1 - jw) * cube[0][0][1] + iw * (1 - jw) * cube[1][0][1] + iw * jw * cube[1][1][1] + (1 - iw) * jw * cube[0][1][1];

  return (u_high - u_low) * kw + u_low;
}
