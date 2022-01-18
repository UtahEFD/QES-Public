#include "test_WINDSGeneralData.h"


test_WINDSGeneralData::test_WINDSGeneralData(const int gridSize[3], const float gridRes[3])
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

  // vertical grid
  dz_array.resize(nz - 1, 0.0);
  z.resize(nz - 1);
  z_face.resize(nz - 1);

  for (size_t k = 0; k < z.size(); k++) {
    dz_array[k] = dz;
  }

  z_face[0] = 0.0;
  z[0] = -0.5 * dz_array[0];
  for (size_t k = 1; k < z.size(); k++) {
    z_face[k] = z_face[k - 1] + dz_array[k];// Location of face centers in z-dir
    z[k] = 0.5 * (z_face[k - 1] + z_face[k]);// Location of cell centers in z-dir
  }

  // horizontal grid (x-direction)
  x.resize(nx - 1);
  for (auto i = 0; i < nx - 1; i++) {
    x[i] = (i + 0.5) * dx;// Location of face centers in x-dir
  }

  // horizontal grid (y-direction)
  y.resize(ny - 1);
  for (auto j = 0; j < ny - 1; j++) {
    y[j] = (j + 0.5) * dy;// Location of face centers in y-dir
  }

  numcell_cout = (nx - 1) * (ny - 1) * (nz - 2);// Total number of cell-centered values in domain
  numcell_cout_2d = (nx - 1) * (ny - 1);// Total number of horizontal cell-centered values in domain
  numcell_cent = (nx - 1) * (ny - 1) * (nz - 1);// Total number of cell-centered values in domain
  numcell_face = nx * ny * nz;// Total number of face-centered values in domain

  // Resize the coefficients for use with the solver
  e.resize(numcell_cent, 1.0);
  f.resize(numcell_cent, 1.0);
  g.resize(numcell_cent, 1.0);
  h.resize(numcell_cent, 1.0);
  m.resize(numcell_cent, 1.0);
  n.resize(numcell_cent, 1.0);

  building_volume_frac.resize(numcell_cent, 1.0);
  terrain_volume_frac.resize(numcell_cent, 1.0);
  ni.resize(numcell_cent, 0.0);
  nj.resize(numcell_cent, 0.0);
  nk.resize(numcell_cent, 0.0);

  icellflag.resize(numcell_cent, 1);
  icellflag_initial.resize(numcell_cent, 1);
  icellflag_footprint.resize(numcell_cout_2d, 1);

  ibuilding_flag.resize(numcell_cent, -1);

  mixingLengths.resize(numcell_cent, 0.0);

  terrain.resize(numcell_cout_2d, 0.0);
  terrain_face_id.resize(nx * ny, 1);
  terrain_id.resize((nx - 1) * (ny - 1), 1);

  /////////////////////////////////////////

  // Set the Wind Velocity data elements to be of the correct size
  // Initialize u0,v0,w0,u,v and w to 0.0
  u0.resize(numcell_face, 0.0);
  v0.resize(numcell_face, 0.0);
  w0.resize(numcell_face, 0.0);

  u.resize(numcell_face, 0.0);
  v.resize(numcell_face, 0.0);
  w.resize(numcell_face, 0.0);

  std::cout << "Memory allocation complete." << std::endl;
}
