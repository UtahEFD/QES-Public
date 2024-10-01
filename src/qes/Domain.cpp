#include "Domain.h"

using namespace qes;

Domain::Domain(int nx, int ny, int nz, float dx, float dy, float dz)
{
  // Internally this information is converted to the staggered grid
  // representation, which adds 1 to nx and ny and 2 to nz (to
  // account for staggered grid and ghost cell).
  domainData.nx = nx + 1;
  domainData.ny = ny + 1;
  domainData.nz = nz + 2;

  domainData.dx = dx;
  domainData.dy = dy;
  domainData.dz = dz;
}


void Domain::defineVerticalStretching(const float &dz_value)
{
  // vertical grid (can be uniform or stretched)
  dz_array.resize(domainData.nz - 1, 0.0);
  // Uniform vertical grid
  for (float &k : dz_array) {
    k = dz_value;
  }
}

void Domain::defineVerticalStretching(const std::vector<float> &dz_value)
{
  // vertical grid (can be uniform or stretched)
  dz_array.resize(domainData.nz - 1, 0.0);
  // Stretched vertical grid
  for (size_t k = 1; k < dz_array.size(); ++k) {
    dz_array[k] = dz_value[k - 1];// Read in custom dz values and set them to dz_array
  }
  dz_array[0] = dz_array[1];// Value for ghost cell below the surface

  float newDz = *std::min_element(dz_array.begin(), dz_array.end());// Set dz to minimum value of
  domainData.dz = newDz;
}

void Domain::defineVerticalGrid()
{
  // Location of face in z-dir
  z_face.resize(domainData.nz, 0.0);
  z_face[0] = -dz_array[0];
  z_face[1] = 0.0;
  for (size_t k = 2; k < z_face.size(); ++k) {
    z_face[k] = z_face[k - 1] + dz_array[k - 1];
  }

  // Location of cell centers in z-dir
  z.resize(domainData.nz - 1, 0.0);
  z[0] = -0.5f * dz_array[0];
  for (size_t k = 1; k < z.size(); ++k) {
    z[k] = 0.5f * (z_face[k] + z_face[k + 1]);
  }
}

void Domain::defineHorizontalGrid()
{
  // horizontal grid (x-direction)
  x.resize(domainData.nx - 1);
  for (auto i = 0; i < domainData.nx - 1; ++i) {
    x[i] = ((float)i + 0.5f) * domainData.dx;// Location of face centers in x-dir
  }

  // horizontal grid (y-direction)
  y.resize(domainData.ny - 1);
  for (auto j = 0; j < domainData.ny - 1; ++j) {
    y[j] = ((float)j + 0.5f) * domainData.dy;// Location of face centers in y-dir
  }
}
