#include <algorithm>

#include "Domain.h"

using namespace qes;

Domain::Domain(const int &nx, const int &ny, const int &nz, const float &dx, const float &dy, const float &dz)
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
  domainData.dxy = std::min(domainData.dx, domainData.dy);

  // Definition of the grid
  // ??? how to define the vertical grid with the different option
  defineVerticalStretching(dz);
  /*if (WID->simParams->verticalStretching == 0) {// Uniform vertical grid
    defineVerticalStretching(dz);
  } else if (WID->simParams->verticalStretching == 1) {// Stretched vertical grid
    defineVerticalStretching(WID->simParams->dz_value);// Read in custom dz values and set them to dz_array
  }*/
  defineVerticalGrid();
  defineHorizontalGrid();
}

Domain::Domain(const std::string &inputFile)
{
  auto *input = new NetCDFInput(inputFile);

  input->getDimensionSize("x_face", domainData.nx);
  input->getDimensionSize("y_face", domainData.ny);
  // nz - face centered value + bottom ghost (consistant with QES-Winds)
  input->getDimensionSize("z_face", domainData.nz);

  x.resize(domainData.nx - 1);
  y.resize(domainData.ny - 1);
  z.resize(domainData.nz - 1);
  z_face.resize(domainData.nz);
  dz_array.resize(domainData.nz - 1, 0.0);

  input->getVariableData("x", x);
  domainData.dx = x[1] - x[0]; /**< Grid resolution in x-direction */

  input->getVariableData("y", y);
  domainData.dy = y[1] - y[0]; /**< Grid resolution in x-direction */
  // dxy = MIN_S(dx, dy);

  input->getVariableData("z", z);
  // check if dz_array is in the NetCDF file
  NcVar NcVar_dz;
  input->getVariable("dz_array", NcVar_dz);
  if (!NcVar_dz.isNull()) {
    input->getVariableData("dz_array", dz_array);
    domainData.dz = *std::min_element(dz_array.begin(), dz_array.end());
  } else {
    domainData.dz = z[1] - z[0];
    for (size_t k = 0; k < z.size(); k++) {
      dz_array[k] = domainData.dz;
    }
  }

  // check if z_face is in the NetCDF file
  NcVar NcVar_zface;
  input->getVariable("z_face", NcVar_zface);
  if (!NcVar_zface.isNull()) {
    input->getVariableData("z_face", z_face);
  } else {
    z_face[0] = -dz_array[0];
    z_face[1] = 0.0;
    for (size_t k = 2; k < z_face.size() - 1; ++k) {
      z_face[k] = z_face[k - 1] + dz_array[k - 1];
    }
  }
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
  for (auto k = 1; k < dz_array.size(); ++k) {
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
  for (auto k = 2; k < z_face.size(); ++k) {
    z_face[k] = z_face[k - 1] + dz_array[k - 1];
  }

  // Location of cell centers in z-dir
  z.resize(domainData.nz - 1, 0.0);
  z[0] = -0.5f * dz_array[0];
  for (auto k = 1; k < z.size(); ++k) {
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
