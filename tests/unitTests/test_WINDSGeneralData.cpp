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

  defineVerticalStretching(dz);
  defineVerticalGrid();
  defineHorizontalGrid();

  timestamp.emplace_back("2020-01-01T00:00:00");

  allocateMemory();
}

test_WINDSGeneralData::test_WINDSGeneralData(const int gridSize[3], const float gridRes[3], const float *dz_values_in)
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


  std::vector<float> dz_values;
  dz_values.resize(nz - 2, 0.0f);
  for (int k = 0; k < nz - 2; ++k) {
    dz_values[k] = dz_values_in[k];
  }

  defineVerticalStretching(dz_values);
  defineVerticalGrid();
  defineHorizontalGrid();

  timestamp.emplace_back("2020-01-01T00:00:00");

  allocateMemory();
}
