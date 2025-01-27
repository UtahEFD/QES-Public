/****************************************************************************
 * Copyright (c) 2024 University of Utah
 * Copyright (c) 2024 University of Minnesota Duluth
 *
 * Copyright (c) 2024 Matthew Moody
 * Copyright (c) 2024 Jeremy Gibbs
 * Copyright (c) 2024 Rob Stoll
 * Copyright (c) 2024 Fabien Margairaz
 * Copyright (c) 2024 Brian Bailey
 * Copyright (c) 2024 Pete Willemsen
 *
 * This file is part of QES-Fire
 *
 * GPL-3.0 License
 *
 * QES-Fire is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, version 3 of the License.
 *
 * QES-Fire is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with QES-Winds. If not, see <https://www.gnu.org/licenses/>.
 ****************************************************************************/
/**
 * @file FuelRead.cpp
 * @brief This function reads fuel data from provided GEOTIF.
 */

#include "util/Geotiff.h"
#include "FuelRead.h"
#include "winds/WINDSInputData.h"
#include "winds/WINDSGeneralData.h"


FuelRead::FuelRead(const std::string &filename,
                   std::tuple<int, int> dim,
                   std::tuple<float, float> cellSize)
{
  int nx = get<0>(dim);
  int ny = get<1>(dim);
  float dx = get<0>(cellSize);
  float dy = get<1>(cellSize);

  std::cout << "Extracting fuel data from " << filename << std::endl;
  const char *tiffFile = filename.c_str();
  // create object of Geotiff class
  Geotiff tiff((const char *)tiffFile);

  // output a value from 2D array
  float **rasterBandData = tiff.GetRasterBand(1);
  // cout << "value at row 10, column 10: " << rasterBandData[10][10] << endl;


  // dump out the Geotransform (6 element array of doubles)
  double *gt = tiff.GetGeoTransform();
  float X_coord = gt[0];
  float X_pixel = gt[1];
  float ROW_rotation = gt[2];
  float Y_coord = gt[3];
  float Col_rotation = gt[4];
  float Y_pixel = gt[5];

  // dump out array (band) dimensions of Geotiff data
  //
  int *dims = tiff.GetDimensions();
  int NCOLS = dims[0];
  int NROWS = dims[1];
  int NBANDS = dims[2];
  std::cout << tiff.GetProjection() << std::endl;
  std::cout << "Fuel Domain Origin = (" << X_coord << "," << Y_coord << ")" << std::endl;
  std::cout << "Fuel Pixel Size = (" << X_pixel << "," << Y_pixel << ")" << std::endl;
  std::cout << "Fuel file size is " << NCOLS << "x" << NROWS << std::endl;

  int idx;
  int rasterData;
  fuelField.resize((nx) * (ny));
  float xStride = (dx / X_pixel);
  float yStride = -(dy / Y_pixel);
  std::cout << "xStride = " << xStride << ", yStride = " << yStride << std::endl;
  std::vector<int> rasterCount;
  rasterCount.resize(ceil(xStride) * ceil(yStride));
  int n = rasterCount.size();
  int countIDX = 0;
  int xpix;
  int ypix;

  for (int j = 0; j < ny; j++) {
    for (int i = 0; i < nx; i++) {

      idx = i + (ny - 1 - j) * (nx);
      rasterData = 0;

      if (xStride < 1) {
        std::fill(rasterCount.begin(), rasterCount.end(), 0);
        for (int jj = 0; jj < yStride; jj++) {
          for (int ii = 0; ii < xStride; ii++) {
            countIDX = ii + jj * floor((xStride));
            // get pixel location
            xpix = floor(i * xStride) + ii;
            ypix = floor(j * yStride) + jj;

            rasterCount[countIDX] = rasterBandData[ypix][xpix];
          }
        }

        // variable to store max of
        // input array which will
        // to have size of count array
        int max = 205;

        // auxiliary(count) array to
        // store count. Initialize
        // count array as 0. Size
        // of count array will be
        // equal to (max + 1).
        int t = max + 1;
        int count[t];
        for (int m = 0; m < t; m++)
          count[m] = 0;

        // Store count of each element
        // of input array
        for (int m = 0; m < n; m++)
          count[rasterCount[m]]++;

        // mode is the index with maximum count
        int mode = 0;
        int k = count[0];
        for (int p = 1; p < t; p++) {
          if (count[p] > k) {
            k = count[p];
            mode = p;
          }
        }
        rasterData = mode;
      } else {
        rasterData = rasterBandData[j][i];
      }
      fuelField[idx] = rasterData;
    }
  }
}
