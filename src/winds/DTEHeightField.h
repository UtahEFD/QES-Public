/****************************************************************************
 * Copyright (c) 2025 University of Utah
 * Copyright (c) 2025 University of Minnesota Duluth
 *
 * Copyright (c) 2025 Behnam Bozorgmehr
 * Copyright (c) 2025 Jeremy A. Gibbs
 * Copyright (c) 2025 Fabien Margairaz
 * Copyright (c) 2025 Eric R. Pardyjak
 * Copyright (c) 2025 Zachary Patterson
 * Copyright (c) 2025 Rob Stoll
 * Copyright (c) 2025 Lucas Ulmer
 * Copyright (c) 2025 Pete Willemsen
 *
 * This file is part of QES-Winds
 *
 * GPL-3.0 License
 *
 * QES-Winds is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, version 3 of the License.
 *
 * QES-Winds is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with QES-Winds. If not, see <https://www.gnu.org/licenses/>.
 ****************************************************************************/

/** @file DTEHeightField.h */

#ifndef __DTE_HEIGHT_FIELD_H__
#define __DTE_HEIGHT_FIELD_H__ 1

#include <string>
#include "util/Triangle.h"
#include "util/Vector3Float.h"
#include "util/Vector3Int.h"


#include "gdal_priv.h"
#include "cpl_conv.h"// for CPLMalloc()
#include "ogrsf_frmts.h"

#include "Cell.h"
#include "Edge.h"
#include <iostream>
#include <iomanip>
#include <fstream>
#include <cmath>
#include <chrono>

class WINDSGeneralData;
class WINDSInputData;

/**
 * @class DTEHeightField
 * @brief Loads digital elevation data.
 *
 * @sa Cell
 * @sa Edge
 * @sa Triangle
 * @sa Vector3Float
 */
class DTEHeightField
{
public:
  friend class test_DTEHeightField;

  DTEHeightField();// this is not likely to produce anything
                   // useful -- Pete

  /**
   * Constructs a GIS Digital Elevation Model for use with QES.
   *
   * @param filename the filename containing the GIS data to load
   * @param dim a 3-tuple of ints representing the dimension of
   * the domain, as in {nx, ny, nz}
   * @param cellSize a 3-tuple of floats representing the size of
   * each domain cell in the 3 dimensions, as in {dx, dy, dz}
   * @param UTMx the UTM origin in x
   * @param UTMy the UTM origin in y
   * @param OriginFlag :document this:
   * @param DEMDistanceX :document this:
   * @param DEMDistanceY :document this:
   * @return a string representing the results of the failed summation.
   */
  DTEHeightField(const std::string &filename,
                 std::tuple<int, int, int> dim,
                 std::tuple<float, float, float> cellSize,
                 float UTMx,
                 float UTMy,
                 int OriginFlag,
                 float DEMDistanceX,
                 float DEMDistanceY);

  /**
   * Loads a GIS Digital Elevation Model
   *
   * @param heightField :document this:
   * @param dim :document this:
   * @param cellSize :document this:
   * @param halo_x :document this:
   * @param halo_y :document this:
   * @return a string representing the results of the failed summation.
   */
  DTEHeightField(const std::vector<double> &heightField,
                 std::tuple<int, int, int> dim,
                 std::tuple<float, float, float> cellSize,
                 float halo_x,
                 float halo_y);

  ~DTEHeightField();

  const std::vector<Triangle *> &getTris() const { return m_triList; }

  /**
   * Takes in a domain to change and a grid size for the size of cells in the domain.
   * Iterates over all Triangle objects
   * and shifts all points so that the minimum value on each axis
   * is 0. This will then use the greatest point divided by the size
   * of one cell in the grid to set the value of the given domain.
   *
   * @param domain Domain that will be changed to match the dem file
   * @param grid Size of each cell in the domain space.
   */
  void setDomain(Vector3Int &domain, Vector3Float &grid);


  /**
   * Takes the Triangle list that represents the dem file and
   * outputs the mesh in an obj file format to the file "s".
   *
   * @param s File that the obj data will be written to.
   */
  void outputOBJ(std::string s);

  /**
   * @details Takes a list of cells, and the domain space and queries
   * the height field at corners of the each cell setting coordinates and the
   * substances present in each cell. This then returns a list of ints that are
   * the id's of all cut-cells(cells that are both terrain and air).
   *
   * @param cells List of cells to be initialized
   * @param nx X dimension in the domain
   * @param ny Y dimension in the domain
   * @param nz Z dimension in the domain
   * @param dx Size of a cell in the X axis
   * @param dy Size of a cell in the Y axis
   * @param dz_array :document this:
   * @param z_face :document this:
   * @param halo_x :document this:
   * @param halo_y :document this:
   * @return List of ID values for all cut cells.
   */
  // void setCells(Cell *cells, WINDSGeneralData *WGD, const WINDSInputData *WID);
  void setCells(WINDSGeneralData *WGD, const WINDSInputData *WID);

  /**
   * Frees the pafScanline.
   *
   * @note Should be called after all DEM querying has taken place.
   */
  void closeScanner();

  /**
   * :document this:
   *
   * @param rasterX :document this:
   * @param rasterY :document this:
   * @param geoX :document this:
   * @param geoY :document this:
   */
  void convertRasterToGeo(double rasterX, double rasterY, double &geoX, double &geoY)
  {
    // Affine transformation from the GDAL geotransform:
    // https://gdal.org/user/raster_data_model.html
    geoX = m_geoTransform[0] + rasterX * m_geoTransform[1] + rasterY * m_geoTransform[2];
    geoY = m_geoTransform[3] + rasterX * m_geoTransform[4] + rasterY * m_geoTransform[5];
  }


  ///@{
  /** :document this: */
  int m_nXSize, m_nYSize;
  ///@}

  ///@{
  /** :document this: */
  float pixelSizeX, pixelSizeY;
  ///@}

  ///@{
  /** :document this: */
  float DEMDistancex, DEMDistancey;
  ///@}

  int originFlag; /**< :document this: */

  double adfMinMax[2]; /**< :document this: */

private:
  std::vector<Vector3Float> terrainPoints; /**< List of terrain points */
  std::vector<Edge<int>> terrainEdges; /**< List of edges that connect the terrain points */
  std::vector<Vector3Float> fluidFacePoints[6]; /**< :document this: */
  Vector3Float location; /**< XYZ location of the cell */
  Vector3Float dimensions; /**< Size of the cell in xyz directions */
  int count = 0;

  /**
   * Given the height of the DEM file at each of it's corners and uses
   * them to calculate at what points cells are intersected by the quad the corners form.
   * The cells are then updated to reflect the cut.
   *
   * @param cells List of cells to be initialized
   * @param i Current x dimension index of the cell
   * @param i Current y dimension index of the cell
   * @param nx X dimension in the domain
   * @param ny Y dimension in the domain
   * @param nz Z dimension in the domain
   * @param dz_array :document this:
   * @param z_face :document this:
   * @param corners Array containing the points that representing the DEM elevation at each of the cells corners
   * @param cutCells List of all cells which the terrain goes through
   */
  // void setCellPoints(Cell *cells, int i, int j, int nx, int ny, int nz, std::vector<float> &dz_array, std::vector<float> z_face, Vector3Float corners[], std::vector<int> &cutCells, WINDSGeneralData *WGD);
  void setCellPoints(const int &i_in,
                     const int &j_in,
                     const int &nx,
                     const int &ny,
                     const int &nz,
                     const float &dx,
                     const float &dy,
                     const std::vector<float> &dz_array,
                     const std::vector<float> &x,
                     const std::vector<float> &y,
                     const std::vector<float> &z,
                     const std::vector<float> &z_face,
                     Vector3Float corners[],
                     std::vector<int> &cutCells,
                     WINDSGeneralData *WGD);

  /**
   * :document this:
   */
  void load();

  /**
   * :document this:
   *
   * @param percentage :document this:
   */
  void printProgress(float percentage);

  // void loadImage();

  /**
   * :document this:
   *
   * @param f1 :document this:
   * @param f2 :document this:
   */
  bool compareEquality(double f1, double f2) const
  {
    const double eps = 1.0e-6;
    return fabs(f1 - f2) < eps;
  }

  /**
   * :document this:
   *
   * @param scanline :document this:
   * @param j :document this:
   * @param k :document this:
   */
  float queryHeight(float *scanline, int j, int k) const
  {
    float height;
    if (j >= m_nXSize || k >= m_nYSize) {
      height = 0.0;

    }
    // if (j * m_nXSize + k >= m_nXSize * m_nYSize
    else {
      // important to remember range is [0, n-1], so need the -1
      // in the flip
      // Previous code had this -- does not seem correct
      // height = scanline[ abs(k-m_nYSize) * m_nXSize + j ];

      height = scanline[(m_nYSize - k - 1) * (m_nXSize) + j] - adfMinMax[0];
    }

    if (height < 0.0 || std::isnan(abs(height))) {
      height = 0.0;
    }

    if (!compareEquality(height, m_rbNoData)) {
      height = height * m_rbScale + m_rbOffset;
    } else {
      height = m_rbMin;
    }


    return height;
  }

  /**
   * Given a height between the two points (z value) this function will create
   * a third point which exists on the line from a to b existing at height h.
   * in the result that a and b exist on the same height, the mid point between
   * the two will be returned instead.
   *
   * @param a First point designating the line
   * @param b Second point designating the line
   * @param height The height at which the third point will be created
   * @return An intermediate point existing on the line from a to b at z value height
   */
  Vector3Float getIntermediate(Vector3Float a, Vector3Float b, float height) const;


  std::string m_filename; /**< :document this: */
  GDALDataset *m_poDataset; /**< :document this: */
  double m_geoTransform[6]; /**< :document this: */

  // int m_nXSize, m_nYSize;
  ///@{
  /** :document this: */
  double m_rbScale, m_rbOffset, m_rbNoData, m_rbMin;
  ///@}

  // Texture relative information
  GDALDataset *m_imageDataset; /**< :document this: */
  ///@{
  /** :document this: */
  int m_imageXSize, m_imageYSize;
  ///@}
  double m_imageGeoTransform[6]; /**< :document this: */

  // float pixelSizeX, pixelSizeY;
  std::tuple<float, float, float> m_cellSize; /**< :document this: */
  std::tuple<int, int, int> m_dim; /**< :document this: */

  ///@{
  /** :document this: */
  float domain_UTMx, domain_UTMy;
  ///@}
  ///@{
  /** :document this */
  float origin_x, origin_y;
  ///@}
  int shift_x = 0; /**< :document this: */
  int shift_y = 0; /**< :document this: */
  // int domain_nx, domain_ny;
  int end_x = 0; /**< :document this: */
  int end_y = 0; /**< :document this: */

  std::vector<Triangle *> m_triList; /**< :document this: */
  ///@{
  /** :document this: */
  float min[3], max[3];
  ///@}

  float *pafScanline; /**< :document this: */
};


#endif
