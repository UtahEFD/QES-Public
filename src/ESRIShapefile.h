/****************************************************************************
 * Copyright (c) 2021 University of Utah
 * Copyright (c) 2021 University of Minnesota Duluth
 *
 * Copyright (c) 2021 Behnam Bozorgmehr
 * Copyright (c) 2021 Jeremy A. Gibbs
 * Copyright (c) 2021 Fabien Margairaz
 * Copyright (c) 2021 Eric R. Pardyjak
 * Copyright (c) 2021 Zachary Patterson
 * Copyright (c) 2021 Rob Stoll
 * Copyright (c) 2021 Pete Willemsen
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

/** @file ESRIShapefile.h */

#pragma once

#include <cassert>
#include <vector>
#include <map>
// #include "gdal.h"
#include "gdal_priv.h"
#include "cpl_conv.h"// for CPLMalloc()
#include "ogrsf_frmts.h"
#include "ogr_spatialref.h"
#include <limits>

#include "PolygonVertex.h"

/**
 * @class ESRIShapefile
 * @brief :document this:
 *
 * :long desc here:
 *
 * @sa PolygonVertex
 */
class ESRIShapefile
{
public:
  ESRIShapefile();
  ESRIShapefile(const std::string &filename,
                const std::string &layerName,
                std::vector<std::vector<polyVert>> &polygons,
                std::vector<float> &building_height,
                float heightFactor);
  ESRIShapefile(const std::string &filename,
                const std::string &layerName,
                std::vector<std::vector<polyVert>> &polygons,
                std::map<std::string, std::vector<float>> &features);
  ESRIShapefile(const std::string &filename,
                const std::string &layerName);
  ~ESRIShapefile();

  /**
   * :document this:
   *
   * @param dim :document this:
   */
  void getLocalDomain(std::vector<float> &dim)
  {
    assert(dim.size() == 2);
    dim[0] = (int)ceil(maxBound[0] - minBound[0]);
    dim[1] = (int)ceil(maxBound[1] - minBound[1]);
  }

  /**
   * :document this:
   *
   * @param ext :document this:
   */
  void getMinExtent(std::vector<float> &ext)
  {
    assert(ext.size() == 2);
    ext[0] = minBound[0];
    ext[1] = minBound[1];
  }

  void getMaxExtent(std::vector<float> &ext)
  {
    assert(ext.size() == 2);
    ext[0] = maxBound[0];
    ext[1] = maxBound[1];
  }

  std::vector<std::vector<polyVert>> m_polygons;
  std::map<std::string, std::vector<float>> m_features;

private:
  /**
   * :document this:
   *
   * @param polygons :document this:
   * @param building_height :document this:
   * @param heightFactor :document this:
   */
  void loadVectorData(std::vector<std::vector<polyVert>> &polygons, std::vector<float> &building_height, float heightFactor);

  /**
   * :document this:
   *
   * @param polygons :document this:
   * @param features :document this:
   */
  void loadVectorData(std::vector<std::vector<polyVert>> &polygons, std::map<std::string, std::vector<float>> &feature);

  std::string m_filename; /**< :document this */
  std::string m_layerName; /**< :document this */

  GDALDataset *m_poDS; /**< :document this */

  OGRSpatialReference *m_SpRef;

  ///@{
  /** :document this */
  std::vector<float> minBound, maxBound;
  ///@}
};
