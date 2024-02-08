/****************************************************************************
 * Copyright (c) 2022 University of Utah
 * Copyright (c) 2022 University of Minnesota Duluth
 *
 * Copyright (c) 2022 Behnam Bozorgmehr
 * Copyright (c) 2022 Jeremy A. Gibbs
 * Copyright (c) 2022 Fabien Margairaz
 * Copyright (c) 2022 Eric R. Pardyjak
 * Copyright (c) 2022 Zachary Patterson
 * Copyright (c) 2022 Rob Stoll
 * Copyright (c) 2022 Lucas Ulmer
 * Copyright (c) 2022 Pete Willemsen
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

/** @file WINDSOutputWorkspace.h */

#pragma once

#include <string>

#include "WINDSGeneralData.h"
#include "util/QESNetCDFOutput.h"
#include "util/QEStime.h"

/**
 * @class WINDSOutputVisualization
 * @brief Specialized output classes derived from QESNetCDFOutput for
 * face center data (used for turbulence,...)
 */
class WINDSOutputWorkspace : public QESNetCDFOutput
{
private:
  WINDSOutputWorkspace() {}

public:
  WINDSOutputWorkspace(WINDSGeneralData *, const std::string &);
  ~WINDSOutputWorkspace()
  {}

  /** save function be call outside */
  void save(QEStime);

protected:
  /**
   * :document this:
   */
  void setAllOutputFields();

private:
  std::vector<float> m_x, m_y, m_z;
  std::vector<float> m_x_face, m_y_face, m_z_face;
  std::vector<float> m_dz_array;

  WINDSGeneralData *m_WGD;

  ///@{
  /**
   * Building data functions.
   * @warning [FM] Feb.28.2020 OBSOLETE
   */
  void setBuildingFields(NcDim *, NcDim *);
  void getBuildingFields();
  ///@

  /**
   * Building data variables
   * @warning [FM] Feb.28.2020 OBSOLETE
   */
  bool buildingFieldsSet = false;

  // [FM] Feb.28.2020 OBSOLETE
  // These variables are used to convert data structure in array so it can be stored in
  // NetCDF file. (Canopy can be building, need to specify)
  // size of these vector = number of buidlings
  std::vector<float> building_rotation, canopy_rotation;

  std::vector<float> L, W, H;
  std::vector<float> length_eff, width_eff, height_eff, base_height;
  std::vector<float> building_cent_x, building_cent_y;

  std::vector<int> i_start, i_end, j_start, j_end, k_end, k_start;
  std::vector<int> i_cut_start, i_cut_end, j_cut_start, j_cut_end, k_cut_end;
  std::vector<int> i_building_cent, j_building_cent;

  std::vector<float> upwind_dir, Lr;
};
