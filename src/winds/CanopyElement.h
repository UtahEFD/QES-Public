/****************************************************************************
 * Copyright (c) 2024 University of Utah
 * Copyright (c) 2024 University of Minnesota Duluth
 *
 * Copyright (c) 2024 Behnam Bozorgmehr
 * Copyright (c) 2024 Jeremy A. Gibbs
 * Copyright (c) 2024 Fabien Margairaz
 * Copyright (c) 2024 Eric R. Pardyjak
 * Copyright (c) 2024 Zachary Patterson
 * Copyright (c) 2024 Rob Stoll
 * Copyright (c) 2024 Lucas Ulmer
 * Copyright (c) 2024 Pete Willemsen
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

/** @file CanopyElement.h */

#pragma once

#include <cmath>
#include <map>
#include "util/ParseInterface.h"
#include "Building.h"

enum CanopyType {
  Homogeneous,
  IsolatedTree,
  ROC
};

/**
 * @class CanopyElement
 * @brief :document this:
 *
 * :long desc here:
 *
 * @sa Building
 */
class CanopyElement : public Building
{
private:
protected:
  ///@{
  /** Minimum position value for a Building */
  float x_min, y_min;
  ///@}

  ///@{
  /** Maximum position value for a Building */
  float x_max, y_max;
  ///@}

  ///@{
  /** Coordinate of center of a cell */
  float x_cent, y_cent;
  ///@}

  float polygon_area; /**< Polygon area */

  ///@{
  /** :document these: */
  int icell_cent, icell_face;
  ///@}

  ///@{
  /** :document these: */
  int nx_canopy, ny_canopy, nz_canopy;
  ///@}

  ///@{
  /** :document these: */
  int numcell_cent_2d, numcell_cent_3d;
  ///@}

public:
  CanopyElement()
  {
  }
  virtual ~CanopyElement()
  {
  }

  /*!
   * For all Canopy classes derived, this need to be defined
   */
  virtual void parseValues() = 0;

  void setPolyBuilding(WINDSGeneralData *WGD);

  virtual void setCellFlags(const WINDSInputData *WID, WINDSGeneralData *WGD, int building_number) = 0;
  virtual void canopyVegetation(WINDSGeneralData *wgd, int building_id) = 0;
  virtual void canopyWake(WINDSGeneralData *wgd, int building_id) = 0;

  virtual void canopyTurbulenceWake(WINDSGeneralData *, TURBGeneralData *, int) {}

  virtual void canopyStress(WINDSGeneralData *, TURBGeneralData *, int) {}

  virtual int getCellFlagCanopy() = 0;
  virtual int getCellFlagWake() = 0;

  std::vector<long> canopy_cell2D, canopy_cell3D; /**< map beteen WINDS grid and canopy grid */
  std::map<int, int> canopy_cellMap2D, canopy_cellMap3D; /**< map beteen WINDS grid and canopy grid */

  CanopyType _cType;

protected:
  /*!
   * For there and below, the canopyInitial function has to be defined
   */
  virtual void setCanopyGrid(WINDSGeneralData *wgd, int building_id);

  /*!
   * This function takes in icellflag defined in the defineCanopy function along with variables initialized in
   * the readCanopy function and initial velocity field components (u0 and v0). This function applies the urban
   * canopy parameterization and returns modified initial velocity field components.
   */
  void canopyCioncoParam(WINDSGeneralData *wgd);

  /*!
   * This is a new function wrote by Lucas Ulmerlmer and is being called from the plantInitial function. The purpose
   * of this function is to use bisection method to find root of the specified equation. It calculates the
   * displacement height when the bisection function is not finding it.
   */
  float canopySlopeMatch(float z0, float canopy_top, float canopy_atten);

  /*!
   *
   */
  float canopyBisection(float ustar, float z0, float canopy_top, float canopy_atten, float vk, float psi_m);

private:
  /*!
   *
   */
  std::vector<float> canopy_atten; /**< Canopy attenuation coefficient */

  std::vector<float> canopy_bot; /**< Canopy bottom */
  std::vector<int> canopy_bot_index; /**< Canopy bottom index */
  std::vector<float> canopy_top; /**< Canopy top */
  std::vector<int> canopy_top_index; /**< Canopy top index */

  std::vector<float> canopy_base; /**< Canopy base */
  std::vector<float> canopy_height; /**< Canopy height */

  std::vector<float> canopy_z0; /**< Canopy surface roughness */
  std::vector<float> canopy_ustar; /**< Velocity gradient at the top of canopy */
  std::vector<float> canopy_d; /**< Canopy displacement length */
};

inline int CanopyElement::getCellFlagCanopy()
{
  return 18;
}

inline int CanopyElement::getCellFlagWake()
{
  return 19;
}
