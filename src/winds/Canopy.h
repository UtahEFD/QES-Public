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

/** @file Canopy.h */

#pragma once

#include <cmath>
#include <vector>
#include <map>

#include "util/PolygonVertex.h"

#include "CanopyElement.h"


// forward declaration of WINDSInputData and WINDSGeneralData, which
// will be used by the derived classes and thus included there in the
// C++ files
class WINDSInputData;
class WINDSGeneralData;
class TURBGeneralData;

/**
 * @class Canopy
 * @brief :document this:
 *
 * :long desc here:
 *
 */
class Canopy
{
private:
public:
  Canopy()
  {}

  Canopy(const WINDSInputData *WID, WINDSGeneralData *WGD);

  virtual ~Canopy()
  {}

  void setCanopyElements(const WINDSInputData *WID, WINDSGeneralData *WGD);

  void applyCanopyVegetation(WINDSGeneralData *WGD);
  void applyCanopyWake(WINDSGeneralData *WGD);

  void applyCanopyTurbulenceWake(WINDSGeneralData *WGD, TURBGeneralData *);


  /*
   * For all Canopy classes derived, this need to be defined
  virtual void parseValues()
  {
      parsePrimitive<int>(true, num_canopies, "num_canopies");
      // read the input data for canopies
      //parseMultiPolymorphs(false, canopies, Polymorph<Building, CanopyHomogeneous>("Homogeneous"));
      //parseMultiPolymorphs(false, canopies, Polymorph<Building, CanopyIsolatedTree>("IsolatedTree"));
      //parseMultiPolymorphs(false, canopies, Polymorph<Building, CanopyWindbreak>("Windbreak"));
      // add other type of canopy here
  }
  */

  virtual int getCellFlagCanopy();
  virtual int getCellFlagWake();

  int wakeFlag;

  /*!
   *
   */
  std::vector<float> canopy_atten_coeff; /**< Canopy attenuation coefficient */

  std::vector<float> canopy_bot; /**< Canopy bottom */
  std::vector<int> canopy_bot_index; /**< Canopy bottom index */
  std::vector<float> canopy_top; /**< Canopy top */
  std::vector<int> canopy_top_index; /**< Canopy top index */

  std::vector<float> canopy_base; /**< Canopy base */
  std::vector<float> canopy_height; /**< Canopy height */

  std::vector<float> canopy_z0; /**< Canopy surface roughness */
  std::vector<float> canopy_ustar; /**< Velocity gradient at the top of canopy */
  std::vector<float> canopy_d; /**< Canopy displacement length */

  std::vector<Building *> allCanopiesV; /**< :document this: */
  std::vector<float> base_height; /**< Base height of trees */
  std::vector<float> effective_height; /**< Effective height of trees */
  std::vector<int> icanopy_flag; /**< :document this: */
  std::vector<int> canopy_id; /**< :document this: */

  std::vector<float> wake_u_defect; /**< :document this: */
  std::vector<float> wake_v_defect; /**< :document this: */

protected:
  int nx_canopy, ny_canopy, nz_canopy;
  int numcell_cent_2d, numcell_cent_3d;

  /*!
   * This function takes in icellflag defined in the defineCanopy function along with variables initialized in
   * the readCanopy function and initial velocity field components (u0 and v0). This function applies the urban
   * canopy parameterization and returns modified initial velocity field components.
   */
  void canopyCioncoParam(WINDSGeneralData *wgd);

  /**
   * Uses linear regression method to define ustar and surface roughness of the canopy.
   *
   * @note Called from canopyCioncoParam
   *
   * @param WGD :document this:
   */
  void canopyRegression(WINDSGeneralData *wgd);


  /**
   * Uses bisection to find root of the specified equation.
   *
   * It calculates the displacement height when the bisection function is not finding it.
   *
   * @note Called from canopyCioncoParam.
   *
   * @param :document this:
   */
  float canopySlopeMatch(float z0, float canopy_top, float canopy_atten);

  /*!
   *
   */
  float canopyBisection(float ustar, float z0, float canopy_top, float canopy_atten, float vk, float psi_m);

private:
  void mergeSort(std::vector<float> &effective_height, std::vector<Building *> allBuildingsV, std::vector<int> &tree_id);
};

inline int Canopy::getCellFlagCanopy()
{
  return 18;
}

inline int Canopy::getCellFlagWake()
{
  return 19;
}
