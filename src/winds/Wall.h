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

/** @file Wall.h */

#pragma once

#include <iostream>
#include <fstream>
#include <cstdlib>
#include <math.h>
#include <vector>
#include <chrono>
#include <limits>

// may need to forward reference this???
class WINDSGeneralData;

using namespace std;

/**
 * @class Wall
 * @brief :document this:
 *
 * :long desc here, if necessary:
 */
class Wall
{

protected:
public:
  Wall()
  {
  }
  ~Wall()
  {
  }

  /**
   * This function takes in the icellflags set by setCellsFlag
   * function for stair-step method and sets related coefficients to
   * zero to define solid walls. It also creates vectors of indices
   * of the cells that have wall to right/left, wall above/bellow
   * and wall in front/back
   *
   * @param WGD :document this:
   */
  void defineWalls(WINDSGeneralData *WGD);


  /**
   * This function takes in vectores of indices of the cells that have wall to right/left,
   * wall above/bellow and wall in front/back and applies the log law boundary condition fix
   * to the cells near Walls (based on Kevin Briggs master's thesis (2015))
   *
   * @param WGD :document this:
   */
  void wallLogBC(WINDSGeneralData *WGD, bool isInitial);


  /**
   * This function takes in icellflag vector and sets velocity components inside the building
   * and terrain to zero.
   *
   * @param WGD :document this:
   */
  void setVelocityZero(WINDSGeneralData *WGD);


  /**
   * This function takes in solver coefficient vectors (n, m, e, f, g and h)
   * and modify them for solver
   *
   * @param WGD :document this:
   */
  void solverCoefficients(WINDSGeneralData *WGD);
};
