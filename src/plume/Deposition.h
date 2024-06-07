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
 * This file is part of QES-Plume
 *
 * GPL-3.0 License
 *
 * QES-Plume is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, version 3 of the License.
 *
 * QES-Plume is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with QES-Plume. If not, see <https://www.gnu.org/licenses/>.
 ****************************************************************************/

/** @file Deposition.h
 * @brief
 */

#pragma once

#include <iostream>
#include <fstream>
#include <vector>
#include <list>
#include <cmath>
#include <cstring>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "util/QEStime.h"
#include "util/calcTime.h"
#include "util/Vector3Float.h"
// #include "Matrix3.h"
#include "Random.h"

#include "util/QESNetCDFOutput.h"
// #include "PlumeOutput.h"
// #include "PlumeOutputParticleData.h"

#include "winds/WINDSGeneralData.h"
#include "winds/TURBGeneralData.h"

#include "Interp.h"
#include "InterpNearestCell.h"
#include "InterpPowerLaw.h"
#include "InterpTriLinear.h"

#include "Particle.h"

class Deposition
{
private:
  Deposition() = default;

protected:
  // !!!! need implement this !!!!
  double boxSizeZ{};
  double c1 = 2.049;
  double c2 = 1.19;

public:
  explicit Deposition(const WINDSGeneralData *);
  ~Deposition() = default;

  void deposit(Particle *p,
               const double &disX,
               const double &disY,
               const double &disZ,
               const double &uTot,
               const double &vTot,
               const double &wTot,
               const double &vs,
               WINDSGeneralData *,
               TURBGeneralData *,
               Interp *);

  std::vector<float> x, y, z, z_face;

#ifdef _OPENMP
  std::vector<std::vector<float>> thread_depcvol;
#else
  std::vector<float> depcvol;
#endif

  int nbrFace;
};
