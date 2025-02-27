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

/**
 * @file test_functions.cpp
 */

#include "test_functions.h"
test_functions::test_functions(WINDSGeneralData *WGD, TURBGeneralData *TGD, const std::string &function_type)
  : domain(WGD->domain)
{
  std::cout << "[Test Functions]\t setting test functions" << std::endl;
  if (function_type == "linear") {
    u_test_function = new test_function_linearY(WGD);
    v_test_function = new test_function_linearZ(WGD);
    w_test_function = new test_function_linearX(WGD);
    c_test_function = new test_function_linearZ(WGD);
  } else if (function_type == "trig") {
    u_test_function = new test_function_trig(WGD);
    v_test_function = new test_function_trig(WGD);
    w_test_function = new test_function_trig(WGD);
    c_test_function = new test_function_trig(WGD);
  } else {
  }
  setTestValues(WGD, TGD);
  std::cout << "[Test Functions]\t done" << std::endl;
}

void test_functions::setTestValues(WINDSGeneralData *WGD, TURBGeneralData *TGD)
{
  // uv on vertical face -> k=0...nz-2
  for (int k = 0; k < domain.nz() - 1; k++) {
    for (int j = 0; j < domain.ny() - 1; j++) {
      for (int i = 0; i < domain.nx() - 1; i++) {
        long faceID = domain.face(i, j, k);
        WGD->u[faceID] = u_test_function->val(i * domain.dx(), domain.y[j], domain.z[k]);
      }
    }
    for (int j = 0; j < domain.ny() - 1; j++) {
      for (int i = 0; i < domain.nx() - 1; i++) {
        long faceID = domain.face(i, j, k);
        // WGD->v[faceID] = cos(a * domain.x[i]) + cos(b * j * WGD->dy) + sin(c * domain.z[k]);
        // WGD->v[faceID] = a * domain.x[i] + b * j * WGD->dy + c * domain.z[k];
        WGD->v[faceID] = v_test_function->val(domain.x[i], j * domain.dy(), domain.z[k]);
      }
    }
  }

  // w on horizontal face -> k=0...nz-1
  for (int k = 0; k < domain.nz() - 1; k++) {
    for (int j = 0; j < domain.ny() - 1; j++) {
      for (int i = 0; i < domain.nx() - 1; i++) {
        long faceID = domain.face(i, j, k);
        WGD->w[faceID] = w_test_function->val(domain.x[i], domain.y[j], domain.z_face[k]);
      }
    }
  }

  // cell center-> k=0...nz-2
  for (int k = 0; k < domain.nz() - 2; k++) {
    for (int j = 0; j < domain.ny() - 1; j++) {
      for (int i = 0; i < domain.nx() - 1; i++) {
        long cellID = domain.cell(i, j, k);
        TGD->txx[cellID] = c_test_function->val(domain.x[i], domain.y[j], domain.z[k]);
        TGD->txy[cellID] = c_test_function->val(domain.x[i], domain.y[j], domain.z[k]);
        TGD->txz[cellID] = c_test_function->val(domain.x[i], domain.y[j], domain.z[k]);
        TGD->tyy[cellID] = c_test_function->val(domain.x[i], domain.y[j], domain.z[k]);
        TGD->tyz[cellID] = c_test_function->val(domain.x[i], domain.y[j], domain.z[k]);
        TGD->tzz[cellID] = c_test_function->val(domain.x[i], domain.y[j], domain.z[k]);

        TGD->div_tau_x[cellID] = c_test_function->val(domain.x[i], domain.y[j], domain.z[k]);
        TGD->div_tau_y[cellID] = c_test_function->val(domain.x[i], domain.y[j], domain.z[k]);
        TGD->div_tau_z[cellID] = c_test_function->val(domain.x[i], domain.y[j], domain.z[k]);

        TGD->CoEps[cellID] = c_test_function->val(domain.x[i], domain.y[j], domain.z[k]);
      }
    }
  }
}
