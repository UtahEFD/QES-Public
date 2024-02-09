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

/**
 * @file TURBOutput.cpp
 * @brief :document this:
 */

#include "TURBOutput.h"

TURBOutput::TURBOutput(TURBGeneralData *tgd, std::string output_file)
  : QESNetCDFOutput(output_file)
{
  std::cout << "[QES-TURB] \t Setting output fields for turbulence data" << std::endl;

  // set list of fields to save, no option available for this file
  output_fields = all_output_fields;

  m_TGD = tgd;

  int nx = m_TGD->nx;
  int ny = m_TGD->ny;
  int nz = m_TGD->nz;

  // unused: long numcell_cout = (nx-1)*(ny-1)*(nz-1);

  // set cell-centered data dimensions
  // space dimensions
  createDimension("x", "x-distance", "m", &(m_TGD->x));
  createDimension("y", "y-distance", "m", &(m_TGD->y));
  createDimension("z", "z-distance", "m", &(m_TGD->z));

  // create attributes space dimensions


  // 3D vector dimension (time dep)
  createDimensionSet("turb-grid", { "t", "z", "y", "x" });

  createField("iturbflag", "icell turb flag", "--", "turb-grid", &(m_TGD->iturbflag));

  // create attributes for velocity gradient tensor
  createField("Gxx", "velocity gradient tensor: Gxx = dudx", "s-1", "turb-grid", &(m_TGD->Gxx));
  createField("Gyx", "velocity gradient tensor: Gyx = dvdx", "s-1", "turb-grid", &(m_TGD->Gyx));
  createField("Gzx", "velocity gradient tensor: Gzx = dwdx", "s-1", "turb-grid", &(m_TGD->Gzx));
  createField("Gxy", "velocity gradient tensor: Gxy = dudy", "s-1", "turb-grid", &(m_TGD->Gxy));
  createField("Gyy", "velocity gradient tensor: Gyy = dvdy", "s-1", "turb-grid", &(m_TGD->Gyy));
  createField("Gzy", "velocity gradient tensor: Gzy = dwdy", "s-1", "turb-grid", &(m_TGD->Gzy));
  createField("Gxz", "velocity gradient tensor: Gxz = dudz", "s-1", "turb-grid", &(m_TGD->Gxz));
  createField("Gyz", "velocity gradient tensor: Gyz = dvdz", "s-1", "turb-grid", &(m_TGD->Gyz));
  createField("Gzz", "velocity gradient tensor: Gzz = dwdz", "s-1", "turb-grid", &(m_TGD->Gzz));

  // create attribute for mixing length
  createField("L", "mixing length", "m", "turb-grid", &(m_TGD->Lm));

  // create derived attributes
  createField("txx", "uu-component of stress tensor", "m2s-2", "turb-grid", &(m_TGD->txx));
  createField("tyy", "vv-component of stress tensor", "m2s-2", "turb-grid", &(m_TGD->tyy));
  createField("tzz", "ww-component of stress tensor", "m2s-2", "turb-grid", &(m_TGD->tzz));
  createField("txy", "uv-component of stress tensor", "m2s-2", "turb-grid", &(m_TGD->txy));
  createField("txz", "uw-component of stress tensor", "m2s-2", "turb-grid", &(m_TGD->txz));
  createField("tyz", "vw-component of stress tensor", "m2s-2", "turb-grid", &(m_TGD->tyz));
  createField("tke", "turbulent kinetic energy", "m2s-2", "turb-grid", &(m_TGD->tke));
  createField("CoEps", "dissipation rate", "m2s-3", "turb-grid", &(m_TGD->CoEps));
  createField("div_tau_x", "x-component of stress-tensor divergence", "ms-2", "turb-grid", &(m_TGD->div_tau_x));
  createField("div_tau_y", "y-component of stress-tensor divergence", "ms-2", "turb-grid", &(m_TGD->div_tau_y));
  createField("div_tau_z", "z-component of stress-tensor divergence", "ms-2", "turb-grid", &(m_TGD->div_tau_z));

  // create output fields
  addOutputFields(set_all_output_fields);
}

// Save output at cell-centered values
void TURBOutput::save(QEStime timeOut)
{

  // set time
  timeCurrent = timeOut;

  // save fields
  saveOutputFields();
};
