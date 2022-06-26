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

/**
 * @file TURBOutput.cpp
 * @brief :document this:
 */

#include "TURBOutput.h"

TURBOutput::TURBOutput(TURBGeneralData *tgd, std::string output_file)
  : QESNetCDFOutput(output_file)
{
  std::cout << "[Output] \t Setting output fields for Turbulence data" << std::endl;

  setAllOutputFields();

  // set list of fields to save, no option available for this file
  output_fields = all_output_fields;

  m_TGD = tgd;

  int nx = m_TGD->nx;
  int ny = m_TGD->ny;
  int nz = m_TGD->nz;

  // unused: long numcell_cout = (nx-1)*(ny-1)*(nz-1);

  // set cell-centered data dimensions
  // space dimensions
  NcDim NcDim_x_cc = addDimension("x", nx - 1);
  NcDim NcDim_y_cc = addDimension("y", ny - 1);
  NcDim NcDim_z_cc = addDimension("z", nz - 1);

  // create attributes space dimensions
  std::vector<NcDim> dim_vect_x;
  dim_vect_x.push_back(NcDim_x_cc);
  createAttVector("x", "x-distance", "m", dim_vect_x, &(m_TGD->x));
  std::vector<NcDim> dim_vect_y;
  dim_vect_y.push_back(NcDim_y_cc);
  createAttVector("y", "y-distance", "m", dim_vect_y, &(m_TGD->y));
  std::vector<NcDim> dim_vect_z;
  dim_vect_z.push_back(NcDim_z_cc);
  createAttVector("z", "z-distance", "m", dim_vect_z, &(m_TGD->z));

  // 3D vector dimension (time dep)
  std::vector<NcDim> dim_vect_cc;
  dim_vect_cc.push_back(NcDim_t);
  dim_vect_cc.push_back(NcDim_z_cc);
  dim_vect_cc.push_back(NcDim_y_cc);
  dim_vect_cc.push_back(NcDim_x_cc);

  createAttVector("iturbflag", "icell turb flag", "--", dim_vect_cc, &(m_TGD->iturbflag));

  // create attributes for strain-rate stress tensor
  /*
    createAttVector("Sxx", "uu-component of strain-rate tensor", "s-1", dim_vect_cc, &(m_TGD->Sxx));
    createAttVector("Syy", "vv-component of strain-rate tensor", "s-1", dim_vect_cc, &(m_TGD->Syy));
    createAttVector("Szz", "ww-component of strain-rate tensor", "s-1", dim_vect_cc, &(m_TGD->Szz));
    createAttVector("Sxy", "uv-component of strain-rate tensor", "s-1", dim_vect_cc, &(m_TGD->Sxy));
    createAttVector("Sxz", "uw-component of strain-rate tensor", "s-1", dim_vect_cc, &(m_TGD->Sxz));
    createAttVector("Syz", "vw-component of strain-rate tensor", "s-1", dim_vect_cc, &(m_TGD->Syz));
  */

  // create attributes for velocity gradient tensor
  createAttVector("Gxx", "velocity gradient tensor: Gxx = dudx", "s-1", dim_vect_cc, &(m_TGD->Gxx));
  createAttVector("Gyx", "velocity gradient tensor: Gyx = dvdx", "s-1", dim_vect_cc, &(m_TGD->Gyx));
  createAttVector("Gzx", "velocity gradient tensor: Gzx = dwdx", "s-1", dim_vect_cc, &(m_TGD->Gzx));
  createAttVector("Gxy", "velocity gradient tensor: Gxy = dudy", "s-1", dim_vect_cc, &(m_TGD->Gxy));
  createAttVector("Gyy", "velocity gradient tensor: Gyy = dvdy", "s-1", dim_vect_cc, &(m_TGD->Gyy));
  createAttVector("Gzy", "velocity gradient tensor: Gzy = dwdy", "s-1", dim_vect_cc, &(m_TGD->Gzy));
  createAttVector("Gxz", "velocity gradient tensor: Gxz = dudz", "s-1", dim_vect_cc, &(m_TGD->Gxz));
  createAttVector("Gyz", "velocity gradient tensor: Gyz = dvdz", "s-1", dim_vect_cc, &(m_TGD->Gyz));
  createAttVector("Gzz", "velocity gradient tensor: Gzz = dwdz", "s-1", dim_vect_cc, &(m_TGD->Gzz));

  // create attribute for mixing length
  createAttVector("L", "mixing length", "m", dim_vect_cc, &(m_TGD->Lm));

  // create derived attributes
  createAttVector("txx", "uu-component of stress tensor", "m2s-2", dim_vect_cc, &(m_TGD->txx));
  createAttVector("tyy", "vv-component of stress tensor", "m2s-2", dim_vect_cc, &(m_TGD->tyy));
  createAttVector("tzz", "ww-component of stress tensor", "m2s-2", dim_vect_cc, &(m_TGD->tzz));
  createAttVector("txy", "uv-component of stress tensor", "m2s-2", dim_vect_cc, &(m_TGD->txy));
  createAttVector("txz", "uw-component of stress tensor", "m2s-2", dim_vect_cc, &(m_TGD->txz));
  createAttVector("tyz", "vw-component of stress tensor", "m2s-2", dim_vect_cc, &(m_TGD->tyz));
  createAttVector("tke", "turbulent kinetic energy", "m2s-2", dim_vect_cc, &(m_TGD->tke));
  createAttVector("CoEps", "dissipation rate", "m2s-3", dim_vect_cc, &(m_TGD->CoEps));
  createAttVector("div_tau_x", "x-component of stress-tensor divergence", "ms-2", dim_vect_cc, &(m_TGD->div_tau_x));
  createAttVector("div_tau_y", "y-component of stress-tensor divergence", "ms-2", dim_vect_cc, &(m_TGD->div_tau_y));
  createAttVector("div_tau_z", "z-component of stress-tensor divergence", "ms-2", dim_vect_cc, &(m_TGD->div_tau_z));

  // create output fields
  addOutputFields();
}

void TURBOutput::setAllOutputFields()
{
  all_output_fields.clear();
  // all possible output fields need to be add to this list
  all_output_fields = { "x",
                        "y",
                        "z",
                        "iturbflag",
                        "Gxx",
                        "Gyx",
                        "Gzx",
                        "Gxy",
                        "Gyy",
                        "Gzy",
                        "Gxz",
                        "Gyz",
                        "Gzz",
                        "L",
                        "txx",
                        "txy",
                        "txz",
                        "tyz",
                        "tyy",
                        "tzz",
                        "tke",
                        "CoEps",
                        "div_tau_x",
                        "div_tau_y",
                        "div_tau_z" };
}


// Save output at cell-centered values
void TURBOutput::save(QEStime timeOut)
{

  // set time
  timeCurrent = timeOut;

  // save fields
  saveOutputFields();
};
