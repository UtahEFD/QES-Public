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

/**
 * @file WINDSOutputVisualization.cpp
 * @brief Specialized output classes derived from QESNetCDFOutput form
 * cell center data (used primarily for visualization)
 */

#include "WINDSOutputVisualization.h"

WINDSOutputVisualization::WINDSOutputVisualization(WINDSGeneralData *WGD,
                                                   WINDSInputData *WID,
                                                   const std::string &output_file)
  : QESNetCDFOutput(output_file)
{
  std::cout << "[QES-WINDS] \t Getting output fields for visualisation file" << std::endl;

  std::vector<std::string> fileOP = WID->fileOptions->outputFields;

  // copy of WGD pointer
  m_WGD = WGD;

  auto [nx, ny, nz] = WGD->domain.getDomainCellNum();
  auto [dx, dy, dz] = WGD->domain.getDomainSize();

  long numcell_cout = (nx - 1) * (ny - 1) * (nz - 2);

  z_out.resize(nz - 2);
  for (auto k = 1; k < nz - 1; k++) {
    z_out[k - 1] = m_WGD->domain.z[k];// Location of face centers in z-dir
  }

  x_out.resize(nx - 1);
  for (auto i = 0; i < nx - 1; i++) {
    x_out[i] = (i + 0.5) * dx;// Location of face centers in x-dir
  }

  y_out.resize(ny - 1);
  for (auto j = 0; j < ny - 1; j++) {
    y_out[j] = (j + 0.5) * dy;// Location of face centers in y-dir
  }

  // Output related data
  u_out.resize(numcell_cout, 0.0);
  v_out.resize(numcell_cout, 0.0);
  w_out.resize(numcell_cout, 0.0);
  mag_out.resize(numcell_cout, 0.0);

  icellflag_out.resize(numcell_cout, 0);
  icellflag2_out.resize(numcell_cout, 0);

  // set cell-centered data dimensions
  // space dimensions
  createDimension("x", "x-distance", "m", &x_out);
  createDimension("y", "y-distance", "m", &y_out);
  createDimension("z", "z-distance", "m", &z_out);

  // create 2D vector (time indep)
  createDimensionSet("terrain-grid", { "y", "x" });
  // create attributes
  createField("terrain", "terrain height", "m", "terrain-grid", &(m_WGD->terrain));

  createDimensionSet("wind-grid", { "t", "z", "y", "x" });
  // create attributes
  createField("u", "x-component velocity", "m s-1", "wind-grid", &u_out);
  createField("v", "y-component velocity", "m s-1", "wind-grid", &v_out);
  createField("w", "z-component velocity", "m s-1", "wind-grid", &w_out);
  createField("mag", "velocity magnitude", "m s-1", "wind-grid", &mag_out);

  createField("icell", "icell flag value", "--", "wind-grid", &icellflag_out);
  createField("icellInitial", "icell flag value", "--", "wind-grid", &icellflag2_out);

  // create output fields

  if (fileOP.empty() || fileOP[0] == "all") {
    addOutputFields(set_all_output_fields);
  } else {
    std::set<std::string> out;
    std::copy(fileOP.begin(), fileOP.end(), std::inserter(out, out.end()));
    addOutputFields(out);
  }
}

// Save output at cell-centered values
void WINDSOutputVisualization::save(QEStime timeOut)
{
  // get grid size (not output var size)
  auto [nx, ny, nz] = m_WGD->domain.getDomainCellNum();

  // set time
  timeCurrent = timeOut;

  // get cell-centered values
  for (auto k = 1; k < nz - 1; k++) {
    for (auto j = 0; j < ny - 1; j++) {
      for (auto i = 0; i < nx - 1; i++) {
        long icell_face = m_WGD->domain.face(i, j, k);
        long icell_cent = m_WGD->domain.cell(i, j, k - 1);
        u_out[icell_cent] = 0.5f * (m_WGD->u[m_WGD->domain.faceAdd(icell_face, 1, 0, 0)] + m_WGD->u[icell_face]);
        v_out[icell_cent] = 0.5f * (m_WGD->v[m_WGD->domain.faceAdd(icell_face, 0, 1, 0)] + m_WGD->v[icell_face]);
        w_out[icell_cent] = 0.5f * (m_WGD->w[m_WGD->domain.faceAdd(icell_face, 0, 0, 1)] + m_WGD->w[icell_face]);
        mag_out[icell_cent] = sqrt(u_out[icell_cent] * u_out[icell_cent]
                                   + v_out[icell_cent] * v_out[icell_cent]
                                   + w_out[icell_cent] * w_out[icell_cent]);

        icellflag_out[icell_cent] = m_WGD->icellflag[m_WGD->domain.cellAdd(icell_cent, 0, 0, 1)];
        icellflag2_out[icell_cent] = m_WGD->icellflag_initial[m_WGD->domain.cellAdd(icell_cent, 0, 0, 1)];
      }
    }
  }

  // save the fields to NetCDF files
  saveOutputFields();
};
