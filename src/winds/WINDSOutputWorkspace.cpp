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
 * @file WINDSOutputWorkspace.cpp
 * @brief Specialized output classes derived from QESNetCDFOutput for
 * face center data (used for turbulence,...)
 */

#include "WINDSOutputWorkspace.h"

WINDSOutputWorkspace::WINDSOutputWorkspace(WINDSGeneralData *WGD, const std::string &output_file)
  : QESNetCDFOutput(output_file)
{
  std::cout << "[Output] \t Setting fields of workspace file" << std::endl;

  // copy of WGD pointer
  m_WGD = WGD;

  // domain size information:
  int nx = m_WGD->nx;
  int ny = m_WGD->ny;
  int nz = m_WGD->nz;

  // Location of cell centers in x-dir
  m_x.resize(nx - 1);
  for (auto i = 0; i < nx - 1; i++) {
    // x_cc[i] = (i + 0.5) * m_WGD->dx;
    m_x[i] = m_WGD->x[i];
  }
  // Location of cell centers in y-dir
  m_y.resize(ny - 1);
  for (auto j = 0; j < ny - 1; j++) {
    // y_cc[j] = (j + 0.5) * m_WGD->dy;
    m_y[j] = m_WGD->y[j];
  }
  // Location of cell centers in z-dir
  m_z.resize(nz - 1);
  m_dz_array.resize(nz - 1, 0.0);
  for (auto k = 0; k < nz - 1; k++) {
    m_z[k] = m_WGD->z[k];
    m_dz_array[k] = m_WGD->dz_array[k];
  }

  // Location of face centers in x-dir
  m_x_face.resize(nx);
  m_x_face[0] = m_WGD->x[0] - m_WGD->dx;
  for (auto i = 0; i < nx - 1; i++) {
    m_x_face[i + 1] = m_WGD->x[i];
  }
  // Location of cell centers in y-dir
  m_y_face.resize(ny);
  m_y_face[0] = m_WGD->y[0] - m_WGD->dy;
  for (auto j = 0; j < ny - 1; j++) {
    m_y_face[j + 1] = m_WGD->y[j] + m_WGD->dy;
  }
  // Location of cell centers in z-dir
  m_z_face.resize(nz);
  for (auto k = 0; k < nz; ++k) {
    m_z_face[k] = m_WGD->z_face[k];
  }

  // set face-centered data dimensions
  // space dimensions
  createDimension("x_face", "x-face", "m", &m_x_face);
  createDimension("y_face", "y-face", "m", &m_y_face);
  createDimension("z_face", "z-face", "m", &m_z_face);

  // 3D vector dimension (time dep)
  createDimensionSet("face-grid", { "t", "z_face", "y_face", "x_face" });
  // create attributes
  createAttVector("u", "x-component velocity", "m s-1", "face-grid", &(m_WGD->u));
  createAttVector("v", "y-component velocity", "m s-1", "face-grid", &(m_WGD->v));
  createAttVector("w", "z-component velocity", "m s-1", "face-grid", &(m_WGD->w));

  createAttVector("u0", "x-component initial velocity", "m s-1", "face-grid", &(m_WGD->u0));
  createAttVector("v0", "y-component initial velocity", "m s-1", "face-grid", &(m_WGD->v0));
  createAttVector("w0", "z-component initial velocity", "m s-1", "face-grid", &(m_WGD->w0));

  // set cell-centered data dimensions
  // space dimensions
  createDimension("x", "x-distance", "m", &m_x);
  createDimension("y", "y-distance", "m", &m_y);
  createDimension("z", "z-distance", "m", &m_z);

  createDimensionSet("z-grid", { "z" });
  createAttVector("dz_array", "dz size of the cells", "m", "z-grid", &m_dz_array);

  // create 2D vector (surface, indep of time)
  createDimensionSet("terrain-grid", { "y", "x" });
  // create attributes
  createAttVector("terrain", "terrain height", "m", "terrain-grid", &(m_WGD->terrain));
  createAttVector("z0_u", "terrain aerodynamic roughness, u", "m", "terrain-grid", &(m_WGD->z0_domain_u));
  createAttVector("z0_v", "terrain aerodynamic roughness, v", "m", "terrain-grid", &(m_WGD->z0_domain_v));

  // 3D vector dimension
  createDimensionSet("cell-grid-no-time", { "z", "y", "x" });
  // create attributes
  createAttVector("mixlength", "distance to nearest object", "m", "cell-grid-no-time", &(m_WGD->mixingLengths));

  // 3D vector dimension (time dep)
  createDimensionSet("cell-grid", { "t", "z", "y", "x" });
  // create attributes
  createAttVector("icellflag", "icell flag value", "--", "cell-grid", &(m_WGD->icellflag));

  // attributes for coefficients for SOR solver
  createAttVector("e", "e cut-cell coefficient", "--", "cell-grid", &(m_WGD->e));
  createAttVector("f", "f cut-cell coefficient", "--", "cell-grid", &(m_WGD->f));
  createAttVector("g", "g cut-cell coefficient", "--", "cell-grid", &(m_WGD->g));
  createAttVector("h", "h cut-cell coefficient", "--", "cell-grid", &(m_WGD->h));
  createAttVector("m", "m cut-cell coefficient", "--", "cell-grid", &(m_WGD->m));
  createAttVector("n", "n cut-cell coefficient", "--", "cell-grid", &(m_WGD->n));

  // attribute for the volume fraction (cut-cell)
  createAttVector("building_volume_frac", "building volume fraction", "--", "cell-grid", &(m_WGD->building_volume_frac));
  createAttVector("terrain_volume_frac", "terrain volume fraction", "--", "cell-grid", &(m_WGD->terrain_volume_frac));

  addOutputFields(set_all_output_fields);
}


// Save output at cell-centered values
void WINDSOutputWorkspace::save(QEStime timeOut)
{

  // set time
  timeCurrent = timeOut;

  // std::string s = timeOut.getTimestamp();
  // std::copy(s.begin(), s.end(), timestamp.begin());

  // save fields
  saveOutputFields();
};


// [FM] Feb.28.2020 OBSOLETE
void WINDSOutputWorkspace::setBuildingFields(NcDim *NcDim_t, NcDim *NcDim_building)
{
  int nBuildings = m_WGD->allBuildingsV.size();

  building_rotation.resize(nBuildings, 0.0);
  canopy_rotation.resize(nBuildings, 0.0);

  L.resize(nBuildings, 0.0);
  W.resize(nBuildings, 0.0);
  H.resize(nBuildings, 0.0);

  length_eff.resize(nBuildings, 0.0);
  width_eff.resize(nBuildings, 0.0);
  height_eff.resize(nBuildings, 0.0);
  base_height.resize(nBuildings, 0.0);

  building_cent_x.resize(nBuildings, 0.0);
  building_cent_y.resize(nBuildings, 0.0);

  i_start.resize(nBuildings, 0);
  i_end.resize(nBuildings, 0);
  j_start.resize(nBuildings, 0);
  j_end.resize(nBuildings, 0);
  k_end.resize(nBuildings, 0);

  i_cut_start.resize(nBuildings, 0);
  i_cut_end.resize(nBuildings, 0);
  j_cut_start.resize(nBuildings, 0);
  j_cut_end.resize(nBuildings, 0);
  k_cut_end.resize(nBuildings, 0);

  i_building_cent.resize(nBuildings, 0);
  j_building_cent.resize(nBuildings, 0);

  upwind_dir.resize(nBuildings, 0.0);
  Lr.resize(nBuildings, 0.0);

  // vector of dimension for building information
  std::vector<NcDim> dim_vect_building;
  dim_vect_building.push_back(*NcDim_building);

  // create attributes
  createAttVector("building_rotation", "rotation of building", "rad", dim_vect_building, &building_rotation);
  createAttVector("canopy_rotation", "rotation of canopy", "rad", dim_vect_building, &building_rotation);

  createAttVector("L", "length of building", "m", dim_vect_building, &L);
  createAttVector("W", "width of building", "m", dim_vect_building, &L);
  createAttVector("H", "height of building", "m", dim_vect_building, &H);

  createAttVector("height_eff", "effective height", "m", dim_vect_building, &height_eff);
  createAttVector("base_height", "base height", "m", dim_vect_building, &base_height);

  createAttVector("building_cent_x", "x-coordinate of centroid", "m", dim_vect_building, &building_cent_x);
  createAttVector("building_cent_y", "y-coordinate of centroid", "m", dim_vect_building, &building_cent_y);

  createAttVector("i_start", "x-index start", "--", dim_vect_building, &i_start);
  createAttVector("i_end", "x-index end", "--", dim_vect_building, &i_end);
  createAttVector("j_start", "y-index start", "--", dim_vect_building, &j_start);
  createAttVector("j_end", "y-index end", "--", dim_vect_building, &j_end);
  createAttVector("k_start", "z-index end", "--", dim_vect_building, &k_end);

  createAttVector("i_cut_start", "x-index start cut-cell", "--", dim_vect_building, &i_cut_start);
  createAttVector("i_cut_end", "x-index end cut-cell", "--", dim_vect_building, &i_cut_end);
  createAttVector("j_cut_start", "y-index start cut-cell", "--", dim_vect_building, &j_cut_start);
  createAttVector("j_cut_end", "y-index end cut-cell", "--", dim_vect_building, &j_cut_end);
  createAttVector("k_cut_start", "z-index end cut-cell", "--", dim_vect_building, &k_cut_end);

  createAttVector("i_building_cent", "x-index of centroid", "--", dim_vect_building, &i_building_cent);
  createAttVector("i_building_cent", "y-index of centroid", "--", dim_vect_building, &i_building_cent);

  // temporary vector to add the fields into output_fields for output.
  std::vector<string> tmp_fields;
  tmp_fields.clear();// clear the vector
  tmp_fields = { "building_rotation", "canopy_rotation", "L", "W", "H", "height_eff", "base_height", "building_cent_x", "building_cent_y", "i_start", "i_end", "j_start", "j_end", "k_start", "i_cut_start", "i_cut_end", "j_cut_start", "j_cut_end", "k_cut_start", "i_building_cent", "j_building_cent" };
  output_fields.insert(output_fields.end(), tmp_fields.begin(), tmp_fields.end());

  // vector of dimension for time dep building information
  std::vector<NcDim> dim_vect_building_t;
  dim_vect_building_t.push_back(*NcDim_t);
  dim_vect_building_t.push_back(*NcDim_building);

  // create attributes
  createAttVector("length_eff", "effective length", "m", dim_vect_building_t, &length_eff);
  createAttVector("width_eff", "effective width", "m", dim_vect_building_t, &width_eff);

  createAttVector("upwind_dir", "upwind wind direction", "rad", dim_vect_building_t, &upwind_dir);
  createAttVector("Lr", "Length of far wake zone", "m", dim_vect_building_t, &Lr);

  // temporary vector to add the fields into output_fields for output.
  tmp_fields.clear();// clear the vector
  tmp_fields = { "length_eff", "width_eff", "upwind_dir", "Lr" };
  output_fields.insert(output_fields.end(), tmp_fields.begin(), tmp_fields.end());

  return;
}

void WINDSOutputWorkspace::setAllOutputFields()
{
  all_output_fields.clear();
  // all possible output fields need to be add to this list
  all_output_fields = { "x",
                        "y",
                        "z",
                        "x_face",
                        "y_face",
                        "z_face",
                        "dz_array",
                        "u",
                        "v",
                        "w",
                        "u0",
                        "v0",
                        "w0",
                        "icellflag",
                        "terrain",
                        "z0_u",
                        "z0_v",
                        "e",
                        "f",
                        "g",
                        "h",
                        "m",
                        "n",
                        "building_volume_frac",
                        "terrain_volume_frac",
                        "mixlength" };
}

// [FM] Feb.28.2020 OBSOLETE
void WINDSOutputWorkspace::getBuildingFields()
{

#if 0
  int nBuildings = m_WGD->allBuildingsV.size();

  // information only needed once (at output_counter==0)
  if (output_counter == 0) {
    // copy time independent fields
    for (int id = 0; id < nBuildings; ++id) {
      building_rotation[id] = m_WGD->allBuildingsV[id]->building_rotation;
      canopy_rotation[id] = m_WGD->allBuildingsV[id]->canopy_rotation;

      L[id] = m_WGD->allBuildingsV[id]->L;
      W[id] = m_WGD->allBuildingsV[id]->W;
      H[id] = m_WGD->allBuildingsV[id]->H;

      height_eff[id] = m_WGD->allBuildingsV[id]->height_eff;
      base_height[id] = m_WGD->allBuildingsV[id]->base_height;

      building_cent_x[id] = m_WGD->allBuildingsV[id]->building_cent_x;
      building_cent_y[id] = m_WGD->allBuildingsV[id]->building_cent_y;

      i_start[id] = m_WGD->allBuildingsV[id]->i_start;
      i_end[id] = m_WGD->allBuildingsV[id]->i_end;
      j_start[id] = m_WGD->allBuildingsV[id]->j_start;
      j_end[id] = m_WGD->allBuildingsV[id]->j_end;
      k_end[id] = m_WGD->allBuildingsV[id]->k_end;

      i_cut_start[id] = m_WGD->allBuildingsV[id]->i_cut_start;
      i_cut_end[id] = m_WGD->allBuildingsV[id]->i_cut_end;
      j_cut_start[id] = m_WGD->allBuildingsV[id]->j_cut_start;
      j_cut_end[id] = m_WGD->allBuildingsV[id]->j_cut_end;
      k_cut_end[id] = m_WGD->allBuildingsV[id]->k_cut_end;

      i_building_cent[id] = m_WGD->allBuildingsV[id]->i_building_cent;
      j_building_cent[id] = m_WGD->allBuildingsV[id]->j_building_cent;
    }
  }

  // copy time dependent fields
  for (int id = 0; id < nBuildings; ++id) {
    length_eff[id] = m_WGD->allBuildingsV[id]->length_eff;
    width_eff[id] = m_WGD->allBuildingsV[id]->width_eff;

    upwind_dir[id] = m_WGD->allBuildingsV[id]->upwind_dir;
    Lr[id] = m_WGD->allBuildingsV[id]->Lr;
  }
#endif
  return;
}
