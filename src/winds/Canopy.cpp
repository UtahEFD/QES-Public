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
 * @file Canopy.cpp
 * @brief :document this:
 *
 * long desc
 *
 */

#include "Canopy.h"

#include "WINDSInputData.h"
#include "WINDSGeneralData.h"

Canopy::Canopy(const WINDSInputData *WID, WINDSGeneralData *WGD)
{
  wakeFlag = WID->canopies->wakeFlag;
  nx_canopy = WGD->nx - 1;
  ny_canopy = WGD->ny - 1;
  nz_canopy = WGD->nz - 1;

  // number of cell cell-center elements (2D)
  numcell_cent_2d = nx_canopy * ny_canopy;
  // number of cell cell-center elements (3D)
  numcell_cent_3d = nx_canopy * ny_canopy * nz_canopy;

  // Resize the canopy-related vectors
  canopy_bot.resize(numcell_cent_2d, 0.0);
  canopy_top.resize(numcell_cent_2d, 0.0);

  canopy_bot_index.resize(numcell_cent_2d, 0);
  canopy_top_index.resize(numcell_cent_2d, 0);

  canopy_base.resize(numcell_cent_2d, 0.0);
  canopy_height.resize(numcell_cent_2d, 0.0);

  canopy_z0.resize(numcell_cent_2d, 0.0);
  canopy_ustar.resize(numcell_cent_2d, 0.0);
  canopy_d.resize(numcell_cent_2d, 0.0);

  canopy_atten_coeff.resize(numcell_cent_3d, 0.0);
  icanopy_flag.resize(numcell_cent_3d, 0);

  wake_u_defect.resize(WGD->numcell_face, 0.0);
  wake_v_defect.resize(WGD->numcell_face, 0.0);
}

void Canopy::setCanopyElements(const WINDSInputData *WID, WINDSGeneralData *WGD)
{

  auto canopysetup_start = std::chrono::high_resolution_clock::now();// Start recording execution time

  if (WID->canopies->SHPData) {

    std::cout << "Creating canopies from shapefile..." << std::flush;

    std::vector<Building *> poly_buildings;
    // FM CLEANUP - NOT USED
    // float corner_height, min_height;
    std::vector<float> shpDomainSize(2), minExtent(2);
    WID->canopies->SHPData->getLocalDomain(shpDomainSize);
    WID->canopies->SHPData->getMinExtent(minExtent);

    //printf("\tShapefile Origin = (%.6f,%.6f)\n", minExtent[0], minExtent[1]);

    // If the shapefile is not covering the whole domain or the UTM coordinates
    // of the QES domain is different than shapefile origin
    if (WID->simParams->UTMx != 0.0 && WID->simParams->UTMy != 0.0) {
      minExtent[0] -= (minExtent[0] - WID->simParams->UTMx);
      minExtent[1] -= (minExtent[1] - WID->simParams->UTMy);
    }

    for (auto pIdx = 0u; pIdx < WID->canopies->SHPData->m_polygons.size(); pIdx++) {

      // convert the global polys to local domain coordinates
      for (auto lIdx = 0u; lIdx < WID->canopies->SHPData->m_polygons[pIdx].size(); lIdx++) {
        WID->canopies->SHPData->m_polygons[pIdx][lIdx].x_poly -= minExtent[0];
        WID->canopies->SHPData->m_polygons[pIdx][lIdx].y_poly -= minExtent[1];
      }

      // Setting base height for tree if there is a DEM file (TODO)
      if (WID->simParams->DTE_heightField && WID->simParams->DTE_mesh) {
        std::cerr << "Isolated tree from shapefile and DEM not implemented...\n";
      } else {
        //base_height.push_back(0.0);
      }

      for (auto lIdx = 0u; lIdx < WID->canopies->SHPData->m_polygons[pIdx].size(); lIdx++) {
        WID->canopies->SHPData->m_polygons[pIdx][lIdx].x_poly += WID->simParams->halo_x;
        WID->canopies->SHPData->m_polygons[pIdx][lIdx].y_poly += WID->simParams->halo_y;
      }

      // Loop to create each of the polygon buildings read in from the shapefile
      int cId = allCanopiesV.size();
      //allCanopiesV.push_back(new CanopyIsolatedTree(WID, WGD, pIdx));
      allCanopiesV.push_back(new CanopyIsolatedTree(WID->canopies->SHPData->m_polygons[pIdx],
                                                    WID->canopies->SHPData->m_features["H"][pIdx],
                                                    WID->canopies->SHPData->m_features["D"][pIdx],
                                                    0.0,
                                                    WID->canopies->SHPData->m_features["LAI"][pIdx],
                                                    cId));
      canopy_id.push_back(cId);
      allCanopiesV[cId]->setPolyBuilding(WGD);
      allCanopiesV[cId]->setCellFlags(WID, WGD, cId);
      effective_height.push_back(allCanopiesV[cId]->height_eff);
    }
    std::cout << "[done]" << std::endl;
  }

  for (size_t i = 0; i < WID->canopies->canopies.size(); i++) {
    int cId = allCanopiesV.size();
    allCanopiesV.push_back(WID->canopies->canopies[i]);

    for (auto pIdx = 0u; pIdx < allCanopiesV[cId]->polygonVertices.size(); pIdx++) {
      allCanopiesV[cId]->polygonVertices[pIdx].x_poly += WID->simParams->halo_x;
      allCanopiesV[cId]->polygonVertices[pIdx].y_poly += WID->simParams->halo_y;
    }

    canopy_id.push_back(cId);
    allCanopiesV[cId]->setPolyBuilding(WGD);
    allCanopiesV[cId]->setCellFlags(WID, WGD, cId);
    effective_height.push_back(allCanopiesV[cId]->height_eff);
  }

  std::cout << "Sorting canopies by height..." << std::flush;
  mergeSort(effective_height, allCanopiesV, canopy_id);
  std::cout << "[done]" << std::endl;

  auto canopysetup_finish = std::chrono::high_resolution_clock::now();// Finish recording execution time

  std::chrono::duration<float> elapsed_cut = canopysetup_finish - canopysetup_start;
  std::cout << "Elapsed time for canopy setup : " << elapsed_cut.count() << " s\n";

  return;
}

void Canopy::applyCanopyVegetation(WINDSGeneralData *WGD)
{
  // Call regression to define ustar and surface roughness of the canopy
  canopyRegression(WGD);

  for (size_t i = 0; i < allCanopiesV.size(); ++i) {
    // for now this does the canopy stuff for us
    //allBuildingsV[building_id[i]]->canopyVegetation(this, building_id[i]);
    allCanopiesV[canopy_id[i]]->canopyVegetation(WGD, canopy_id[i]);
  }

  return;
}

void Canopy::applyCanopyWake(WINDSGeneralData *WGD)
{

  if (wakeFlag == 1) {
    for (size_t i = 0; i < allCanopiesV.size(); ++i) {
      // for now this does the canopy stuff for us
      //allBuildingsV[building_id[i]]->canopyVegetation(this, building_id[i]);
      allCanopiesV[canopy_id[i]]->canopyWake(WGD, canopy_id[i]);
    }

    for (size_t id = 0u; id < wake_u_defect.size(); ++id) {
      WGD->u0[id] *= (1. - wake_u_defect[id]);
      wake_u_defect[id] = 0.0;
    }

    for (size_t id = 0u; id < wake_v_defect.size(); ++id) {
      WGD->v0[id] *= (1. - wake_v_defect[id]);
      wake_v_defect[id] = 0.0;
    }
  }

  return;
}

// Function to apply the urban canopy parameterization
// Based on the version contain Lucas Ulmer's modifications
void Canopy::canopyCioncoParam(WINDSGeneralData *WGD)
{

  float avg_atten; /**< average attenuation of the canopy */
  float veg_vel_frac; /**< vegetation velocity fraction */
  int num_atten;

  // Call regression to define ustar and surface roughness of the canopy
  //canopyRegression(WGD);

  for (auto j = 0; j < ny_canopy; j++) {
    for (auto i = 0; i < nx_canopy; i++) {
      int icell_2d = i + j * nx_canopy;

      if (canopy_top[icell_2d] > 0) {
        int icell_3d = i + j * nx_canopy + (canopy_top_index[icell_2d] - 1) * nx_canopy * ny_canopy;

        // Call the bisection method to find the root
        canopy_d[icell_2d] = canopyBisection(canopy_ustar[icell_2d],
                                             canopy_z0[icell_2d],
                                             canopy_height[icell_2d],
                                             canopy_atten_coeff[icell_3d],
                                             WGD->vk,
                                             0.0);
        // std::cout << "WGD->vk:" << WGD->vk << "\n";
        // std::cout << "WGD->canopy_atten[icell_cent]:" << WGD->canopy_atten[icell_cent] << "\n";
        if (canopy_d[icell_2d] == 10000) {
          std::cout << "bisection failed to converge"
                    << "\n";
          canopy_d[icell_2d] = canopySlopeMatch(canopy_z0[icell_2d], canopy_height[icell_2d], canopy_atten_coeff[icell_3d]);
        }

        /**< velocity at the height of the canopy */
        // Local variable - not being used by anything... so
        // commented out for now.
        //
        // float u_H = (WGD->canopy_ustar[id]/WGD->vk)*
        //  log((WGD->canopy_top[id]-WGD->canopy_d[id])/WGD->canopy_z0[id]);

        for (auto k = 1; k < WGD->nz - 1; k++) {
          int icell_face = i + j * WGD->nx + k * WGD->nx * WGD->ny;
          float z_rel = WGD->z[k] - WGD->terrain[icell_2d];

          if (WGD->z[k] < canopy_base[icell_2d]) {
            // below the terrain or building
          } else if (WGD->z[k] < canopy_top[icell_2d]) {
            if (canopy_atten_coeff[icell_3d] > 0) {
              icell_3d = i + j * nx_canopy + k * nx_canopy * ny_canopy;
              avg_atten = canopy_atten_coeff[icell_3d];


              if (canopy_atten_coeff[icell_3d + nx_canopy * ny_canopy] != canopy_atten_coeff[icell_3d]
                  || canopy_atten_coeff[icell_3d - nx_canopy * ny_canopy] != canopy_atten_coeff[icell_3d]) {
                num_atten = 1;
                if (canopy_atten_coeff[icell_3d + nx_canopy * ny_canopy] > 0) {
                  avg_atten += canopy_atten_coeff[icell_3d + nx_canopy * ny_canopy];
                  num_atten += 1;
                }
                if (canopy_atten_coeff[icell_3d - nx_canopy * ny_canopy] > 0) {
                  avg_atten += canopy_atten_coeff[icell_3d - nx_canopy * ny_canopy];
                  num_atten += 1;
                }
                avg_atten /= num_atten;
              }

              /*
              veg_vel_frac = log((canopy_top[icell_2d] - canopy_d[icell_2d])/
                                 canopy_z0[icell_2d])*exp(avg_atten*((WGD->z[k]/canopy_top[icell_2d])-1))/
                  log(WGD->z[k]/canopy_z0[icell_2d]);
              */

              // correction on the velocity within the canopy
              veg_vel_frac = log((canopy_height[icell_2d] - canopy_d[icell_2d]) / canopy_z0[icell_2d])
                             * exp(avg_atten * ((z_rel / canopy_height[icell_2d]) - 1)) / log(z_rel / canopy_z0[icell_2d]);
              // check if correction is bound and well defined
              if (veg_vel_frac > 1 || veg_vel_frac < 0) {
                veg_vel_frac = 1;
              }

              WGD->u0[icell_face] *= veg_vel_frac;
              WGD->v0[icell_face] *= veg_vel_frac;

              // at the edge of the canopy need to adjust velocity at the next face
              // use canopy_top to detect the edge (worke with level changes)
              if (j < WGD->ny - 2) {
                if (canopy_top[icell_2d + nx_canopy] == 0.0) {
                  WGD->v0[icell_face + WGD->nx] *= veg_vel_frac;
                }
              }
              if (i < WGD->nx - 2) {
                if (canopy_top[icell_2d + 1] == 0.0) {
                  WGD->u0[icell_face + 1] *= veg_vel_frac;
                }
              }
            }
          } else {
            // correction on the velocity above the canopy
            veg_vel_frac = log((z_rel - canopy_d[icell_2d]) / canopy_z0[icell_2d]) / log(z_rel / canopy_z0[icell_2d]);
            // check if correction is bound and well defined
            if (veg_vel_frac > 1 || veg_vel_frac < 0) {
              veg_vel_frac = 1;
            }

            WGD->u0[icell_face] *= veg_vel_frac;
            WGD->v0[icell_face] *= veg_vel_frac;

            // at the edge of the canopy need to adjust velocity at the next face
            // use canopy_top to detect the edge (worke with level changes)
            if (j < WGD->ny - 2) {
              icell_3d = i + j * nx_canopy + canopy_bot_index[icell_2d] * nx_canopy * ny_canopy;
              if (canopy_top[icell_2d + nx_canopy] == 0.0) {
                WGD->v0[icell_face + WGD->nx] *= veg_vel_frac;
              }
            }
            if (i < WGD->nx - 2) {
              icell_3d = i + j * nx_canopy + canopy_bot_index[icell_2d] * nx_canopy * ny_canopy;
              if (canopy_top[icell_2d + 1] == 0.0) {
                WGD->u0[icell_face + 1] *= veg_vel_frac;
              }
            }
          }
        }// end of for(auto k=1; k < WGD->nz-1; k++)
      }
    }
  }

  return;
}

void Canopy::canopyRegression(WINDSGeneralData *WGD)
{

  int k_top(0), counter;
  float sum_x, sum_y, sum_xy, sum_x_sq, local_mag;
  float y, xm, ym;

  for (auto j = 0; j < ny_canopy; j++) {
    for (auto i = 0; i < nx_canopy; i++) {
      int id = i + j * nx_canopy;
      if (canopy_top_index[id] > 0) {
        for (auto k = canopy_top_index[id]; k < WGD->nz - 2; k++) {
          k_top = k;
          if (canopy_top[id] + canopy_height[id] < WGD->z[k + 1])
            break;
        }
        if (k_top == canopy_top_index[id]) {
          k_top = canopy_top_index[id] + 1;
        }
        if (k_top > WGD->nz - 1) {
          k_top = WGD->nz - 1;
        }
        sum_x = 0;
        sum_y = 0;
        sum_xy = 0;
        sum_x_sq = 0;
        counter = 0;
        for (auto k = canopy_top_index[id]; k <= k_top; k++) {
          counter += 1;
          int icell_face = i + j * WGD->nx + k * WGD->nx * WGD->ny;
          local_mag = sqrt(pow(WGD->u0[icell_face], 2.0) + pow(WGD->v0[icell_face], 2.0));
          y = log(WGD->z[k] - WGD->terrain[i + j * (WGD->nx - 1)]);
          sum_x += local_mag;
          sum_y += y;
          sum_xy += local_mag * y;
          sum_x_sq += pow(local_mag, 2.0);
        }

        canopy_ustar[id] = WGD->vk * (((counter * sum_x_sq) - pow(sum_x, 2.0)) / ((counter * sum_xy) - (sum_x * sum_y)));
        xm = sum_x / counter;
        ym = sum_y / counter;
        canopy_z0[id] = exp(ym - ((WGD->vk / canopy_ustar[id])) * xm);

        //std::cout << xm << " " << ym << " " << sum_y << " " << sum_x_sq << " " << canopy_ustar[id] << " " << canopy_z0[id] << std::endl;

      }// end of if (canopy_top_index[id] > 0)
    }
  }

  return;
}

float Canopy::canopyBisection(float ustar, float z0, float canopy_top, float canopy_atten, float vk, float psi_m)
{
  int iter;
  float uhc, d, d1, d2;
  float tol, fnew, fi;

  tol = z0 / 100;
  fnew = tol * 10;

  d1 = z0;
  d2 = canopy_top;
  d = (d1 + d2) / 2;

  uhc = (ustar / vk) * (log((canopy_top - d1) / z0) + psi_m);
  fi = ((canopy_atten * uhc * vk) / ustar) - canopy_top / (canopy_top - d1);

  if (canopy_atten > 0) {
    iter = 0;
    while (iter < 200 && abs(fnew) > tol && d < canopy_top && d > z0) {
      iter += 1;
      d = (d1 + d2) / 2;
      uhc = (ustar / vk) * (log((canopy_top - d) / z0) + psi_m);
      fnew = ((canopy_atten * uhc * vk) / ustar) - canopy_top / (canopy_top - d);
      if (fnew * fi > 0) {
        d1 = d;
      } else if (fnew * fi < 0) {
        d2 = d;
      }
    }
    if (d > canopy_top) {
      d = 10000;
    }

  } else {
    d = 0.99 * canopy_top;
  }

  return d;
}

float Canopy::canopySlopeMatch(float z0, float canopy_top, float canopy_atten)
{

  int iter;
  float tol, d, d1, d2, f;

  tol = z0 / 100;
  // f is the root of the equation (to find d)
  // log[(H-d)/z0] = H/[a(H-d)]
  f = tol * 10;

  // initial bound for bisection method (d1,d2)
  // d1 min displacement possible
  // d2 max displacement possible - canopy top
  d1 = z0;
  if (z0 <= canopy_top) {
    d1 = z0;
  } else if (z0 > canopy_top) {
    d1 = 0.1;
  }
  d2 = canopy_top;
  d = (d1 + d2) / 2;

  if (canopy_atten > 0) {
    iter = 0;
    // bisection method to find the displacement height
    while (iter < 200 && abs(f) > tol && d < canopy_top && d > z0) {
      iter += 1;
      d = (d1 + d2) / 2;
      f = log((canopy_top - d) / z0) - (canopy_top / (canopy_atten * (canopy_top - d)));
      if (f > 0) {
        d1 = d;
      } else if (f < 0) {
        d2 = d;
      }
    }
    // if displacement found higher that canopy top => shifted down
    if (d > canopy_top) {
      d = 0.7 * canopy_top;
    }
  } else {
    // return this if attenuation coeff is 0.
    d = 10000;
  }

  // return displacement height
  return d;
}

void Canopy::mergeSort(std::vector<float> &effective_height, std::vector<Building *> allCanopiesV, std::vector<int> &tree_id)
{
  // if the size of the array is 1, it is already sorted
  if (allCanopiesV.size() == 1) {
    return;
  }

  if (allCanopiesV.size() > 1) {
    // make left and right sides of the data
    std::vector<float> effective_height_L, effective_height_R;
    std::vector<int> tree_id_L, tree_id_R;
    std::vector<Building *> allCanopiesV_L, allCanopiesV_R;
    effective_height_L.resize(allCanopiesV.size() / 2);
    effective_height_R.resize(allCanopiesV.size() - allCanopiesV.size() / 2);
    tree_id_L.resize(allCanopiesV.size() / 2);
    tree_id_R.resize(allCanopiesV.size() - allCanopiesV.size() / 2);
    allCanopiesV_L.resize(allCanopiesV.size() / 2);
    allCanopiesV_R.resize(allCanopiesV.size() - allCanopiesV.size() / 2);

    // copy data from the main data set to the left and right children
    size_t lC = 0, rC = 0;
    for (size_t i = 0; i < allCanopiesV.size(); i++) {
      if (i < allCanopiesV.size() / 2) {
        effective_height_L[lC] = effective_height[i];
        allCanopiesV_L[lC] = allCanopiesV[i];
        tree_id_L[lC++] = tree_id[i];

      } else {
        effective_height_R[rC] = effective_height[i];
        allCanopiesV_R[rC] = allCanopiesV[i];
        tree_id_R[rC++] = tree_id[i];
      }
    }
    // recursively sort the children
    mergeSort(effective_height_L, allCanopiesV_L, tree_id_L);
    mergeSort(effective_height_R, allCanopiesV_R, tree_id_R);

    // compare the sorted children to place the data into the main array
    lC = rC = 0;
    for (size_t i = 0; i < allCanopiesV.size(); i++) {
      if (rC == effective_height_R.size() || (lC != effective_height_L.size() && effective_height_L[lC] > effective_height_R[rC])) {
        effective_height[i] = effective_height_L[lC];
        tree_id[i] = tree_id_L[lC++];
      } else {
        effective_height[i] = effective_height_R[rC];
        tree_id[i] = tree_id_R[rC++];
      }
    }
  }

  return;
}
