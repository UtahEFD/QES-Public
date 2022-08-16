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
 * @file WINDSGeneralData.cpp
 * @brief :document this:
 */

#include "WINDSGeneralData.h"

#define PBSTR "||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||"
#define PBWIDTH 60
#define LIMIT 99999999.0f

WINDSGeneralData::WINDSGeneralData(const WINDSInputData *WID, int solverType)
{
  if (WID->simParams->upwindCavityFlag == 1) {
    lengthf_coeff = 2.0;
  } else {
    lengthf_coeff = 1.5;
  }

  // Determines how wakes behind buildings are calculated
  if (WID->simParams->wakeFlag > 1) {
    cavity_factor = 1.1;
    wake_factor = 0.1;
  } else {
    cavity_factor = 1.0;
    wake_factor = 0.0;
  }

  // converting the domain rotation to radians from degrees -- input
  // assumes degrees
  theta = (WID->simParams->domainRotation * pi / 180.0);

  // Pull Domain Size information from the WINDSInputData structure --
  // this is either read in from the XML files and/or potentially
  // calculated based on the geographic data that was loaded
  // (i.e. DEM files). It is important to get this data from the
  // main input structure.
  //
  // This is done to make reference to nx, ny and nz easier in this function
  //Vector3Int domainInfo;
  //domainInfo = *(WID->simParams->domain);
  nx = WID->simParams->domain[0];
  ny = WID->simParams->domain[1];
  nz = WID->simParams->domain[2];

  // Modify the domain size to fit the Staggered Grid used in the solver
  nx += 1;// +1 for Staggered grid
  ny += 1;// +1 for Staggered grid
  nz += 2;// +2 for staggered grid and ghost cell

  //Vector3 gridInfo;
  //gridInfo = *(WID->simParams->grid);
  dx = WID->simParams->grid[0];// Grid resolution in x-direction
  dy = WID->simParams->grid[1];// Grid resolution in y-direction
  dz = WID->simParams->grid[2];// Grid resolution in z-direction
  dxy = MIN_S(dx, dy);

  numcell_cout = (nx - 1) * (ny - 1) * (nz - 2);// Total number of cell-centered values in domain
  numcell_cout_2d = (nx - 1) * (ny - 1);// Total number of horizontal cell-centered values in domain
  numcell_cent = (nx - 1) * (ny - 1) * (nz - 1);// Total number of cell-centered values in domain
  numcell_face = nx * ny * nz;// Total number of face-centered values in domain

  // where should this really go?
  //
  // Need to now take all WRF station data and convert to
  // sensors
  if (WID->simParams->wrfInputData) {

    WRFInput *wrf_ptr = WID->simParams->wrfInputData;

    // Use WID->simParams->m_domIType == WRFOnly -- to indicate wrf
    // interp data usage
    if (WID->simParams->m_domIType == SimulationParameters::DomainInputType::WRFOnly) {

      // WRFOnly is the PREEVENTS Fire usage cases...

      wrf_nx = wrf_ptr->fm_nx;
      wrf_ny = wrf_ptr->fm_ny;

      WID->metParams->sensors.resize(wrf_ptr->fm_nx * wrf_ptr->fm_ny);

      for (auto i = 0; i < wrf_ptr->fm_nx * wrf_ptr->fm_ny; i++) {
        // create a new sensor element
        if (!WID->metParams->sensors[i]) {
          WID->metParams->sensors[i] = new Sensor();
        }

        // create one time series for each sensor, for now
        WID->metParams->sensors[i]->TS.resize(1);
        if (!WID->metParams->sensors[i]->TS[0]) {
          WID->metParams->sensors[i]->TS[0] = new TimeSeries();
        }

        // WRF profile data -- sensor blayer flag is 4
        WID->metParams->sensors[i]->TS[0]->site_blayer_flag = 4;
      }

      // Also need to correctly add time stamps
      // example from below:
      //   for (size_t t = 0; t < sensortime_id.size(); t++) {
      //      timestamp.push_back(bt::from_time_t(sensortime[t]));
      //   }

      // initialize our time info...
      QEStime tmp("2022-01-01T00:00");
      sensortime.push_back(tmp);// that's roughly 01/19/22 at 1:00pm Central
      sensortime_id.push_back(0);
      timestamp.push_back(sensortime[0]);

      totalTimeIncrements = WID->simParams->totalTimeIncrements;

      // Here to take care of the first time this is built
      for (auto i = 0; i < wrf_ptr->fm_nx; i++) {
        for (auto j = 0; j < wrf_ptr->fm_ny; j++) {
          int index = i + j * wrf_ptr->fm_nx;
          WID->metParams->sensors[index]->site_coord_flag = 1;
          WID->metParams->sensors[index]->site_xcoord = (i + halo_index_x + 0.5) * dx;
          WID->metParams->sensors[index]->site_ycoord = (j + halo_index_y + 0.5) * dy;

          WID->metParams->sensors[index]->TS[0]->site_wind_dir.resize(wrf_ptr->ht_fmw.size());
          WID->metParams->sensors[index]->TS[0]->site_z_ref.resize(wrf_ptr->ht_fmw.size());
          WID->metParams->sensors[index]->TS[0]->site_U_ref.resize(wrf_ptr->ht_fmw.size());

          WID->metParams->sensors[index]->TS[0]->site_z0 = 0.1;// should get per cell from WRF data...  we do load per atm cell...
          WID->metParams->sensors[index]->TS[0]->site_one_overL = 0.0;

          // for each height in the WRF profile
          for (auto p = 0; p < wrf_ptr->ht_fmw.size(); p++) {
            int id = index + p * wrf_ptr->fm_nx * wrf_ptr->fm_ny;
            WID->metParams->sensors[index]->TS[0]->site_z_ref[p] = wrf_ptr->ht_fmw[p];
            WID->metParams->sensors[index]->TS[0]->site_U_ref[p] = sqrt((wrf_ptr->u0_fmw[id] * wrf_ptr->u0_fmw[id]) + (wrf_ptr->v0_fmw[id] * wrf_ptr->v0_fmw[id]));
            WID->metParams->sensors[index]->TS[0]->site_wind_dir[p] = 180 + (180 / pi) * atan2(wrf_ptr->v0_fmw[id], wrf_ptr->u0_fmw[id]);
          }
        }
      }
      // #endif

      // u0 and v0 are wrf_ptr->fm_nx * wrf_ptr->fm_ny *
      // wrf_ptr->ht_fmw.size()
      //
      // The heights themselves are in the wrf_ptr->ht_fmw array

    }

    else if (wrf_ptr->statData.size() > 0) {
      std::cout << "Size of WRF station/sensor profile data: " << wrf_ptr->statData.size() << std::endl;
      WID->metParams->sensors.resize(wrf_ptr->statData.size());

      for (auto i = 0; i < wrf_ptr->statData.size(); i++) {
        std::cout << "Station " << i << " ("
                  << wrf_ptr->statData[i].xCoord << ", "
                  << wrf_ptr->statData[i].yCoord << ")" << std::endl;

        if (!WID->metParams->sensors[i])
          WID->metParams->sensors[i] = new Sensor();

        WID->metParams->sensors[i]->site_xcoord = wrf_ptr->statData[i].xCoord;
        WID->metParams->sensors[i]->site_ycoord = wrf_ptr->statData[i].yCoord;

        // Need to allocate time series... not fully pulling all WRF
        // time series yet... just first
        WID->metParams->sensors[i]->TS.resize(1);
        // Also need to allocate the space...
        if (!WID->metParams->sensors[i]->TS[0])
          WID->metParams->sensors[i]->TS[0] = new TimeSeries;

        // WRF profile data -- sensor blayer flag is 4
        WID->metParams->sensors[i]->TS[0]->site_blayer_flag = 4;

        // Make sure to set size_z0 to be z0 from WRF cell
        WID->metParams->sensors[i]->TS[0]->site_z0 = wrf_ptr->statData[i].z0;

        //
        // 1 time series for now - how do we deal with this for
        // new time steps???  Need to figure out ASAP.
        //
        // Also need to correctly add time stamps
        // example from below:
        //   for (size_t t = 0; t < sensortime_id.size(); t++) {
        //      timestamp.push_back(bt::from_time_t(sensortime[t]));
        //   }

        for (int t = 0; t < 1; t++) {
          std::cout << "\tTime Series: " << t << std::endl;

          sensortime.push_back(t);
          sensortime_id.push_back(t);
          timestamp.push_back(sensortime[t]);

          int profDataSz = wrf_ptr->statData[i].profiles[t].size();
          WID->metParams->sensors[i]->TS[0]->site_wind_dir.resize(profDataSz);
          WID->metParams->sensors[i]->TS[0]->site_z_ref.resize(profDataSz);
          WID->metParams->sensors[i]->TS[0]->site_U_ref.resize(profDataSz);


          for (size_t p = 0; p < wrf_ptr->statData[i].profiles[t].size(); p++) {
            std::cout << "\t" << wrf_ptr->statData[i].profiles[t][p].zCoord
                      << ", " << wrf_ptr->statData[i].profiles[t][p].ws
                      << ", " << wrf_ptr->statData[i].profiles[t][p].wd << std::endl;

            WID->metParams->sensors[i]->TS[0]->site_z_ref[p] = wrf_ptr->statData[i].profiles[t][p].zCoord;
            WID->metParams->sensors[i]->TS[0]->site_U_ref[p] = wrf_ptr->statData[i].profiles[t][p].ws;
            WID->metParams->sensors[i]->TS[0]->site_wind_dir[p] = wrf_ptr->statData[i].profiles[t][p].wd;
          }
        }
      }
    }
  } else {// no wrf data to deal with...
    /* FM NOTES:
       WARNING this is unfinished.
       + does support halo for QES coord (site coord == 1)
       - does not support halo for UTM coord (site coord == 2)
       - does not support halo for lon/lat coord (site coord == 3)
    */

    // If the sensor file specified in the xml
    if (WID->metParams->sensorName.size() > 0) {
      for (size_t i = 0; i < WID->metParams->sensorName.size(); i++) {
        // Create new sensor object
        WID->metParams->sensors.push_back(new Sensor(WID->metParams->sensorName[i]));
      }
    }

    // If there are more than one timestep
    //if (WID->simParams->totalTimeIncrements > 0) {
    // Loop to include all the timestep for the first sensor
    for (size_t i = 0; i < WID->metParams->sensors[0]->TS.size(); i++) {
      sensortime.push_back(WID->metParams->sensors[0]->TS[i]->time);
      sensortime_id.push_back(i);
    }

    // Loop to include all the unique timesteps of the rest of the sensors
    for (size_t i = 0; i < WID->metParams->sensors.size(); i++) {
      for (size_t j = 0; j < WID->metParams->sensors[i]->TS.size(); j++) {
        size_t count = 0;
        for (size_t k = 0; k < sensortime.size(); k++) {
          if (WID->metParams->sensors[i]->TS[j]->time != sensortime[k]) {
            count += 1;
          }
        }
        // If the timestep is not allready included in the list
        if (count == sensortime.size()) {
          sensortime.push_back(WID->metParams->sensors[i]->TS[j]->time);
          sensortime_id.push_back(sensortime.size() - 1);
        }

        // If the timestep is not allready included in the list
        if (count == sensortime.size()) {
          sensortime.push_back(WID->metParams->sensors[i]->TS[j]->time);
          sensortime_id.push_back(sensortime.size() - 1);
        }
      }
    }

    // Sort the timesteps from low to high (earliest to latest)
    mergeSortTime(sensortime, sensortime_id);

    // adding time stamps
    for (size_t t = 0; t < sensortime_id.size(); t++) {
      timestamp.push_back(sensortime[t]);
    }

    if (WID->simParams->totalTimeIncrements == 0) {
      totalTimeIncrements = timestamp.size();
    } else {
      totalTimeIncrements = WID->simParams->totalTimeIncrements;
    }

    // Adding halo to sensor location (if in QEScoord site_coord_flag==1)
    for (size_t i = 0; i < WID->metParams->sensors.size(); i++) {
      if (WID->metParams->sensors[i]->site_coord_flag == 1) {
        WID->metParams->sensors[i]->site_xcoord += WID->simParams->halo_x;
        WID->metParams->sensors[i]->site_ycoord += WID->simParams->halo_y;
      }
    }
  }

  // /////////////////////////
  // Calculation of z0 domain info MAY need to move to WINDSInputData
  // or somewhere else once we know the domain size
  // /////////////////////////
  z0_domain_u.resize(nx * ny);
  z0_domain_v.resize(nx * ny);
  if (WID->metParams->z0_domain_flag == 0)// Uniform z0 for the whole domain
  {
    for (auto i = 0; i < nx; i++) {
      for (auto j = 0; j < ny; j++) {
        id = i + j * nx;
        z0_domain_u[id] = WID->metParams->sensors[0]->TS[0]->site_z0;
        z0_domain_v[id] = WID->metParams->sensors[0]->TS[0]->site_z0;
      }
    }
  } else {
    for (auto i = 0; i < nx / 2; i++) {
      for (auto j = 0; j < ny; j++) {
        id = i + j * nx;
        z0_domain_u[id] = 0.5;
        z0_domain_v[id] = 0.5;
      }
    }
    for (auto i = nx / 2; i < nx; i++) {
      for (auto j = 0; j < ny; j++) {
        id = i + j * nx;
        z0_domain_u[id] = 0.1;
        z0_domain_v[id] = 0.1;
      }
    }
  }

  z0 = 0.1f;
  if (WID->buildings)
    z0 = WID->buildings->wallRoughness;

  // /////////////////////////
  // Definition of the grid
  // /////////////////////////

  // vertical grid (can be uniform or stretched)
  dz_array.resize(nz - 1, 0.0);
  z.resize(nz - 1);
  z_face.resize(nz - 1);

  if (WID->simParams->verticalStretching == 0) {// Uniform vertical grid
    for (size_t k = 1; k < z.size(); k++) {
      dz_array[k] = dz;
    }
  } else if (WID->simParams->verticalStretching == 1) {// Stretched vertical grid
    for (size_t k = 1; k < z.size(); k++) {
      dz_array[k] = WID->simParams->dz_value[k - 1];// Read in custom dz values and set them to dz_array
    }
  }

  dz_array[0] = dz_array[1];// Value for ghost cell below the surface
  dz = *std::min_element(dz_array.begin(), dz_array.end());// Set dz to minimum value of

  z_face[0] = 0.0;
  z[0] = -0.5 * dz_array[0];
  for (size_t k = 1; k < z.size(); k++) {
    z_face[k] = z_face[k - 1] + dz_array[k];// Location of face centers in z-dir
    z[k] = 0.5 * (z_face[k - 1] + z_face[k]);// Location of cell centers in z-dir
  }


  // horizontal grid (x-direction)
  x.resize(nx - 1);
  for (auto i = 0; i < nx - 1; i++) {
    x[i] = (i + 0.5) * dx;// Location of face centers in x-dir
  }

  // horizontal grid (y-direction)
  y.resize(ny - 1);
  for (auto j = 0; j < ny - 1; j++) {
    y[j] = (j + 0.5) * dy;// Location of face centers in y-dir
  }

  // Resize the canopy-related vectors
  // canopy_atten.resize( numcell_cent, 0.0 );
  // canopy_top.resize( (nx-1)*(ny-1), 0.0 );
  // canopy_top_index.resize( (nx-1)*(ny-1), 0 );
  // canopy_z0.resize( (nx-1)*(ny-1), 0.0 );
  // canopy_ustar.resize( (nx-1)*(ny-1), 0.0 );
  // canopy_d.resize( (nx-1)*(ny-1), 0.0 );


  // Resize the coefficients for use with the solver
  e.resize(numcell_cent, 1.0);
  f.resize(numcell_cent, 1.0);
  g.resize(numcell_cent, 1.0);
  h.resize(numcell_cent, 1.0);
  m.resize(numcell_cent, 1.0);
  n.resize(numcell_cent, 1.0);

  building_volume_frac.resize(numcell_cent, 1.0);
  terrain_volume_frac.resize(numcell_cent, 1.0);
  ni.resize(numcell_cent, 0.0);
  nj.resize(numcell_cent, 0.0);
  nk.resize(numcell_cent, 0.0);

  icellflag.resize(numcell_cent, 1);
  icellflag_initial.resize(numcell_cent, 1);
  icellflag_footprint.resize(numcell_cout_2d, 1);

  ibuilding_flag.resize(numcell_cent, -1);

  mixingLengths.resize(numcell_cent, 0.0);

  terrain.resize(numcell_cout_2d, 0.0);
  terrain_face_id.resize(nx * ny, 1);
  terrain_id.resize((nx - 1) * (ny - 1), 1);

  /////////////////////////////////////////

  // Set the Wind Velocity data elements to be of the correct size
  // Initialize u0,v0,w0,u,v and w to 0.0
  u0.resize(numcell_face, 0.0);
  v0.resize(numcell_face, 0.0);
  w0.resize(numcell_face, 0.0);

  u.resize(numcell_face, 0.0);
  v.resize(numcell_face, 0.0);
  w.resize(numcell_face, 0.0);

  depcvol.resize(numcell_cent, 0.0);

  std::cout << "Memory allocation complete." << std::endl;

  // defining ground solid cells (ghost cells below the surface)
  for (int j = 0; j < ny - 1; j++) {
    for (int i = 0; i < nx - 1; i++) {
      int icell_cent = i + j * (nx - 1);
      icellflag[icell_cent] = 2;
    }
  }

  halo_index_x = (WID->simParams->halo_x / dx);
  // WID->simParams->halo_x = halo_index_x*dx;
  halo_index_y = (WID->simParams->halo_y / dy);
  // WID->simParams->halo_y = halo_index_y*dy;

  int ii, jj, idx;
  if (WID->simParams->DTE_heightField) {
    // ////////////////////////////////
    // Retrieve terrain height field //
    // ////////////////////////////////
    for (int i = 0; i < nx - 2 * halo_index_x - 1; i++) {
      for (int j = 0; j < ny - 2 * halo_index_y - 1; j++) {
        // Gets height of the terrain for each cell
        ii = i + halo_index_x;
        jj = j + halo_index_y;
        idx = ii + jj * (nx - 1);
        terrain[idx] = WID->simParams->DTE_mesh->getHeight(i * dx + dx * 0.5f, j * dy + dy * 0.5f);
        if (terrain[idx] < 0.0) {
          terrain[idx] = 0.0;
        }
        id = ii + jj * nx;
        for (size_t k = 0; k < z.size() - 1; k++) {
          terrain_face_id[id] = k;
          if (terrain[idx] < z_face[k]) {
            break;
          }
        }
      }
    }

    for (int i = halo_index_x; i < nx - halo_index_x - 1; i++) {
      for (int j = 0; j < halo_index_y; j++) {
        id = i + halo_index_y * nx;
        icell_face = i + j * nx;
        terrain_face_id[icell_face] = terrain_face_id[id];
      }

      for (int j = ny - halo_index_y - 1; j < ny; j++) {
        id = i + (ny - halo_index_y - 1) * nx;
        icell_face = i + j * nx;
        terrain_face_id[icell_face] = terrain_face_id[id];
      }
    }

    for (int j = halo_index_y; j < ny - halo_index_y - 1; j++) {
      for (int i = 0; i < halo_index_x; i++) {
        id = halo_index_x + j * nx;
        icell_face = i + j * nx;
        terrain_face_id[icell_face] = terrain_face_id[id];
      }

      for (int i = nx - halo_index_x - 1; i < nx; i++) {
        id = (nx - halo_index_x - 1) + j * nx;
        icell_face = i + j * nx;
        terrain_face_id[icell_face] = terrain_face_id[id];
      }
    }

    for (int i = 0; i < halo_index_x; i++) {
      for (int j = 0; j < halo_index_y; j++) {
        id = halo_index_x + halo_index_y * nx;
        icell_face = i + j * nx;
        terrain_face_id[icell_face] = terrain_face_id[id];
      }

      for (int j = ny - halo_index_y - 1; j < ny; j++) {
        id = halo_index_x + (ny - halo_index_y - 1) * nx;
        icell_face = i + j * nx;
        terrain_face_id[icell_face] = terrain_face_id[id];
      }
    }

    for (int i = nx - halo_index_x - 1; i < nx - 1; i++) {
      for (int j = 0; j < halo_index_y; j++) {
        id = (nx - halo_index_x - 1) + halo_index_y * nx;
        icell_face = i + j * nx;
        terrain_face_id[icell_face] = terrain_face_id[id];
      }

      for (int j = ny - halo_index_y - 1; j < ny - 1; j++) {
        id = (nx - halo_index_x - 1) + (ny - halo_index_y - 2) * nx;
        icell_face = i + j * nx;
        terrain_face_id[icell_face] = terrain_id[id];
      }
    }

    for (int i = 0; i < nx - 2 * halo_index_x - 1; i++) {
      for (int j = 0; j < ny - 2 * halo_index_y - 1; j++) {
        // Gets height of the terrain for each cell
        ii = i + halo_index_x;
        jj = j + halo_index_y;
        idx = ii + jj * (nx - 1);
        for (size_t k = 0; k < z.size() - 1; k++) {
          terrain_id[idx] = k;
          if (terrain[idx] < z[k]) {
            break;
          }
        }
      }
    }

    for (int i = halo_index_x; i < nx - halo_index_x - 1; i++) {
      for (int j = 0; j < halo_index_y; j++) {
        id = i + halo_index_y * (nx - 1);
        icell_cent = i + j * (nx - 1);
        terrain_id[icell_cent] = terrain_id[id];
        terrain[icell_cent] = terrain[id];
      }

      for (int j = ny - halo_index_y - 1; j < ny - 1; j++) {
        id = i + (ny - halo_index_y - 2) * (nx - 1);
        icell_cent = i + j * (nx - 1);
        terrain_id[icell_cent] = terrain_id[id];
        terrain[icell_cent] = terrain[id];
      }
    }

    for (int j = halo_index_y; j < ny - halo_index_y - 1; j++) {
      for (int i = 0; i < halo_index_x; i++) {
        id = halo_index_x + j * (nx - 1);
        icell_cent = i + j * (nx - 1);
        terrain_id[icell_cent] = terrain_id[id];
        terrain[icell_cent] = terrain[id];
      }

      for (int i = nx - halo_index_x - 1; i < nx - 1; i++) {
        id = (nx - halo_index_x - 2) + j * (nx - 1);
        icell_cent = i + j * (nx - 1);
        terrain_id[icell_cent] = terrain_id[id];
        terrain[icell_cent] = terrain[id];
      }
    }

    for (int i = 0; i < halo_index_x; i++) {
      for (int j = 0; j < halo_index_y; j++) {
        id = halo_index_x + halo_index_y * (nx - 1);
        icell_cent = i + j * (nx - 1);
        terrain_id[icell_cent] = terrain_id[id];
        terrain[icell_cent] = terrain[id];
      }

      for (int j = ny - halo_index_y - 1; j < ny - 1; j++) {
        id = halo_index_x + (ny - halo_index_y - 2) * (nx - 1);
        icell_cent = i + j * (nx - 1);
        terrain_id[icell_cent] = terrain_id[id];
        terrain[icell_cent] = terrain[id];
      }
    }

    for (int i = nx - halo_index_x - 1; i < nx - 1; i++) {
      for (int j = 0; j < halo_index_y; j++) {
        id = (nx - halo_index_x - 2) + halo_index_y * (nx - 1);
        icell_cent = i + j * (nx - 1);
        terrain_id[icell_cent] = terrain_id[id];
        terrain[icell_cent] = terrain[id];
      }

      for (int j = ny - halo_index_y - 1; j < ny - 1; j++) {
        id = (nx - halo_index_x - 2) + (ny - halo_index_y - 2) * (nx - 1);
        icell_cent = i + j * (nx - 1);
        terrain_id[icell_cent] = terrain_id[id];
        terrain[icell_cent] = terrain[id];
      }
    }
  }

  //////////////////////////////////////////////////////////////////////////////////
  /////    Create sensor velocity profiles and generate initial velocity field /////
  //////////////////////////////////////////////////////////////////////////////////
  // Calling inputWindProfile function to generate initial velocity
  // field from sensors information (located in Sensor.cpp)

  // Pete could move to input param processing...
  assert(WID->metParams->sensors.size() > 0);// extra
  std::cout << "Sensors have been loaded (total sensors = " << WID->metParams->sensors.size() << ")." << std::endl;

  ////////////////////////////////////////////////////////
  //////              Apply Terrain code             /////
  ///////////////////////////////////////////////////////
  // Handle remaining Terrain processing components here
  ////////////////////////////////////////////////////////


  if (WID->simParams->DTE_heightField) {
    if (WID->simParams->meshTypeFlag == 0 && WID->simParams->readCoefficientsFlag == 0) {
      auto start_stair = std::chrono::high_resolution_clock::now();
      for (int i = 0; i < nx - 1; i++) {
        for (int j = 0; j < ny - 1; j++) {
          // Gets height of the terrain for each cell
          int idx = i + j * (nx - 1);
          for (size_t k = 0; k < z.size() - 1; k++) {
            if (terrain[idx] < z[k + 1]) {
              break;
            }

            // ////////////////////////////////
            // Stair-step (original QUIC)    //
            // ////////////////////////////////
            icell_cent = i + j * (nx - 1) + (k + 1) * (nx - 1) * (ny - 1);
            icellflag[icell_cent] = 2;
          }
        }
      }

      auto finish_stair = std::chrono::high_resolution_clock::now();// Finish recording execution time

      std::chrono::duration<float> elapsed_stair = finish_stair - start_stair;
      std::cout << "Elapsed time for terrain with stair-step: " << elapsed_stair.count() << " s\n";
    }

    if (WID->simParams->meshTypeFlag == 1 && WID->simParams->readCoefficientsFlag == 0) {
      //////////////////////////////////
      //        Cut-cell method       //
      //////////////////////////////////

      auto start_cut = std::chrono::high_resolution_clock::now();

      // Calling calculateCoefficient function to calculate area fraction coefficients for cut-cells
      //WID->simParams->DTE_heightField->setCells(cells, this, WID);
      WID->simParams->DTE_heightField->setCells(this, WID);

      auto finish_cut = std::chrono::high_resolution_clock::now();// Finish recording execution time

      std::chrono::duration<float> elapsed_cut = finish_cut - start_cut;
      std::cout << "Elapsed time for terrain with cut-cell: " << elapsed_cut.count() << " s\n";
    }
  }
  ///////////////////////////////////////////////////////
  //////   END END END of  Apply Terrain code       /////
  ///////////////////////////////////////////////////////


  // WINDS Input Data will have read in the specific types of
  // buildings, canopies, etc... but we need to merge all of that
  // onto a single vector of Building* -- this vector is called
  //
  // allBuildingsVector
  allBuildingsV.clear();// make sure there's nothing on it

  // After Terrain is processed, handle remaining processing of SHP
  // file data

  // SHP processing is done.  Now, consolidate all "buildings" onto
  // the same list...  this includes any canopies and building types
  // that were read in via the XML file...


  // Add all the Building* that were read in from XML to this list
  // too -- could be RectBuilding, PolyBuilding, whatever is derived
  // from Building in the end...
  if (WID->buildings) {
    auto buildingsetup_start = std::chrono::high_resolution_clock::now();// Start recording execution time
    //
    if (WID->buildings->SHPData) {
      std::cout << "Creating buildings from shapefile..." << std::flush;

      // std::vector<Building *> poly_buildings;


      float corner_height, min_height;

      std::vector<float> shpDomainSize(2), minExtent(2);
      WID->buildings->SHPData->getLocalDomain(shpDomainSize);
      WID->buildings->SHPData->getMinExtent(minExtent);

      // printf("\tShapefile Origin = (%.6f,%.6f)\n", minExtent[0], minExtent[1]);

      // If the shapefile is not covering the whole domain or the UTM coordinates
      // of the QES domain is different than shapefile origin
      if (WID->simParams->UTMx != 0.0 && WID->simParams->UTMy != 0.0) {
        minExtent[0] -= (minExtent[0] - WID->simParams->UTMx);
        minExtent[1] -= (minExtent[1] - WID->simParams->UTMy);
      }

      for (size_t pIdx = 0u; pIdx < WID->buildings->SHPData->m_polygons.size(); pIdx++) {

        // convert the global polys to local domain coordinates
        for (auto lIdx = 0u; lIdx < WID->buildings->SHPData->m_polygons[pIdx].size(); lIdx++) {
          WID->buildings->SHPData->m_polygons[pIdx][lIdx].x_poly -= minExtent[0];
          WID->buildings->SHPData->m_polygons[pIdx][lIdx].y_poly -= minExtent[1];
        }

        // Setting base height for buildings if there is a DEM file
        if (WID->simParams->DTE_heightField && WID->simParams->DTE_mesh) {
          // Get base height of every corner of building from terrain height
          min_height = WID->simParams->DTE_mesh->getHeight(WID->buildings->SHPData->m_polygons[pIdx][0].x_poly,
                                                           WID->buildings->SHPData->m_polygons[pIdx][0].y_poly);
          if (min_height < 0) {
            min_height = 0.0;
          }
          for (size_t lIdx = 1u; lIdx < WID->buildings->SHPData->m_polygons[pIdx].size(); lIdx++) {
            corner_height = WID->simParams->DTE_mesh->getHeight(WID->buildings->SHPData->m_polygons[pIdx][lIdx].x_poly,
                                                                WID->buildings->SHPData->m_polygons[pIdx][lIdx].y_poly);

            if (corner_height < min_height && corner_height >= 0.0) {
              min_height = corner_height;
            }
          }
          base_height.push_back(min_height);
        } else {
          base_height.push_back(0.0);
        }

        for (size_t lIdx = 0u; lIdx < WID->buildings->SHPData->m_polygons[pIdx].size(); lIdx++) {
          WID->buildings->SHPData->m_polygons[pIdx][lIdx].x_poly += WID->simParams->halo_x;
          WID->buildings->SHPData->m_polygons[pIdx][lIdx].y_poly += WID->simParams->halo_y;
        }

        // Loop to create each of the polygon buildings read in from the shapefile
        int bId = allBuildingsV.size();

        // allBuildingsV.push_back(new PolyBuilding(WID, this, pIdx));
        allBuildingsV.push_back(new PolyBuilding(WID->buildings->SHPData->m_polygons[pIdx],
                                                 WID->buildings->SHPData->m_features[WID->buildings->shpHeightField][pIdx] * WID->buildings->heightFactor,
                                                 base_height[bId],
                                                 bId));
        building_id.push_back(bId);
        allBuildingsV[bId]->setPolyBuilding(this);
        allBuildingsV[bId]->setCellFlags(WID, this, bId);
        effective_height.push_back(allBuildingsV[bId]->height_eff);
      }
      std::cout << "[done]" << std::endl;
    }
    std::cout << "Consolidating building data..." << std::endl;

    float corner_height, min_height;
    for (size_t i = 0; i < WID->buildings->buildings.size(); i++) {

      allBuildingsV.push_back(WID->buildings->buildings[i]);
      int j = allBuildingsV.size() - 1;
      building_id.push_back(j);

      for (size_t pIdx = 0u; pIdx < allBuildingsV[j]->polygonVertices.size(); pIdx++) {
        allBuildingsV[j]->polygonVertices[pIdx].x_poly += WID->simParams->halo_x;
        allBuildingsV[j]->polygonVertices[pIdx].y_poly += WID->simParams->halo_y;
      }

      // Setting base height for buildings if there is a DEM file
      if (WID->simParams->DTE_heightField && WID->simParams->DTE_mesh) {
        // Get base height of every corner of building from terrain height
        min_height = WID->simParams->DTE_mesh->getHeight(allBuildingsV[j]->polygonVertices[0].x_poly,
                                                         allBuildingsV[j]->polygonVertices[0].y_poly);
        if (min_height < 0) {
          min_height = 0.0;
        }
        for (size_t lIdx = 1; lIdx < allBuildingsV[j]->polygonVertices.size(); lIdx++) {
          corner_height = WID->simParams->DTE_mesh->getHeight(allBuildingsV[j]->polygonVertices[lIdx].x_poly,
                                                              allBuildingsV[j]->polygonVertices[lIdx].y_poly);

          if (corner_height < min_height && corner_height >= 0.0) {
            min_height = corner_height;
          }
        }
        allBuildingsV[j]->base_height = min_height;
      } else {
        // allBuildingsV[j]->base_height = 0.0;
      }
      allBuildingsV[j]->ID = j;
      allBuildingsV[j]->setPolyBuilding(this);
      allBuildingsV[j]->setCellFlags(WID, this, j);
      effective_height.push_back(allBuildingsV[j]->height_eff);
    }


    // We want to sort ALL buildings here...  use the allBuildingsV to
    // do this... (remember some are canopies) so we may need a
    // virtual function in the Building class to get the appropriate
    // data for the sort.
    std::cout << "Sorting buildings by height..." << std::flush;
    mergeSort(effective_height, building_id);
    std::cout << "[done]" << std::endl;

    auto buildingsetup_finish = std::chrono::high_resolution_clock::now();// Finish recording execution time

    std::chrono::duration<float> elapsed_cut = buildingsetup_finish - buildingsetup_start;
    std::cout << "Elapsed time for building setup : " << elapsed_cut.count() << " s\n";
  }

  // Add all the Canopy* to it (they are derived from Building)
  canopy = 0;
  if (WID->canopies) {
    canopy = new Canopy(WID, this);
    canopy->setCanopyElements(WID, this);
  }

  auto wallsetup_start = std::chrono::high_resolution_clock::now();

  std::cout << "Defining Solid Walls..." << std::flush;
  wall = new Wall();
  // Boundary condition for building edges
  wall->defineWalls(this);
  wall->solverCoefficients(this);
  std::cout << "[done]" << std::endl;

  auto wallsetup_finish = std::chrono::high_resolution_clock::now();// Finish recording execution time
  std::chrono::duration<float> elapsed_wall = wallsetup_finish - wallsetup_start;
  std::cout << "Elapsed time for defining walls: " << elapsed_wall.count() << " s\n";

  for (auto id = 0u; id < icellflag.size(); id++) {
    icellflag_initial[id] = icellflag[id];
  }

  /////////////////////////////////////////////////////////
  /////       Read coefficients from a file            ////
  /////////////////////////////////////////////////////////

  if (WID->simParams->readCoefficientsFlag == 1) {

    NetCDFInput *NCDFInput = new NetCDFInput(WID->simParams->coeffFile);

    int ncnx, ncny, ncnz, ncnt;
    std::vector<size_t> start;
    std::vector<size_t> count;

    start = { 0, 0, 0, 0 };
    NCDFInput->getDimensionSize("x", ncnx);
    NCDFInput->getDimensionSize("y", ncny);
    NCDFInput->getDimensionSize("z", ncnz);
    NCDFInput->getDimensionSize("t", ncnt);

    count = { static_cast<unsigned long>(1),
              static_cast<unsigned long>(ncnz - 1),
              static_cast<unsigned long>(ncny - 1),
              static_cast<unsigned long>(ncnx - 1) };


    NCDFInput->getVariableData("icellflag", start, count, icellflag);

    resetICellFlag();

    // Read in solver coefficients
    NCDFInput->getVariableData("e", start, count, e);
    NCDFInput->getVariableData("f", start, count, f);
    NCDFInput->getVariableData("g", start, count, g);
    NCDFInput->getVariableData("h", start, count, h);
    NCDFInput->getVariableData("m", start, count, m);
    NCDFInput->getVariableData("n", start, count, n);
  }

  return;
}


WINDSGeneralData::WINDSGeneralData(const std::string inputFile)
{

  std::cout << "[WINDS Data] \t Loading QES-winds fields " << std::endl;

  // fullname passed to WINDSGeneralData
  input = new NetCDFInput(inputFile);

  // create wall instance for BC
  wall = new Wall();

  // nx,ny - face centered value (consistant with QES-Winds)
  input->getDimensionSize("x_face", nx);
  input->getDimensionSize("y_face", ny);
  // nz - face centered value + bottom ghost (consistant with QES-Winds)
  input->getDimensionSize("z_face", nz);
  // nt - number of time instance in data
  input->getDimensionSize("t", nt);

  numcell_cout = (nx - 1) * (ny - 1) * (nz - 2); /**< Total number of cell-centered values in domain */
  numcell_cout_2d = (nx - 1) * (ny - 1); /**< Total number of horizontal cell-centered values in domain */
  numcell_cent = (nx - 1) * (ny - 1) * (nz - 1); /**< Total number of cell-centered values in domain */
  numcell_face = nx * ny * nz; /**< Total number of face-centered values in domain */

  // get grid information
  x.resize(nx - 1);
  y.resize(ny - 1);
  z.resize(nz - 1);
  z_face.resize(nz - 1);
  dz_array.resize(nz - 1, 0.0);

  input->getVariableData("x", x);
  dx = x[1] - x[0]; /**< Grid resolution in x-direction */

  input->getVariableData("y", y);
  dy = y[1] - y[0]; /**< Grid resolution in x-direction */
  dxy = MIN_S(dx, dy);

  input->getVariableData("z", z);
  // check if dz_array is in the NetCDF file
  NcVar NcVar_dz;
  input->getVariable("dz_array", NcVar_dz);
  if (!NcVar_dz.isNull()) {
    input->getVariableData("dz_array", dz_array);
    dz = *std::min_element(dz_array.begin(), dz_array.end());
  } else {
    dz = z[1] - z[0];
    for (size_t k = 0; k < z.size(); k++) {
      dz_array[k] = dz;
    }
  }

  // check if z_face is in the NetCDF file
  NcVar NcVar_zface;
  input->getVariable("z_face", NcVar_zface);
  if (!NcVar_zface.isNull()) {
    input->getVariableData("z_face", z_face);
  } else {
    z_face[0] = 0.0;
    for (size_t k = 1; k < z.size(); k++) {
      z_face[k] = z_face[k - 1] + dz_array[k]; /**< Location of face centers in z-dir */
    }
  }

  // get time variables
  std::vector<float> t;
  t.resize(nt);
  input->getVariableData("t", t);

  // check if times is in the NetCDF file
  NcVar NcVar_timestamp;
  input->getVariable("timestamp", NcVar_timestamp);
  std::cout << "WGD check 0" << std::endl;
  std::cout << "isNull = " << NcVar_timestamp.isNull() << std::endl;

  if (NcVar_timestamp.isNull()) {
    std::cout << "WGD check 1" << std::endl;
    std::cout << "-----------------------------------------------------------------" << std::endl;
    std::cout << "[WARNING] No timestamp in NetCDF file" << std::endl;
    std::cout << "-----------------------------------------------------------------" << std::endl;
    QEStime tmp("2022-01-01T00:00");
    for (int k = 0; k < nt; k++) {
      // ptime test= from_iso_extended_string(WID->metParams->sensors[i]->TS[t]->timeStamp)
      timestamp.push_back(tmp + t[k]);
    }
  } else {

    std::cout << "WGD check 2" << std::endl;
    for (int k = 0; k < nt; k++) {
      std::cout << "WGD check 2a" << std::endl;
      std::vector<size_t> start_time;
      std::vector<size_t> count_time;
      start_time = { static_cast<unsigned long>(k), 0 };
      count_time = { 1, 19 };

      std::vector<char> timestamp_tmp;
      std::cout << "WGD check 2b" << std::endl;
      std::cout << " start_time = " << start_time.size() << std::endl;

      NcVar_timestamp.getVar(start_time, count_time, &timestamp_tmp[0]);
      std::cout << "WGD check 2c" << std::endl;
      std::string tmp;

      for (int i = 0; i < 19; ++i) {
        tmp[i] = timestamp_tmp[i];
      }
      std::cout << "WGD check 2d" << std::endl;
      QEStime time(tmp);
      std::cout << "read at time " << time << std::endl;
      std::cout << "WGD check 2e" << std::endl;

      timestamp.push_back(time);
      std::cout << "WGD check 2f" << std::endl;
    }
  }
  std::cout << "WGD check 3" << std::endl;
  totalTimeIncrements = nt;

  // netCDF variables
  std::vector<size_t> start;
  std::vector<size_t> count_2d;

  start = { 0, 0 };
  count_2d = { static_cast<unsigned long>(ny - 1),
               static_cast<unsigned long>(nx - 1) };

  // terrain (cell-center)
  terrain.resize((ny - 1) * (nx - 1), 0.0);
  NcVar NcVar_terrain;
  input->getVariable("terrain", NcVar_terrain);
  if (!NcVar_terrain.isNull()) {// => terrain data in QES-Winds file
    input->getVariableData("terrain", start, count_2d, terrain);
  } else {// => no external terrain data provided
    std::cout << "[WINDS Data] \t no terrain data found -> assumed flat" << std::endl;
  }

  // icellflag (see .h for velues)
  icellflag.resize(numcell_cent, -1);
  /// coefficients for SOR solver
  e.resize(numcell_cent, 1.0);
  f.resize(numcell_cent, 1.0);
  g.resize(numcell_cent, 1.0);
  h.resize(numcell_cent, 1.0);
  m.resize(numcell_cent, 1.0);
  n.resize(numcell_cent, 1.0);

  building_volume_frac.resize(numcell_cent, 1.0);
  terrain_volume_frac.resize(numcell_cent, 1.0);
  ni.resize(numcell_cent, 0.0);
  nj.resize(numcell_cent, 0.0);
  nk.resize(numcell_cent, 0.0);

  icellflag.resize(numcell_cent, 1);
  ibuilding_flag.resize(numcell_cent, -1);

  mixingLengths.resize(numcell_cent, 0.0);

  terrain.resize(numcell_cout_2d, 0.0);
  terrain_id.resize(nx * ny, 1);
  z0_domain_u.resize(nx * ny, 0.1);
  z0_domain_v.resize(nx * ny, 0.1);


  // Set the Wind Velocity data elements to be of the correct size
  // Initialize u0,v0,w0,u,v and w to 0.0
  u0.resize(numcell_face, 0.0);
  v0.resize(numcell_face, 0.0);
  w0.resize(numcell_face, 0.0);

  u.resize(numcell_face, 0.0);
  v.resize(numcell_face, 0.0);
  w.resize(numcell_face, 0.0);

  return;
}

void WINDSGeneralData::loadNetCDFData(int stepin)
{

  std::cout << "[WINDS Data] \t loading data at step " << stepin << std::endl;
#if 0
  std::vector<size_t> start_time;
  std::vector<size_t> count_time;
  start_time = { static_cast<unsigned long>(stepin), 0 };
  count_time = { 1, 19 };

  NcVar NcVar_timestamp;
  input->getVariable("timestamp", NcVar_timestamp);

  std::vector<char> timestamp_tmp;
  NcVar_timestamp.getVar(start_time, count_time, &timestamp_tmp[0]);
  std::string tmp;
  for (int i = 0; i < 19; ++i) {
    tmp[i] = timestamp_tmp[i];
  }
  QEStime time(tmp);
  std::cout << "read at time " << time << std::endl;

  timestamp.push_back(time);
#endif
  // netCDF variables
  std::vector<size_t> start;
  std::vector<size_t> count_cc;
  std::vector<size_t> count_fc;

  start = { static_cast<unsigned long>(stepin), 0, 0, 0 };
  count_cc = { 1,
               static_cast<unsigned long>(nz - 1),
               static_cast<unsigned long>(ny - 1),
               static_cast<unsigned long>(nx - 1) };
  count_fc = { 1,
               static_cast<unsigned long>(nz),
               static_cast<unsigned long>(ny),
               static_cast<unsigned long>(nx) };

  // cell-center variables
  // icellflag (see .h for velues)
  input->getVariableData("icellflag", start, count_cc, icellflag);
  /// coefficients for SOR solver
  NcVar NcVar_SORcoeff;
  input->getVariable("e", NcVar_SORcoeff);

  if (!NcVar_SORcoeff.isNull()) {
    input->getVariableData("e", start, count_cc, e);
    input->getVariableData("f", start, count_cc, f);
    input->getVariableData("g", start, count_cc, g);
    input->getVariableData("h", start, count_cc, h);
    input->getVariableData("m", start, count_cc, m);
    input->getVariableData("n", start, count_cc, n);
  } else {
    std::cout << "[WINDS Data] \t no SORcoeff data found -> assumed e,f,g,h,m,n=1" << std::endl;
  }

  // face-center variables
  input->getVariableData("u", start, count_fc, u);
  input->getVariableData("v", start, count_fc, v);
  input->getVariableData("w", start, count_fc, w);

  // clear wall indices container (guarantee entry vector)
  wall_right_indices.clear();
  wall_left_indices.clear();
  wall_above_indices.clear();
  wall_below_indices.clear();
  wall_front_indices.clear();
  wall_back_indices.clear();

  // define new wall indices container for new data
  wall->defineWalls(this);


  return;
}
void WINDSGeneralData::applyWindProfile(const WINDSInputData *WID, int timeIndex, int solveType)
{
  std::cout << "Applying Wind Profile...\n";

  if (WID->simParams->wrfCoupling) {

    std::cout << "Using WRF Coupling..." << std::endl;

    WRFInput *wrf_ptr = WID->simParams->wrfInputData;

    for (auto i = 0; i < wrf_ptr->fm_nx; i++) {
      for (auto j = 0; j < wrf_ptr->fm_ny; j++) {
        int index = i + j * wrf_ptr->fm_nx;
        WID->metParams->sensors[index]->site_coord_flag = 1;
        WID->metParams->sensors[index]->site_xcoord = (i + halo_index_x + 0.5) * dx;
        WID->metParams->sensors[index]->site_ycoord = (j + halo_index_y + 0.5) * dy;

        WID->metParams->sensors[index]->TS[0]->site_wind_dir.resize(wrf_ptr->ht_fmw.size());
        WID->metParams->sensors[index]->TS[0]->site_z_ref.resize(wrf_ptr->ht_fmw.size());
        WID->metParams->sensors[index]->TS[0]->site_U_ref.resize(wrf_ptr->ht_fmw.size());

        WID->metParams->sensors[index]->TS[0]->site_z0 = 0.1;
        WID->metParams->sensors[index]->TS[0]->site_one_overL = 0.0;

        // hack to make time equivalencies
        WID->metParams->sensors[index]->TS[0]->time = "2022-01-01T00:00:00";

        for (auto p = 0; p < wrf_ptr->ht_fmw.size(); p++) {
          int id = index + p * wrf_ptr->fm_nx * wrf_ptr->fm_ny;
          WID->metParams->sensors[index]->TS[0]->site_z_ref[p] = wrf_ptr->ht_fmw[p];
          WID->metParams->sensors[index]->TS[0]->site_U_ref[p] = sqrt(pow(wrf_ptr->u0_fmw[id], 2.0) + pow(wrf_ptr->v0_fmw[id], 2.0));
          WID->metParams->sensors[index]->TS[0]->site_wind_dir[p] = 180 + (180 / pi) * atan2(wrf_ptr->v0_fmw[id], wrf_ptr->u0_fmw[id]);
        }
      }
    }

    // use the time Index of 0 (forcing it) because time series are not worked out with WRF well..
    WID->metParams->sensors[0]->inputWindProfile(WID, this, 0, solveType);
  } else {
    WID->metParams->sensors[0]->inputWindProfile(WID, this, timeIndex, solveType);
  }


  max_velmag = 0.0;
  for (auto i = 0; i < nx; i++) {
    for (auto j = 0; j < ny; j++) {
      int icell_face = i + j * nx + (nz - 2) * nx * ny;
      max_velmag = MAX_S(max_velmag, sqrt(pow(u0[icell_face], 2.0) + pow(v0[icell_face], 2.0)));
    }
  }
  max_velmag *= 1.2;

  return;
}

void WINDSGeneralData::applyParametrizations(const WINDSInputData *WID)
{
  // std::cout << "[Winds] \t applying Parameterization" << std::endl;

  auto start_param = std::chrono::high_resolution_clock::now();// Start recording execution time

  // ///////////////////////////////////////
  // Generic Parameterization Related Stuff
  // ///////////////////////////////////////
  if (canopy) {
    std::cout << "Applying vegetation parameterization...\n";
    canopy->applyCanopyVegetation(this);
  }

  ///////////////////////////////////////////
  //   Upwind Cavity Parameterization     ///
  ///////////////////////////////////////////
  if (WID->simParams->upwindCavityFlag > 0) {
    std::cout << "Applying upwind cavity parameterization...\n";
    for (size_t i = 0; i < allBuildingsV.size(); i++) {
      allBuildingsV[building_id[i]]->upwindCavity(WID, this);
    }
    // std::cout << "Upwind cavity parameterization done...\n";
  }

  //////////////////////////////////////////////////
  //   Far-Wake and Cavity Parameterizations     ///
  //////////////////////////////////////////////////
  if (WID->simParams->wakeFlag > 0) {
    std::cout << "Applying wake behind building parameterization...\n";
    for (size_t i = 0; i < allBuildingsV.size(); i++) {
      allBuildingsV[building_id[i]]->polygonWake(WID, this, building_id[i]);
    }
    // std::cout << "Wake behind building parameterization done...\n";
  }

  ///////////////////////////////////////////
  //   Street Canyon Parameterization     ///
  ///////////////////////////////////////////
  if (WID->simParams->streetCanyonFlag == 1) {
    std::cout << "Applying street canyon parameterization...\n";
    for (size_t i = 0; i < allBuildingsV.size(); i++) {
      allBuildingsV[building_id[i]]->streetCanyon(this);
    }
    // std::cout << "Street canyon parameterization done...\n";
  } else if (WID->simParams->streetCanyonFlag == 2) {
    std::cout << "Applying street canyon parameterization...\n";
    for (size_t i = 0; i < allBuildingsV.size(); i++) {
      allBuildingsV[building_id[i]]->streetCanyonModified(this);
    }
    // std::cout << "Street canyon parameterization done...\n";
  }

  ///////////////////////////////////////////
  //      Sidewall Parameterization       ///
  ///////////////////////////////////////////
  if (WID->simParams->sidewallFlag > 0) {
    std::cout << "Applying sidewall parameterization...\n";
    for (size_t i = 0; i < allBuildingsV.size(); i++) {
      allBuildingsV[building_id[i]]->sideWall(WID, this);
    }
    // std::cout << "Sidewall parameterization done...\n";
  }


  ///////////////////////////////////////////
  //      Rooftop Parameterization        ///
  ///////////////////////////////////////////
  if (WID->simParams->rooftopFlag > 0) {
    std::cout << "Applying rooftop parameterization...\n";
    for (size_t i = 0; i < allBuildingsV.size(); i++) {
      allBuildingsV[building_id[i]]->rooftop(WID, this);
    }
    // std::cout << "Rooftop parameterization done...\n";
  }

  // ///////////////////////////////////////
  // Generic Parameterization Related Stuff
  // ///////////////////////////////////////
  if (canopy) {
    std::cout << "Applying canopy wake parameterization...\n";
    canopy->applyCanopyWake(this);
  }

  ///////////////////////////////////////////
  //         Street Intersection          ///
  ///////////////////////////////////////////
  /*if (WID->simParams->streetCanyonFlag > 0 && WID->simParams->streetIntersectionFlag > 0 && allBuildingsV.size() > 0)
    {
    std::cout << "Applying Blended Region Parameterization...\n";
    allBuildingsV[0]->streetIntersection (WID, this);
    allBuildingsV[0]->poisson (WID, this);
    std::cout << "Blended Region Parameterization done...\n";
    }*/


  /*
   * Calling wallLogBC to read in vectores of indices of the cells that have wall to right/left,
   * wall above/below and wall in front/back and applies the log law boundary condition fix
   * to the cells near Walls
   *
   */
  // wall->wallLogBC (this);

  wall->setVelocityZero(this);

  auto finish_param = std::chrono::high_resolution_clock::now();// Finish recording execution time

  std::chrono::duration<float> elapsed_param = finish_param - start_param;
  std::cout << "Elapsed time for parameterization: " << elapsed_param.count() << " s\n";

  return;
}

void WINDSGeneralData::resetICellFlag()
{
  for (size_t id = 0; id < icellflag.size(); id++) {
    icellflag[id] = icellflag_initial[id];
  }
  return;
}

void WINDSGeneralData::printTimeProgress(int index)
{
  float percentage = (float)(index + 1) / (float)totalTimeIncrements;
  int val = (int)floor(percentage * 100);
  int lpad = (int)(percentage * PBWIDTH);
  int rpad = PBWIDTH - lpad;
  std::cout << "-------------------------------------------------------------------" << std::endl;
  std::cout << "Running time step (" << index + 1 << "/" << totalTimeIncrements << ") at "
            << timestamp[index] << std::endl;
  printf("%3d%% [%.*s%*s]\n", val, lpad, PBSTR, rpad, "");
  fflush(stdout);
  std::cout << "-------------------------------------------------------------------" << std::endl;
}


void WINDSGeneralData::mergeSort(std::vector<float> &effective_height, std::vector<int> &building_id)
{
  // if the size of the array is 1, it is already sorted
  if (building_id.size() == 1) {
    return;
  }

  if (building_id.size() > 1) {
    // make left and right sides of the data
    std::vector<float> effective_height_L, effective_height_R;
    std::vector<int> building_id_L, building_id_R;
    effective_height_L.resize(building_id.size() / 2);
    effective_height_R.resize(building_id.size() - building_id.size() / 2);
    building_id_L.resize(building_id.size() / 2);
    building_id_R.resize(building_id.size() - building_id.size() / 2);

    // copy data from the main data set to the left and right children
    size_t lC = 0, rC = 0;
    for (size_t i = 0; i < building_id.size(); i++) {
      if (i < building_id.size() / 2) {
        effective_height_L[lC] = effective_height[i];
        building_id_L[lC++] = building_id[i];

      } else {
        effective_height_R[rC] = effective_height[i];
        building_id_R[rC++] = building_id[i];
      }
    }
    // recursively sort the children
    mergeSort(effective_height_L, building_id_L);
    mergeSort(effective_height_R, building_id_R);

    // compare the sorted children to place the data into the main array
    lC = rC = 0;
    for (size_t i = 0; i < building_id.size(); i++) {
      if (rC == effective_height_R.size() || (lC != effective_height_L.size() && effective_height_L[lC] < effective_height_R[rC])) {
        effective_height[i] = effective_height_L[lC];
        building_id[i] = building_id_L[lC++];
      } else {
        effective_height[i] = effective_height_R[rC];
        building_id[i] = building_id_R[rC++];
      }
    }
  }

  return;
}

void WINDSGeneralData::mergeSortTime(std::vector<QEStime> &sensortime, std::vector<int> &sensortime_id)
{
  // if the size of the array is 1, it is already sorted
  if (sensortime_id.size() == 1) {
    return;
  }

  if (sensortime_id.size() > 1) {
    // make left and right sides of the data
    std::vector<QEStime> sensortime_L, sensortime_R;
    std::vector<int> sensortime_id_L, sensortime_id_R;
    sensortime_L.resize(sensortime_id.size() / 2);
    sensortime_R.resize(sensortime_id.size() - sensortime_id.size() / 2);
    sensortime_id_L.resize(sensortime_id.size() / 2);
    sensortime_id_R.resize(sensortime_id.size() - sensortime_id.size() / 2);

    // copy data from the main data set to the left and right children
    size_t lC = 0, rC = 0;
    for (size_t i = 0; i < sensortime_id.size(); i++) {
      if (i < sensortime_id.size() / 2) {
        sensortime_L[lC] = sensortime[i];
        sensortime_id_L[lC++] = sensortime_id[i];

      } else {
        sensortime_R[rC] = sensortime[i];
        sensortime_id_R[rC++] = sensortime_id[i];
      }
    }

    // recursively sort the children
    mergeSortTime(sensortime_L, sensortime_id_L);
    mergeSortTime(sensortime_R, sensortime_id_R);

    // compare the sorted children to place the data into the main array
    lC = rC = 0;
    for (size_t i = 0; i < sensortime_id.size(); i++) {
      if (rC == sensortime_R.size() || (lC != sensortime_L.size() && sensortime_L[lC] < sensortime_R[rC])) {
        sensortime[i] = sensortime_L[lC];
        sensortime_id[i] = sensortime_id_L[lC++];
      } else {
        sensortime[i] = sensortime_R[rC];
        sensortime_id[i] = sensortime_id_R[rC++];
      }
    }
  }

  return;
}


float WINDSGeneralData::canopyBisection(float ustar, float z0, float canopy_top, float canopy_atten, float vk, float psi_m)
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
