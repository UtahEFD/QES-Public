#include "URBGeneralData.h"

URBGeneralData::URBGeneralData(const URBInputData* UID, bool calcMixLength)
    : m_calcMixingLength( calcMixLength )
{
   if ( UID->simParams->upwindCavityFlag == 1)
   {
      lengthf_coeff = 2.0;
   }
   else
   {
      lengthf_coeff = 1.5;
    }

    // Determines how wakes behind buildings are calculated
    if ( UID->simParams->wakeFlag > 1)
    {
        cavity_factor = 1.1;
        wake_factor = 0.1;
    }
    else
    {
        cavity_factor = 1.0;
        wake_factor = 0.0;
    }

    // converting the domain rotation to radians from degrees -- input
    // assumes degrees
    theta = (UID->simParams->domainRotation * pi / 180.0);

    // Pull Domain Size information from the UrbInputData structure --
    // this is either read in from the XML files and/or potentially
    // calculated based on the geographic data that was loaded
    // (i.e. DEM files). It is important to get this data from the
    // main input structure.
    //
    // This is done to make reference to nx, ny and nz easier in this function
    Vector3<int> domainInfo;
    domainInfo = *(UID->simParams->domain);
    nx = domainInfo[0];
    ny = domainInfo[1];
    nz = domainInfo[2];

    // Modify the domain size to fit the Staggered Grid used in the solver
    nx += 1;        /// +1 for Staggered grid
    ny += 1;        /// +1 for Staggered grid
    nz += 2;        /// +2 for staggered grid and ghost cell

    Vector3<float> gridInfo;
    gridInfo = *(UID->simParams->grid);
    dx = gridInfo[0];		/**< Grid resolution in x-direction */
    dy = gridInfo[1];		/**< Grid resolution in y-direction */
    dz = gridInfo[2];		/**< Grid resolution in z-direction */
    dxy = MIN_S(dx, dy);

    numcell_cout    = (nx-1)*(ny-1)*(nz-2);        /**< Total number of cell-centered values in domain */
    numcell_cout_2d = (nx-1)*(ny-1);               /**< Total number of horizontal cell-centered values in domain */
    numcell_cent    = (nx-1)*(ny-1)*(nz-1);        /**< Total number of cell-centered values in domain */
    numcell_face    = nx*ny*nz;                    /**< Total number of face-centered values in domain */


    // where should this really go?
    //
    // Need to now take all WRF station data and convert to
    // sensors
    if (UID->simParams->wrfInputData) {

        WRFInput *wrf_ptr = UID->simParams->wrfInputData;

        std::cout << "Size of stat data: " << wrf_ptr->statData.size() << std::endl;
        UID->metParams->sensors.resize( wrf_ptr->statData.size() );

        for (int i=0; i<wrf_ptr->statData.size(); i++) {
            std::cout << "Station " << i << " ("
                      << wrf_ptr->statData[i].xCoord << ", "
                      << wrf_ptr->statData[i].yCoord << ")" << std::endl;

            if (!UID->metParams->sensors[i])
                UID->metParams->sensors[i] = new Sensor();

            UID->metParams->sensors[i]->site_xcoord = wrf_ptr->statData[i].xCoord;
            UID->metParams->sensors[i]->site_ycoord = wrf_ptr->statData[i].yCoord;

            // WRF profile data -- sensor blayer flag is 4
            UID->metParams->sensors[i]->site_blayer_flag = 4;

            // Make sure to set size_z0 to be z0 from WRF cell
            UID->metParams->sensors[i]->site_z0 = wrf_ptr->statData[i].z0;

            //
            // 1 time series for now - how do we deal with this for
            // new time steps???  Need to figure out ASAP.
            //
            for (int t=0; t<1; t++) {
                std::cout << "\tTime Series: " << t << std::endl;

                int profDataSz = wrf_ptr->statData[i].profiles[t].size();
                UID->metParams->sensors[i]->site_wind_dir.resize( profDataSz );
                UID->metParams->sensors[i]->site_z_ref.resize( profDataSz );
                UID->metParams->sensors[i]->site_U_ref.resize( profDataSz );


                for (int p=0; p<wrf_ptr->statData[i].profiles[t].size(); p++) {
                    std::cout << "\t" << wrf_ptr->statData[i].profiles[t][p].zCoord
                              << ", " << wrf_ptr->statData[i].profiles[t][p].ws
                              << ", " << wrf_ptr->statData[i].profiles[t][p].wd << std::endl;

                    UID->metParams->sensors[i]->site_z_ref[p] = wrf_ptr->statData[i].profiles[t][p].zCoord;
                    UID->metParams->sensors[i]->site_U_ref[p] = wrf_ptr->statData[i].profiles[t][p].ws;
                    UID->metParams->sensors[i]->site_wind_dir[p] = wrf_ptr->statData[i].profiles[t][p].wd;

                }
            }
        }

    }

    // /////////////////////////
    // Calculation of z0 domain info MAY need to move to UrbInputData
    // or somewhere else once we know the domain size
    // /////////////////////////
    z0_domain_u.resize( nx*ny );
    z0_domain_v.resize( nx*ny );
    if (UID->metParams->z0_domain_flag == 0)      // Uniform z0 for the whole domain
    {
        for (auto i=0; i<nx; i++)
        {
            for (auto j=0; j<ny; j++)
            {
                id = i+j*nx;
                z0_domain_u[id] = UID->metParams->sensors[0]->site_z0;
                z0_domain_v[id] = UID->metParams->sensors[0]->site_z0;
            }
        }
    }
    else
    {
        for (auto i=0; i<nx/2; i++)
        {
            for (auto j=0; j<ny; j++)
            {
                id = i+j*nx;
                z0_domain_u[id] = 0.5;
                z0_domain_v[id] = 0.5;
            }
        }
        for (auto i=nx/2; i<nx; i++)
        {
            for (auto j=0; j<ny; j++)
            {
                id = i+j*nx;
                z0_domain_u[id] = 0.1;
                z0_domain_v[id] = 0.1;
            }
        }
    }

    z0 = 0.1f;
    if (UID->buildings)
        z0 = UID->buildings->wallRoughness;

    dz_array.resize( nz-1, 0.0 );
    z.resize( nz-1 );
    z_face.resize( nz-1);

    if (UID->simParams->verticalStretching == 0)    // Uniform vertical grid
    {
        for (auto k=1; k<z.size(); k++)
        {
            dz_array[k] = dz;
        }
    }
    else if (UID->simParams->verticalStretching == 1)     // Stretched vertical grid
    {
        for (auto k=1; k<z.size(); k++)
        {
            dz_array[k] = UID->simParams->dz_value[k-1];      // Read in custom dz values and set them to dz_array
        }
    }

    dz_array[0] = dz_array[1];                  // Value for ghost cell below the surface
    dz = *std::min_element(dz_array.begin() , dz_array.end());     // Set dz to minimum value of

    z[0] = -0.5*dz_array[0];
    z_face[0] = 0.0;
    for (auto k=1; k<z.size(); k++)
    {
        z[k] = z[k-1] + dz_array[k];     /**< Location of face centers in z-dir */
        z_face[k] = z_face[k-1] + dz_array[k];
    }

    x.resize( nx-1 );
    for (size_t i=0; i<nx-1; i++)
    {
        x[i] = (i+0.5)*dx;          /**< Location of face centers in x-dir */
    }

    y.resize( ny-1 );
    for (auto j=0; j<ny-1; j++)
    {
        y[j] = (j+0.5)*dy;          /**< Location of face centers in y-dir */
    }

    // Resize the canopy-related vectors
    canopy_atten.resize( numcell_cent, 0.0 );
    canopy_top.resize( (nx-1)*(ny-1), 0.0 );
    canopy_top_index.resize( (nx-1)*(ny-1), 0 );
    canopy_z0.resize( (nx-1)*(ny-1), 0.0 );
    canopy_ustar.resize( (nx-1)*(ny-1), 0.0 );
    canopy_d.resize( (nx-1)*(ny-1), 0.0 );


    // Resize the coefficients for use with the solver e.resize( numcell_cent, 1.0 );
    e.resize( numcell_cent, 1.0 );
    f.resize( numcell_cent, 1.0 );
    g.resize( numcell_cent, 1.0 );
    h.resize( numcell_cent, 1.0 );
    m.resize( numcell_cent, 1.0 );
    n.resize( numcell_cent, 1.0 );

    building_volume_frac.resize( numcell_cent, 1.0 );
    terrain_volume_frac.resize( numcell_cent, 1.0 );

    icellflag.resize( numcell_cent, 1 );
    ibuilding_flag.resize ( numcell_cent, -1 );

    if (m_calcMixingLength)
        mixingLengths.resize( numcell_cent, 0.0 );

    terrain.resize( numcell_cout_2d, 0.0 );
    terrain_id.resize( nx*ny, 1 );

    /////////////////////////////////////////

    // Set the Wind Velocity data elements to be of the correct size
    // Initialize u0,v0,w0,u,v and w to 0.0
    u0.resize( numcell_face, 0.0 );
    v0.resize( numcell_face, 0.0 );
    w0.resize( numcell_face, 0.0 );

    u.resize( numcell_face, 0.0 );
    v.resize( numcell_face, 0.0 );
    w.resize( numcell_face, 0.0 );

    std::cout << "Memory allocation complete." << std::endl;

    /// defining ground solid cells (ghost cells below the surface)
    for (int j = 0; j < ny-1; j++)
    {
        for (int i = 0; i < nx-1; i++)
        {
            int icell_cent = i + j*(nx-1);
            icellflag[icell_cent] = 2;
        }
    }

    int halo_index_x = (UID->simParams->halo_x/dx);
    UID->simParams->halo_x = halo_index_x*dx;
    int halo_index_y = (UID->simParams->halo_y/dy);
    UID->simParams->halo_y = halo_index_y*dy;

    //////////////////////////////////////////////////////////////////////////////////
    /////    Create sensor velocity profiles and generate initial velocity field /////
    //////////////////////////////////////////////////////////////////////////////////
    // Calling inputWindProfile function to generate initial velocity
    // field from sensors information (located in Sensor.cpp)

    // Pete could move to input param processing...
    assert( UID->metParams->sensors.size() > 0 );  // extra
    // check
    // to
    // be safe
    // Guaranteed to always have at least 1 sensor!
    // Pete thinks inputWindProfile should be a function of MetParams
    // so it would have access to all the sensors naturally.
    // Make this change later.
    //    UID->metParams->inputWindProfile(UID, this);
    UID->metParams->sensors[0]->inputWindProfile(UID, this);

    std::cout << "Sensors have been loaded (total sensors = " << UID->metParams->sensors.size() << ")." << std::endl;

    max_velmag = 0.0;
    for (auto i=0; i<nx; i++)
    {
        for (auto j=0; j<ny; j++)
        {
            int icell_face = i+j*nx+(nz-2)*nx*ny;
            max_velmag = MAX_S(max_velmag, sqrt(pow(u0[icell_face],2.0)+pow(v0[icell_face],2.0)));
        }
    }
    max_velmag *= 1.2;


    ////////////////////////////////////////////////////////
    //////              Apply Terrain code             /////
    ///////////////////////////////////////////////////////
    // Handle remaining Terrain processing components here
    ////////////////////////////////////////////////////////

    if (UID->simParams->DTE_heightField)
    {
        // ////////////////////////////////
        // Retrieve terrain height field //
        // ////////////////////////////////
        for (int i = 0; i < nx-1; i++)
        {
            for (int j = 0; j < ny-1; j++)
            {
                // Gets height of the terrain for each cell
                int idx = i + j*(nx-1);
                terrain[idx] = UID->simParams->DTE_mesh->getHeight(i * dx + dx * 0.5f, j * dy + dy * 0.5f);
                if (terrain[idx] < 0.0)
                {
                    terrain[idx] = 0.0;
                }
                id = i+j*nx;
                for (auto k=0; k<z.size()-1; k++)
                {
                    terrain_id[id] = k+1;
                    if (terrain[idx] < z[k+1])
                    {
                        break;
                    }
                    if (UID->simParams->meshTypeFlag == 0)
                    {
                        // ////////////////////////////////
                        // Stair-step (original QUIC)    //
                        // ////////////////////////////////
                        int ii = i+UID->simParams->halo_x/dx;
                        int jj = j+UID->simParams->halo_y/dy;
                        int icell_cent = ii+jj*(nx-1)+(k+1)*(nx-1)*(ny-1);
                        icellflag[icell_cent] = 2;
                    }
                }
            }
        }

        if (UID->simParams->meshTypeFlag == 1)
        {
            //////////////////////////////////
            //        Cut-cell method       //
            //////////////////////////////////

            // Calling calculateCoefficient function to calculate area fraction coefficients for cut-cells
            cut_cell.calculateCoefficient(cells, UID->simParams->DTE_heightField, nx, ny, nz, dx, dy, dz_array, n, m, f, e, h, g, pi, icellflag,
                                          terrain_volume_frac, z_face, UID->simParams->halo_x, UID->simParams->halo_y);
        }
    }
    ///////////////////////////////////////////////////////
    //////   END END END of  Apply Terrain code       /////
    ///////////////////////////////////////////////////////


   // Urb Input Data will have read in the specific types of
   // buildings, canopies, etc... but we need to merge all of that
   // onto a single vector of Building* -- this vector is called
   //
   // allBuildingsVector
   allBuildingsV.clear();  // make sure there's nothing on it

   // After Terrain is processed, handle remaining processing of SHP
   // file data

   if (UID->simParams->SHPData)
   {
      auto buildingsetup = std::chrono::high_resolution_clock::now(); // Start recording execution time

      std::vector<Building*> poly_buildings;


      float corner_height, min_height;

      std::vector<float> shpDomainSize(2), minExtent(2);
      UID->simParams->SHPData->getLocalDomain( shpDomainSize );
      UID->simParams->SHPData->getMinExtent( minExtent );

      // float domainOffset[2] = { 0, 0 };
      for (auto pIdx = 0; pIdx<UID->simParams->shpPolygons.size(); pIdx++)
      {
         // convert the global polys to local domain coordinates
         for (auto lIdx=0; lIdx<UID->simParams->shpPolygons[pIdx].size(); lIdx++)
         {
            UID->simParams->shpPolygons[pIdx][lIdx].x_poly -= minExtent[0] ;
            UID->simParams->shpPolygons[pIdx][lIdx].y_poly -= minExtent[1] ;
         }
      }

      // Setting base height for buildings if there is a DEM file
      if (UID->simParams->DTE_heightField && UID->simParams->DTE_mesh)
      {
          for (auto pIdx = 0; pIdx < UID->simParams->shpPolygons.size(); pIdx++)
          {
              // Get base height of every corner of building from terrain height
              min_height = UID->simParams->DTE_mesh->getHeight(UID->simParams->shpPolygons[pIdx][0].x_poly,
                                                               UID->simParams->shpPolygons[pIdx][0].y_poly);
              if (min_height < 0)
              {
                  min_height = 0.0;
              }
              for (auto lIdx = 1; lIdx < UID->simParams->shpPolygons[pIdx].size(); lIdx++)
              {
                  corner_height = UID->simParams->DTE_mesh->getHeight(UID->simParams->shpPolygons[pIdx][lIdx].x_poly,
                                                                      UID->simParams->shpPolygons[pIdx][lIdx].y_poly);

                  if (corner_height < min_height && corner_height >= 0.0)
                  {
                      min_height = corner_height;
                  }
              }
              base_height.push_back(min_height);
          }
      }
      else
      {
          for (auto pIdx = 0; pIdx < UID->simParams->shpPolygons.size(); pIdx++)
          {
              base_height.push_back(0.0);
          }
      }

      for (auto pIdx = 0; pIdx < UID->simParams->shpPolygons.size(); pIdx++)
      {
          for (auto lIdx=0; lIdx < UID->simParams->shpPolygons[pIdx].size(); lIdx++)
          {
              UID->simParams->shpPolygons[pIdx][lIdx].x_poly += UID->simParams->halo_x;
              UID->simParams->shpPolygons[pIdx][lIdx].y_poly += UID->simParams->halo_y;
          }
      }

        std::cout << "Creating buildings from shapefile...\n";
        // Loop to create each of the polygon buildings read in from the shapefile
        for (auto pIdx = 0; pIdx < UID->simParams->shpPolygons.size(); pIdx++)
        {
            allBuildingsV.push_back (new PolyBuilding (UID, this, pIdx));
            building_id.push_back(allBuildingsV.size()-1);
            allBuildingsV[pIdx]->setPolyBuilding(this);
            allBuildingsV[pIdx]->setCellFlags(UID, this, pIdx);
            effective_height.push_back (allBuildingsV[pIdx]->height_eff);
        }
        std::cout << "Buildings created from shapefile...\n";
   }


    // SHP processing is done.  Now, consolidate all "buildings" onto
    // the same list...  this includes any canopies and building types
    // that were read in via the XML file...

    // Add all the Canopy* to it (they are derived from Building)
    if ( UID->canopies )
    {
      for (int i = 0; i < UID->canopies->canopies.size(); i++)
      {
         allBuildingsV.push_back( UID->canopies->canopies[i] );
         effective_height.push_back(allBuildingsV[i]->height_eff);
         building_id.push_back(allBuildingsV.size()-1);
      }
    }


   // Add all the Building* that were read in from XML to this list
   // too -- could be RectBuilding, PolyBuilding, whatever is derived
   // from Building in the end...
   if ( UID->buildings )
   {
      for (int i = 0; i < UID->buildings->buildings.size(); i++)
      {
          allBuildingsV.push_back( UID->buildings->buildings[i] );
          int j = allBuildingsV.size()-1;
          building_id.push_back( j );
          allBuildingsV[j]->setPolyBuilding(this);
          allBuildingsV[j]->setCellFlags(UID, this, j);
          effective_height.push_back(allBuildingsV[i]->height_eff);
      }
    }

    // We want to sort ALL buildings here...  use the allBuildingsV to
    // do this... (remember some are canopies) so we may need a
    // virtual function in the Building class to get the appropriate
    // data for the sort.
    mergeSort( effective_height, allBuildingsV, building_id );

    // ///////////////////////////////////////
    // ///////////////////////////////////////

    wall = new Wall();

    std::cout << "Defining Solid Walls...\n";
    // Boundary condition for building edges
    wall->defineWalls(this);
    std::cout << "Walls Defined...\n";

    wall->solverCoefficients (this);

    // ///////////////////////////////////////
    // Generic Parameterization Related Stuff
    // ///////////////////////////////////////
    for (int i = 0; i < allBuildingsV.size(); i++)
    {
        // for now this does the canopy stuff for us
        std::cout << "Applying canopy vegetation parameterization...\n";
        allBuildingsV[building_id[i]]->canopyVegetation(this);
        std::cout << "Canopy vegetation parameterization done...\n";
    }

    ///////////////////////////////////////////
    //   Upwind Cavity Parameterization     ///
    ///////////////////////////////////////////
    if (UID->simParams->upwindCavityFlag > 0)
    {
        std::cout << "Applying upwind cavity parameterization...\n";
        for (int i = 0; i < allBuildingsV.size(); i++)
        {
            allBuildingsV[building_id[i]]->upwindCavity(UID, this);
        }
        std::cout << "Upwind cavity parameterization done...\n";
    }

    //////////////////////////////////////////////////
    //   Far-Wake and Cavity Parameterizations     ///
    //////////////////////////////////////////////////
    if (UID->simParams->wakeFlag > 0)
    {
        std::cout << "Applying wake behind building parameterization...\n";
        for (int i = 0; i < allBuildingsV.size(); i++)
        {
            allBuildingsV[building_id[i]]->polygonWake(UID, this, building_id[i]);
        }
        std::cout << "Wake behind building parameterization done...\n";
    }

    ///////////////////////////////////////////
    //   Street Canyon Parameterization     ///
    ///////////////////////////////////////////
    if (UID->simParams->streetCanyonFlag > 0)
    {
        std::cout << "Applying street canyon parameterization...\n";
        for (int i = 0; i < allBuildingsV.size(); i++)
        {
            allBuildingsV[building_id[i]]->streetCanyon(this);
        }
        std::cout << "Street canyon parameterization done...\n";
    }

    ///////////////////////////////////////////
    //      Sidewall Parameterization       ///
    ///////////////////////////////////////////
    if (UID->simParams->sidewallFlag > 0)
    {
        std::cout << "Applying sidewall parameterization...\n";
        for (int i = 0; i < allBuildingsV.size(); i++)
        {
            allBuildingsV[building_id[i]]->sideWall(UID, this);
        }
        std::cout << "Sidewall parameterization done...\n";
    }


    ///////////////////////////////////////////
    //      Rooftop Parameterization        ///
    ///////////////////////////////////////////
    if (UID->simParams->rooftopFlag > 0)
    {
        std::cout << "Applying rooftop parameterization...\n";
        for (int i = 0; i < allBuildingsV.size(); i++)
        {
            allBuildingsV[building_id[i]]->rooftop (UID, this);
        }
        std::cout << "Rooftop parameterization done...\n";
    }

    ///////////////////////////////////////////
    //         Street Intersection          ///
    ///////////////////////////////////////////
    /*if (UID->simParams->streetCanyonFlag > 0 && UID->simParams->streetIntersectionFlag > 0 && allBuildingsV.size() > 0)
    {
      std::cout << "Applying Blended Region Parameterization...\n";
      allBuildingsV[0]->streetIntersection (UID, this);
      allBuildingsV[0]->poisson (UID, this);
      std::cout << "Blended Region Parameterization done...\n";
    }*/


    /*
     * Calling wallLogBC to read in vectores of indices of the cells that have wall to right/left,
     * wall above/below and wall in front/back and applies the log law boundary condition fix
     * to the cells near Walls
     *
     */
     //wall->wallLogBC (this);

    wall->setVelocityZero (this);

    /*******Add raytrace code here********/
    if (m_calcMixingLength){
        std::cout << "Computing mixing length scales..." << std::endl;
        auto mlStartTime = std::chrono::high_resolution_clock::now();
        UID->simParams->DTE_mesh->calculateMixingLength(nx, ny, nz, dx, dy, dz, icellflag, mixingLengths);
        auto mlEndTime = std::chrono::high_resolution_clock::now();

        std::chrono::duration<double> mlElapsed = mlEndTime - mlStartTime;
        std::cout << "\telapsed time: " << mlElapsed.count() << " s\n";
    }

}


void URBGeneralData::mergeSort( std::vector<float> &effective_height, std::vector<Building*> allBuildingsV, std::vector<int> &building_id)
{
   //if the size of the array is 1, it is already sorted
   if ( allBuildingsV.size() == 1)
   {
      return;
   }

   if ( allBuildingsV.size() > 1)
   {
      //make left and right sides of the data
      std::vector<float> effective_height_L, effective_height_R;
      std::vector<int> building_id_L, building_id_R;
      std::vector<Building*> allBuildingsV_L, allBuildingsV_R;
      effective_height_L.resize(allBuildingsV.size() / 2);
      effective_height_R.resize(allBuildingsV.size() - allBuildingsV.size() / 2);
      building_id_L.resize(allBuildingsV.size() / 2);
      building_id_R.resize(allBuildingsV.size() - allBuildingsV.size()/2);
      allBuildingsV_L.resize(allBuildingsV.size() / 2);
      allBuildingsV_R.resize(allBuildingsV.size() - allBuildingsV.size() / 2);

      //copy data from the main data set to the left and right children
      int lC = 0, rC = 0;
      for (auto i = 0; i < allBuildingsV.size(); i++)
      {
         if (i < allBuildingsV.size() / 2)
         {
            effective_height_L[lC] = effective_height[i];
            allBuildingsV_L[lC] = allBuildingsV[i];
            building_id_L[lC++] = building_id[i];

         }
         else
         {
            effective_height_R[rC] = effective_height[i];
            allBuildingsV_R[rC] = allBuildingsV[i];
            building_id_R[rC++] = building_id[i];

         }
      }
      //recursively sort the children
      mergeSort( effective_height_L, allBuildingsV_L, building_id_L );
      mergeSort( effective_height_R, allBuildingsV_R, building_id_R );

      //compare the sorted children to place the data into the main array
      lC = rC = 0;
      for (unsigned int i = 0; i < allBuildingsV.size(); i++)
      {
         if (rC == effective_height_R.size() || ( lC != effective_height_L.size() &&
                                                  effective_height_L[lC] < effective_height_R[rC]))
         {
            effective_height[i] = effective_height_L[lC];
            building_id[i] = building_id_L[lC++];
         }
         else
         {
            effective_height[i] = effective_height_R[rC];
            building_id[i] = building_id_R[rC++];
         }
      }
   }

   return;
}




float URBGeneralData::canopyBisection(float ustar, float z0, float canopy_top, float canopy_atten, float vk, float psi_m)
{
    int iter;
    float  uhc, d, d1, d2;
    float tol, fnew, fi;

    tol = z0/100;
    fnew = tol*10;

    d1 = z0;
    d2 = canopy_top;
    d = (d1+d2)/2;

    uhc = (ustar/vk)*(log((canopy_top-d1)/z0)+psi_m);
    fi = ((canopy_atten*uhc*vk)/ustar)-canopy_top/(canopy_top-d1);

    if (canopy_atten > 0)
    {
        iter = 0;
        while (iter < 200 && abs(fnew) > tol && d < canopy_top && d > z0)
        {
            iter += 1;
            d = (d1+d2)/2;
            uhc = (ustar/vk)*(log((canopy_top-d)/z0)+psi_m);
            fnew = ((canopy_atten*uhc*vk)/ustar) - canopy_top/(canopy_top-d);
            if(fnew*fi>0)
            {
                d1 = d;
            }
            else if(fnew*fi<0)
            {
                d2 = d;
            }
        }
        if (d > canopy_top)
        {
            d = 10000;
        }

    }
    else
    {
        d = 0.99*canopy_top;
    }

    return d;
}



URBGeneralData::URBGeneralData()
    : m_calcMixingLength( false )
{
}

URBGeneralData::~URBGeneralData()
{
}
