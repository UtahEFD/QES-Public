#include "URBGeneralData.h"
#include "URBInputData.h"

URBGeneralData::URBGeneralData(const URBInputData* UID)
{
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

    // /////////////////////////
    // Calculation of z0 domain info MAY need to move to UrbInputData
    // or somewhere else once we know the domain size
    // /////////////////////////
    z0_domain.resize( nx*ny );
    if (UID->metParams->z0_domain_flag == 0)      // Uniform z0 for the whole domain
    {
        for (auto i=0; i<nx; i++)
        {
            for (auto j=0; j<ny; j++)
            {
                id = i+j*nx;
                z0_domain[id] = UID->metParams->sensors[0]->site_z0;
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
                z0_domain[id] = 0.5;
            }
        }
        for (auto i=nx/2; i<nx; i++)
        {
            for (auto j=0; j<ny; j++)
            {
                id = i+j*nx;
                z0_domain[id] = 0.1;
            }
        }
    }

    

    // Urb Input Data will have read in the specific types of
    // buildings, canoopies, etc... but we need to merge all of that
    // onto a single vector of Building* -- this vector is called
    //
    // allBuildingsVector
    allBuildingsV.clear();  // make sure there's nothing on it
    
    // Add all the Canopy* to it (they are derived from Building)
    for (int i = 0; i < UID->canopies->canopies.size(); i++)
    {
        allBuildingsV.push_back( UID->canopies->canopies[i] ); 
    }    

    // Add all the Building* that were read in from XML to this list
    // too -- could be RectBuilding, PolyBuilding, whatever is derived
    // from Building in the end...
    for (int i = 0; i < UID->buildings->buildings.size(); i++)
    {
        allBuildingsV.push_back( UID->buildings->buildings[i] );
    }

    // !!!!!! Pete ---- Make sure polybuildings from SHP file get on
    // !!!!!! this list too!!!!
    
    // At this point, the allBuildingsV will be complete and ready for
    // use below... parameterizations, etc...
    
    z0 = 0.1f
    if (UID->buildings)
        z0 = UID->buildings->wallRoughness;

    dz_array.resize( nz-1, 0.0 );
    z.resize( nz-1 );

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
    for (auto k=1; k<z.size(); k++)
    {
      z[k] = z[k-1] + dz_array[k];     /**< Location of face centers in z-dir */
    }

    z_out.resize( nz-2 );
    for (size_t k=1; k<z.size(); k++)
    {
      z_out[k-1] = (float)z[k];    /**< Location of face centers in z-dir */
    }

    x.resize( nx-1 );
    x_out.resize( nx-1 );
    for (size_t i=0; i<x.size(); i++)
    {
      x_out[i] = (i+0.5)*dx;          /**< Location of face centers in x-dir */
      x[i] = (float)x_out[i];
    }

    y.resize( ny-1 );
    y_out.resize( ny-1 );
    for (auto j=0; j<ny-1; j++)
    {
      y_out[j] = (j+0.5)*dy;          /**< Location of face centers in y-dir */
      y[j] = (float)y_out[j];
    }
    
    
    // Resize the coefficients for use with the solver e.resize( numcell_cent, 1.0 );
    e.resize( numcell_cent, 1.0 );
    f.resize( numcell_cent, 1.0 );
    g.resize( numcell_cent, 1.0 );

    h.resize( numcell_cent, 1.0 );
    m.resize( numcell_cent, 1.0 );
    n.resize( numcell_cent, 1.0 );

    icellflag.resize( numcell_cent, 1 );    

    // /////////////////////////////////////////
    // Output related data --- should be part of some URBOutputData
    // class to separate from Input and GeneralData
    u_out.resize( numcell_cout, 0.0 );
    v_out.resize( numcell_cout, 0.0 );
    w_out.resize( numcell_cout, 0.0 );

    terrain.resize( numcell_cout_2d, 0.0 );
    terrain_id.resize( nx*ny, 1 );
    icellflag_out.resize( numcell_cout, 0.0 );
    /////////////////////////////////////////

    // Set the Wind Velocity data elements to be of the correct size
    // Initialize u0,v0,w0,u,v and w to 0.0
    u0.resize( numcell_face, 0.0 );
    v0.resize( numcell_face, 0.0 );
    w0.resize( numcell_face, 0.0 );
    
    
    //////////////////////////////////////////////////////////////////////////////////
    /////    Create sensor velocity profiles and generate initial velocity field /////
    //////////////////////////////////////////////////////////////////////////////////
    // Calling inputWindProfile function to generate initial velocity
    // field from sensors information (located in Sensor.cpp)

    // Pete could move to input param processing...
    assert( UID->metParams->sensors && (UID->metParams->sensors.size() > 0) );  // extra
                                                                                // check
                                                                                // to
                                                                                // be safe
    // Guaranteed to always have at least 1 sensor!
    // Pete thinks inputWindProfile should be a function of MetParams
    // so it would have access to all the sensors naturally.
    // Make this change later.
    //    UID->metParams->inputWindProfile(UID, this);
    UID->metParams->sensors[0]->inputWindProfile(UID, this);

    max_velmag = 0.0;
    for (auto i=0; i<nx; i++)
    {
      for (auto j=0; j<ny; j++)
      {
        icell_face = i+j*nx+nz*nx*ny;
        max_velmag = MAX_S(max_velmag, sqrt(pow(u0[icell_face],2.0)+pow(v0[icell_face],2.0)));
      }
    }
    max_velmag *= 1.2;




    // ///////////////////////////////////////
    // All iCellFlags should now be set!!!
    // ///////////////////////////////////////

    /// defining ground solid cells (ghost cells below the surface)
    for (int j = 0; j < ny-1; j++)
    {
        for (int i = 0; i < nx-1; i++)
        {
            int icell_cent = i + j*(nx-1);
            icellflag[icell_cent] = 0.0;
        }
    }

    //
    // Terrain stuff should go here too... somehow
    //

    // Now all buildings
    for (int i = 0; i < allBuildingsV.size(); i++)
    {
        // for now this does the canopy stuff for us
        allBuildingsV[i]->setCellFlags();
    }

    std::cout << "All building types (canopies, buildings, terrain) created and initialized...\n";


    // We want to sort ALL buildings here...  use the allBuildingsV to
    // do this... (remember some are canopies) so we may need a
    // virtual function in the Building class to get the appropriate
    // data for the sort.
    mergeSort( effective_height, shpPolygons, base_height, building_height);


    // ///////////////////////////////////////
    // Generic Parameterization Related Stuff
    // ///////////////////////////////////////
    for (int i = 0; i < allBuildingsV.size(); i++)
    {
        // for now this does the canopy stuff for us
        allBuildingsV[i]->callParameterizationSpecial();  
    }

    // Deal with the rest of the parameterization somehow all
    // here... very generically.
    for (int i = 0; i < allBuildingsV.size(); i++)
    {
        // for now this does the canopy stuff for us
        allBuildingsV[i]->callParameterizationOne();
    }

}


float URGBGeneralData::canopyBisection(float ustar, float z0, float canopy_top, float canopy_atten, float vk, float psi_m)
{
    int iter;
    float tol, uhc, d, d1, d2, fi, fnew;

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
{
}

URBGeneralData::~URBGeneralData()
{
}
