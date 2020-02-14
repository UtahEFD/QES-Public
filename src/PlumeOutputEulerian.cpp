//
//  NetCDFOutputEulerian.cpp
//  
//  This class handles saving output files for input Eulerian data
//  This is a specialized output class derived 
//   and inheriting from NetCDFOutputGeneric.
//
//  Created by Fabien Margairaz on 01/25/20
//  Modified by Loren Atwood 02/08/20
//


#include "PlumeOutputEulerian.h"


// note that this sets the output file and the bool for whether to do output, in the netcdf inherited classes
PlumeOutputEulerian::PlumeOutputEulerian(PlumeInputData* PID,Urb* urb_ptr,Turb* turb_ptr,Eulerian* eul_ptr,std::string output_file)
  : NetCDFOutputGeneric(output_file)
{
    
    std::cout << "[PlumeOutputEulerian] set up NetCDF file " << output_file << std::endl;


    // no need for output frequency for this output, it is expected to only happen once, assumed to be at time zero
    

    // setup copy of pointers to the classes that save needs so output data can be grabbed directly
    urb = urb_ptr;
    turb = turb_ptr;
    eul = eul_ptr;


    // --------------------------------------------------------
    // setup the output information storage
    // --------------------------------------------------------

    // get the grid number of points
    // LA future work: this whole structure will have to change when we finally adjust the inputs for the true grids
    //  would mean cell centered urb data and face centered turb data. For now, decided just to assume they have the same grid
    int nx = urb->nx;
    int ny = urb->ny;
    int nz = urb->nz;
    int nCells = nz*ny*nx;

    // initialization of the other particle data containers, setting initial vals to different noDataVals
    epps.resize(nCells,-999.0);


    // need to set the epps vars now
    for( int kk = 0; kk < nz; kk++ )
    {
        for( int jj = 0; jj < ny; jj++ )
        {
            for( int ii = 0; ii < nx; ii++ )
            {
                int idx = kk*ny*nx + jj*nx + ii;
                epps.at(idx) = turb->CoEps.at(idx)/eul->C_0;
            }
        }
    }


    // --------------------------------------------------------
    // setup the netcdf output information storage
    // --------------------------------------------------------

    // setup desired output fields string
    // LA future work: can be added in fileOptions at some point
    output_fields = {   "t","x","y","z",
                        "u","v","w",
                        "sig_x","sig_y","sig_z","epps","tke",
                        "txx","txy","txz","tyy","tyz","tzz",
                        "dtxxdx","dtxydy","dtxzdz",
                        "dtxydx","dtyydy","dtyzdz",
                        "dtxzdx","dtyzdy","dtzzdz",
                        "flux_div_x","flux_div_y","flux_div_z"};


    // set data dimensions, which in this case are cell-centered dimensions
    // time dimension
    NcDim NcDim_t = addDimension("t");
    // space dimensions
    NcDim NcDim_x = addDimension("x",nx);
    NcDim NcDim_y = addDimension("y",ny);
    NcDim NcDim_z = addDimension("z",nz);
    
    // create attributes for time dimension
    std::vector<NcDim> dim_vect_t;
    dim_vect_t.push_back(NcDim_t);
    createAttScalar("t","time","s",dim_vect_t,&time);

    // create attributes space dimensions
    std::vector<NcDim> dim_vect_x;
    dim_vect_x.push_back(NcDim_x);
    createAttVector("x","x-distance","m",dim_vect_x,&(urb->x));
    std::vector<NcDim> dim_vect_y;
    dim_vect_y.push_back(NcDim_y);
    createAttVector("y","y-distance","m",dim_vect_y,&(urb->y));
    std::vector<NcDim> dim_vect_z;
    dim_vect_z.push_back(NcDim_z);
    createAttVector("z","z-distance","m",dim_vect_z,&(urb->z));


    // create 3D vector and put in the dimensions (nt,nz,ny,nx).
    // !!! make sure the order is specificall nt,nz,ny,nx in this spot,
    //  the order doesn't seem to matter for other spots
    std::vector<NcDim> dim_vect_3d;
    dim_vect_3d.push_back(NcDim_t);
    dim_vect_3d.push_back(NcDim_z);
    dim_vect_3d.push_back(NcDim_y);
    dim_vect_3d.push_back(NcDim_x);
    

    // create attributes for all output information
    createAttVector("u","x-component mean velocity","m s-1",dim_vect_3d,&urb->u);
    createAttVector("v","y-component mean velocity","m s-1",dim_vect_3d,&urb->v);
    createAttVector("w","z-component mean velocity","m s-1",dim_vect_3d,&urb->w);
    createAttVector("sig_x","x-component variance","m2 s-2",dim_vect_3d,&turb->sig_x);
    createAttVector("sig_y","y-component variance","m2 s-2",dim_vect_3d,&turb->sig_y);
    createAttVector("sig_z","z-component variance","m2 s-2",dim_vect_3d,&turb->sig_z);
    createAttVector("epps","dissipation rate","m2 s-3",dim_vect_3d,&epps);
    createAttVector("tke","turbulent kinetic energy","m2 s-2",dim_vect_3d,&turb->tke);
    createAttVector("txx","uu-component of stress tensor","m2 s-2",dim_vect_3d,&turb->txx);
    createAttVector("txy","uv-component of stress tensor","m2 s-2",dim_vect_3d,&turb->txy);
    createAttVector("txz","uw-component of stress tensor","m2 s-2",dim_vect_3d,&turb->txz);
    createAttVector("tyy","vv-component of stress tensor","m2 s-2",dim_vect_3d,&turb->tyy);
    createAttVector("tyz","vw-component of stress tensor","m2 s-2",dim_vect_3d,&turb->tyz);
    createAttVector("tzz","ww-component of stress tensor","m2 s-2",dim_vect_3d,&turb->tzz);
    createAttVector("dtxxdx","derivative of txx in the x direction","m s-2",dim_vect_3d,&eul->dtxxdx);
    createAttVector("dtxydy","derivative of txy in the y direction","m s-2",dim_vect_3d,&eul->dtxydy);
    createAttVector("dtxzdz","derivative of txz in the z direction","m s-2",dim_vect_3d,&eul->dtxzdz);
    createAttVector("dtxydx","derivative of txy in the x direction","m s-2",dim_vect_3d,&eul->dtxydx);
    createAttVector("dtyydy","derivative of tyy in the y direction","m s-2",dim_vect_3d,&eul->dtyydy);
    createAttVector("dtyzdz","derivative of tyz in the z direction","m s-2",dim_vect_3d,&eul->dtyzdz);
    createAttVector("dtxzdx","derivative of txz in the x direction","m s-2",dim_vect_3d,&eul->dtxzdx);
    createAttVector("dtyzdy","derivative of tyz in the y direction","m s-2",dim_vect_3d,&eul->dtyzdy);
    createAttVector("dtzzdz","derivative of tzz in the z direction","m s-2",dim_vect_3d,&eul->dtzzdz);
    createAttVector("flux_div_x","momentum flux through the x-plane","m s-2",dim_vect_3d,&eul->flux_div_x);
    createAttVector("flux_div_y","momentum flux through the y-plane","m s-2",dim_vect_3d,&eul->flux_div_y);
    createAttVector("flux_div_z","momentum flux through the z-plane","m s-2",dim_vect_3d,&eul->flux_div_z);


    // create output fields
    addOutputFields();

}

// Save output at cell-centered values
void PlumeOutputEulerian::save(float currentTime)
{

    // all the values should already be set by the constructor and by the Eulerian class
    // so just output what is found in the containers


    // set output time for correct netcdf output
    time = currentTime;

    // save the fields to NetCDF files
    saveOutputFields();


    // FM: only remove time dep variables from output array after first save
    // LA note: the output counter is an inherited variable
    if( output_counter == 0 )
    {
        rmTimeIndepFields();
    }

    // increment inherited output counter for next time insertion
    output_counter += 1;

};
