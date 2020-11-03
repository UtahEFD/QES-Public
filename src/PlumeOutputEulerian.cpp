//
//  NetCDFOutputEulerian.cpp
//  
//  This class handles saving output files for input Eulerian data
//  This is a specialized output class derived 
//   and inheriting from QESNetCDFOutput.
//
//  Created by Fabien Margairaz on 01/25/20
//  Modified by Loren Atwood 02/08/20
//

#include "PlumeOutputEulerian.h"


// note that this sets the output file and the bool for whether to do output, in the netcdf inherited classes
PlumeOutputEulerian::PlumeOutputEulerian(PlumeInputData* PID,WINDSGeneralData* WGD,TURBGeneralData* TGD,Eulerian* eul,std::string output_file)
  : QESNetCDFOutput(output_file)
{
    
    std::cout << "[PlumeOutputEulerian] set up NetCDF file " << output_file << std::endl;

    // no need for output frequency for this output, it is expected to only happen once, assumed to be at time zero    

    // setup copy of pointers to the classes that save needs so output data can be grabbed directly
    WGD_ = WGD;
    TGD_ = TGD;
    eul_ = eul;

    // --------------------------------------------------------
    // setup the output information storage
    // --------------------------------------------------------

    // get the grid number of points
    // LA future work: this whole structure will have to change when we finally adjust the inputs for the true grids
    //  would mean cell centered urb data and face centered turb data. For now, decided just to assume they have the same grid
    int nx = WGD_->nx;
    int ny = WGD_->ny;
    int nz = WGD_->nz;
    int nFaces = nx*ny*nx;
    int nCells = (nz-1)*(ny-1)*(nx-1);

    // initialization of the other particle data containers, setting initial vals to different noDataVals
    epps.resize(nCells,-999.0);

    // --------------------------------------------------------
    // setup the netcdf output information storage
    // --------------------------------------------------------

    // setup desired output fields string
    output_fields = {   "t","x","y","z",
                        "u","v","w",
                        "sig_x","sig_y","sig_z",
                        "epps","tke",
                        "txx","txy","txz","tyy","tyz","tzz",
                        "dtxxdx","dtxydy","dtxzdz",
                        "dtxydx","dtyydy","dtyzdz",
                        "dtxzdx","dtyzdy","dtzzdz"};


    // set data dimensions, which in this case are cell-centered dimensions
    // time dimension
    NcDim NcDim_t = addDimension("t");
    // space dimensions
    NcDim NcDim_x_cell = addDimension("x",nx-1);
    NcDim NcDim_y_cell = addDimension("y",ny-1);
    NcDim NcDim_z_cell = addDimension("z",nz-1);
    
    NcDim NcDim_x_face = addDimension("x_face",nx);
    NcDim NcDim_y_face = addDimension("y_face",ny);
    NcDim NcDim_z_face = addDimension("z_face",nz);
    
    // create attributes for time dimension
    std::vector<NcDim> dim_vect_t;
    dim_vect_t.push_back(NcDim_t);
    createAttScalar("t","time","s",dim_vect_t,&time);

    // create attributes space dimensions
    std::vector<NcDim> dim_vect_x;
    dim_vect_x.push_back(NcDim_x_cell);
    createAttVector("x","x-distance","m",dim_vect_x,&(WGD_->x));
    std::vector<NcDim> dim_vect_y;
    dim_vect_y.push_back(NcDim_y_cell);
    createAttVector("y","y-distance","m",dim_vect_y,&(WGD_->y));
    std::vector<NcDim> dim_vect_z;
    dim_vect_z.push_back(NcDim_z_cell);
    createAttVector("z","z-distance","m",dim_vect_z,&(WGD_->z));


    // create 3D vector and put in the dimensions (nt,nz,ny,nx).
    // !!! make sure the order is specificall nt,nz,ny,nx 

    // for face-center variable
    std::vector<NcDim> dim_vect_3d_face;
    dim_vect_3d_face.push_back(NcDim_t);
    dim_vect_3d_face.push_back(NcDim_z_face);
    dim_vect_3d_face.push_back(NcDim_y_face);
    dim_vect_3d_face.push_back(NcDim_x_face);
    
    // for cell-center variable
    std::vector<NcDim> dim_vect_3d_cell;
    dim_vect_3d_cell.push_back(NcDim_t);
    dim_vect_3d_cell.push_back(NcDim_z_cell);
    dim_vect_3d_cell.push_back(NcDim_y_cell);
    dim_vect_3d_cell.push_back(NcDim_x_cell);
    
    // create attributes for all output information 
    // -> QES-winds variables
    createAttVector("u","x-component mean velocity","m s-1",dim_vect_3d_face,&WGD_->u);
    createAttVector("v","y-component mean velocity","m s-1",dim_vect_3d_face,&WGD_->v);
    createAttVector("w","z-component mean velocity","m s-1",dim_vect_3d_face,&WGD_->w);
    // -> QES-turb variables
    createAttVector("epps","dissipation rate","m2 s-3",dim_vect_3d_cell,&epps);
    createAttVector("tke","turbulent kinetic energy","m2 s-2",dim_vect_3d_cell,&TGD_->tke);
    createAttVector("txx","uu-component of stress tensor","m2 s-2",dim_vect_3d_cell,&TGD_->txx);
    createAttVector("txy","uv-component of stress tensor","m2 s-2",dim_vect_3d_cell,&TGD_->txy);
    createAttVector("txz","uw-component of stress tensor","m2 s-2",dim_vect_3d_cell,&TGD_->txz);
    createAttVector("tyy","vv-component of stress tensor","m2 s-2",dim_vect_3d_cell,&TGD_->tyy);
    createAttVector("tyz","vw-component of stress tensor","m2 s-2",dim_vect_3d_cell,&TGD_->tyz);
    createAttVector("tzz","ww-component of stress tensor","m2 s-2",dim_vect_3d_cell,&TGD_->tzz);
    // -> Eulerian variables
    createAttVector("sig_x","x-component variance","m2 s-2",dim_vect_3d_cell,&eul_->sig_x);
    createAttVector("sig_y","y-component variance","m2 s-2",dim_vect_3d_cell,&eul_->sig_y);
    createAttVector("sig_z","z-component variance","m2 s-2",dim_vect_3d_cell,&eul_->sig_z);
    createAttVector("dtxxdx","derivative of txx in the x direction","m s-2",dim_vect_3d_cell,&eul_->dtxxdx);
    createAttVector("dtxydy","derivative of txy in the y direction","m s-2",dim_vect_3d_cell,&eul_->dtxydy);
    createAttVector("dtxzdz","derivative of txz in the z direction","m s-2",dim_vect_3d_cell,&eul_->dtxzdz);
    createAttVector("dtxydx","derivative of txy in the x direction","m s-2",dim_vect_3d_cell,&eul_->dtxydx);
    createAttVector("dtyydy","derivative of tyy in the y direction","m s-2",dim_vect_3d_cell,&eul_->dtyydy);
    createAttVector("dtyzdz","derivative of tyz in the z direction","m s-2",dim_vect_3d_cell,&eul_->dtyzdz);
    createAttVector("dtxzdx","derivative of txz in the x direction","m s-2",dim_vect_3d_cell,&eul_->dtxzdx);
    createAttVector("dtyzdy","derivative of tyz in the y direction","m s-2",dim_vect_3d_cell,&eul_->dtyzdy);
    createAttVector("dtzzdz","derivative of tzz in the z direction","m s-2",dim_vect_3d_cell,&eul_->dtzzdz);
 
    // create output fields
    addOutputFields();

}

// Save output at cell-centered values
void PlumeOutputEulerian::save(float currentTime)
{
    // all the values should already be set by the constructor and by the Eulerian class
    // so just output what is found in the containers
    for(size_t id = 0; id < TGD_->CoEps.size(); id++ ) {
        epps.at(id) = TGD_->CoEps.at(id)/eul_->C_0;
    }

    // set output time for correct netcdf output
    time = currentTime;

    // save the fields to NetCDF files
    saveOutputFields();

    // FM: only remove time dep variables from output array after first save
    // LA note: the output counter is an inherited variable
    if( output_counter == 0 ) {
        rmTimeIndepFields();
    }

    // increment inherited output counter for next time insertion
    output_counter += 1;

};
