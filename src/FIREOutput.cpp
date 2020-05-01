#include "FIREOutput.h"

FIREOutput::FIREOutput(URBGeneralData *ugd,Fire* fire,std::string output_file)
    : QESNetCDFOutput(output_file)
{

    output_fields={"t","x","y","z","u","v","w","icell","terrain",
                   "burn"};

    // copy of ugd pointer
    ugd_=ugd;
    fire_=fire;
    
    int nx = ugd_->nx;
    int ny = ugd_->ny;
    int nz = ugd_->nz;
    
    long numcell_cout = (nx-1)*(ny-1)*(nz-2);
    
    // Location of face centers in z-dir (without ghost cell)
    z_out.resize( nz-2 );
    for (auto k=1; k<nz-1; k++) {
        z_out[k-1] = ugd_->z[k]; 
    }
  
    // Output data container
    u_out.resize( numcell_cout, 0.0 );
    v_out.resize( numcell_cout, 0.0 );
    w_out.resize( numcell_cout, 0.0 );
    icellflag_out.resize( numcell_cout, 0.0 );

    // set cell-centered data dimensions
    // time dimension
    NcDim NcDim_t=addDimension("t");
  
    // space dimensions
    NcDim NcDim_x=addDimension("x",ugd_->nx-1);
    NcDim NcDim_y=addDimension("y",ugd_->ny-1);
    NcDim NcDim_z=addDimension("z",ugd_->nz-2);

      // create attributes for time dimension
    std::vector<NcDim> dim_vect_t;
    dim_vect_t.push_back(NcDim_t);
    createAttScalar("t","time","s",dim_vect_t,&time);
    
    // create attributes space dimensions
    std::vector<NcDim> dim_vect_x;
    dim_vect_x.push_back(NcDim_x);
    createAttVector("x","x-distance","m",dim_vect_x,&(ugd_->x));
    std::vector<NcDim> dim_vect_y;
    dim_vect_y.push_back(NcDim_y);
    createAttVector("y","y-distance","m",dim_vect_y,&(ugd_->y));
    std::vector<NcDim> dim_vect_z;
    dim_vect_z.push_back(NcDim_z);
    createAttVector("z","z-distance","m",dim_vect_z,&z_out);
    
    // create 2D vector (x,y)
    std::vector<NcDim> dim_vect_2d;
    dim_vect_2d.push_back(NcDim_y);
    dim_vect_2d.push_back(NcDim_x);
    // create attributes
    createAttVector("terrain","terrain height","m",dim_vect_2d,&(ugd_->terrain));

    // create 3D vector (x,y,t)
    std::vector<NcDim> dim_vect_3d;
    dim_vect_3d.push_back(NcDim_t);
    dim_vect_3d.push_back(NcDim_y);
    dim_vect_3d.push_back(NcDim_x);
    // create attributes
    createAttVector("burn","burn flag value","--",dim_vect_3d,&(fire_->burn_out));

    // create 4D vector (x,y,z,t)
    std::vector<NcDim> dim_vect_4d;
    dim_vect_4d.push_back(NcDim_t);
    dim_vect_4d.push_back(NcDim_z);
    dim_vect_4d.push_back(NcDim_y);
    dim_vect_4d.push_back(NcDim_x);
    // create attributes for velocity
    createAttVector("u","x-component velocity","m s-1",dim_vect_4d,&u_out);
    createAttVector("v","y-component velocity","m s-1",dim_vect_4d,&v_out);
    createAttVector("w","z-component velocity","m s-1",dim_vect_4d,&w_out);
    createAttVector("icell","icell flag value","--",dim_vect_4d,&icellflag_out);
  
    // create output fields
    addOutputFields();

}


// Save output at cell-centered values
void FIREOutput::save(float timeOut)
{
    // get grid size (not output var size)
    int nx = ugd_->nx;
    int ny = ugd_->ny;
    int nz = ugd_->nz;
    
    // set time
    time = (double)timeOut;
    
    // get cell-centered values
    for (auto k = 1; k < nz-1; k++) {
        for (auto j = 0; j < ny-1; j++) {
            for (auto i = 0; i < nx-1; i++) {
                int icell_face = i + j*nx + k*nx*ny;
                int icell_cent = i + j*(nx-1) + (k-1)*(nx-1)*(ny-1);
                u_out[icell_cent] = 0.5*(ugd_->u[icell_face+1]+ugd_->u[icell_face]);
                v_out[icell_cent] = 0.5*(ugd_->v[icell_face+nx]+ugd_->v[icell_face]);
                w_out[icell_cent] = 0.5*(ugd_->w[icell_face+nx*ny]+ugd_->w[icell_face]);
                icellflag_out[icell_cent] = ugd_->icellflag[icell_cent+((nx-1)*(ny-1))];
            }
        }
    }
    
    // save the fields to NetCDF files
    saveOutputFields();
    
    // remove x, y, z and terrain
    // from output array after first save
    if (output_counter==0) {
        rmTimeIndepFields();
    }
    
    // increment for next time insertion
    output_counter +=1;
};
