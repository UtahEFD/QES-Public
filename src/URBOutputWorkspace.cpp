#include "URBOutputWorkspace.h"

URBOutputWorkspace::URBOutputWorkspace(URBGeneralData *ugd,std::string output_file)
  : NetCDFOutputGeneric(output_file)
{
  std::cout<<"Setting fields of TURBInputFile file"<<std::endl;
  
  // set list of fields to save, no option available for this file
  if (ugd->includesMixingLength()) {
      output_fields = {"t","x_cc","y_cc","z_cc","u","v","w","icell","terrain",
                       "mixlength",
                       "e","f","g","h","m","n"};
  }
  else {
      output_fields = {"t","x_cc","y_cc","z_cc","u","v","w","icell","terrain",
                       "e","f","g","h","m","n"};
  }
  
  // copy of ugd pointer
  ugd_=ugd;

  int nx = ugd_->nx;
  int ny = ugd_->ny;
  int nz = ugd_->nz;

  // copy of ugd pointer
  ugd_=ugd;

  // Location of face centers in z-dir
  z_cc.resize( nz-1 );
  for (auto k=0; k<nz-1; k++) {
    z_cc[k] = (k-0.5)*ugd_->dz; 
  }
  // Location of face centers in x-dir
  x_cc.resize( nx-1 );
  for (auto i=0; i<nx-1; i++) {
    x_cc[i] = (i+0.5)*ugd_->dx; 
  }
  // Location of face centers in y-dir
  y_cc.resize( ny-1 );
  for (auto j=0; j<ny-1; j++) {
    y_cc[j] = (j+0.5)*ugd_->dy; 
  }

  // set time data dimensions
  NcDim NcDim_t=addDimension("t");
  // create attributes for time dimension 
  std::vector<NcDim> dim_vect_t;
  dim_vect_t.push_back(NcDim_t);
  createAttScalar("t","time","s",dim_vect_t,&time);

  // set face-centered data dimensions
  // space dimensions 
  NcDim NcDim_x_fc=addDimension("x",ugd_->nx);
  NcDim NcDim_y_fc=addDimension("y",ugd_->ny);
  NcDim NcDim_z_fc=addDimension("z",ugd_->nz);
  
  // 3D vector dimension (time dep)
  std::vector<NcDim> dim_vect_fc;
  dim_vect_fc.push_back(NcDim_t);
  dim_vect_fc.push_back(NcDim_z_fc);
  dim_vect_fc.push_back(NcDim_y_fc);
  dim_vect_fc.push_back(NcDim_x_fc);
  // create attributes
  createAttVector("u","x-component velocity","m s-1",dim_vect_fc,&(ugd_->u));
  createAttVector("v","y-component velocity","m s-1",dim_vect_fc,&(ugd_->v));
  createAttVector("w","z-component velocity","m s-1",dim_vect_fc,&(ugd_->w));

  // set cell-centered data dimensions
  // space dimensions 
  NcDim NcDim_x_cc=addDimension("x_cc",ugd_->nx-1);
  NcDim NcDim_y_cc=addDimension("y_cc",ugd_->ny-1);
  NcDim NcDim_z_cc=addDimension("z_cc",ugd_->nz-1);
  
  // create attributes space dimensions 
  std::vector<NcDim> dim_vect_x_cc;
  dim_vect_x_cc.push_back(NcDim_x_cc);
  createAttVector("x_cc","x-distance","m",dim_vect_x_cc,&x_cc);
  std::vector<NcDim> dim_vect_y_cc;
  dim_vect_y_cc.push_back(NcDim_y_cc);
  createAttVector("y_cc","y-distance","m",dim_vect_y_cc,&y_cc);
  std::vector<NcDim> dim_vect_z_cc;
  dim_vect_z_cc.push_back(NcDim_z_cc); 
  createAttVector("z_cc","z-distance","m",dim_vect_z_cc,&z_cc);

  // create 2D vector (surface, indep of time)
  std::vector<NcDim> dim_vect_2d;
  dim_vect_2d.push_back(NcDim_y_cc);
  dim_vect_2d.push_back(NcDim_x_cc);
  // create attributes 
  createAttVector("terrain","terrain height","m",dim_vect_2d,&(ugd_->terrain));

  // 3D vector dimension (time dep)
  std::vector<NcDim> dim_vect_cc;
  dim_vect_cc.push_back(NcDim_t);
  dim_vect_cc.push_back(NcDim_z_cc);
  dim_vect_cc.push_back(NcDim_y_cc);
  dim_vect_cc.push_back(NcDim_x_cc);

  // create attributes 
  createAttVector("icell","icell flag value","--",dim_vect_cc,&(ugd_->icellflag));

  if (ugd->includesMixingLength()) {
      createAttVector("mixlength","mixing length value","--",dim_vect_cc,&(ugd_->mixingLengths));
  }

  // attributes for coefficients for SOR solver
  createAttVector("e","e cut-cell coefficient","--",dim_vect_cc,&(ugd_->e));  
  createAttVector("f","f cut-cell coefficient","--",dim_vect_cc,&(ugd_->f)); 
  createAttVector("g","g cut-cell coefficient","--",dim_vect_cc,&(ugd_->g)); 
  createAttVector("h","h cut-cell coefficient","--",dim_vect_cc,&(ugd_->h)); 
  createAttVector("m","m cut-cell coefficient","--",dim_vect_cc,&(ugd_->m)); 
  createAttVector("n","n cut-cell coefficient","--",dim_vect_cc,&(ugd_->n)); 
  
  // adding building informations
  /* FM -> commented in workingBranch
  if (ugd_->allBuildingsV.size()>0) {
    // building dimension
    NcDim NcDim_building=addDimension("building",ugd_->allBuildingsV.size());
    // vector of dimension for building information 
    std::vector<NcDim> dim_vect_building;
    dim_vect_building.push_back(NcDim_building);
        
    // vector of dimension for time dep building information 
    std::vector<NcDim> dim_vect_building_t;
    dim_vect_building_t.push_back(NcDim_t);
    dim_vect_building_t.push_back(NcDim_building);
    // create attributes
    createAttVector("effective_height","effective height of building","m",
		    dim_vect_building_t,&(ugd_->effective_height)); 
    output_fields.push_back("effective_height");
    
  }
  */

  // create output fields
  addOutputFields();
}

  
// Save output at cell-centered values
void URBOutputWorkspace::save(float timeOut)
{
  
  // set time
  time = (double)timeOut;
  
  // save fields
  saveOutputFields();

  // remmove time indep from output array after first save
  if (output_counter==0) {
    rmTimeIndepFields();
  }
  
  // increment for next time insertion
  output_counter +=1;

  
};
