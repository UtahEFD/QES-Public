#include "URBOutput_TURBInputFile.h"

URBOutput_TURBInputFile::URBOutput_TURBInputFile(URBGeneralData *ugd,std::string output_file)
  : URBOutput_Generic(output_file)
{
  std::cout<<"Setting fields of TURBInputFile file"<<std::endl;
  
  // set list of fields to save, no option available for this file
  output_fields = {"t","u","v","w","icell","terrain",
		   "e","f","g","h","m","n"};
  
  // set time data dimensions
  NcDim NcDim_t=addDimension("t");
  // create attributes for time dimension 
  std::vector<NcDim> dim_vect_t;
  dim_vect_t.push_back(NcDim_t);
  createAttScalar("t","time","s",dim_vect_t,&time);

  // set face-centered data dimensions
  // space dimensions 
  NcDim NcDim_x_fc=addDimension("x",ugd->nx);
  NcDim NcDim_y_fc=addDimension("y",ugd->ny);
  NcDim NcDim_z_fc=addDimension("z",ugd->nz);
  
  // 3D vector dimension (time dep)
  std::vector<NcDim> dim_vect_fc;
  dim_vect_fc.push_back(NcDim_t);
  dim_vect_fc.push_back(NcDim_z_fc);
  dim_vect_fc.push_back(NcDim_y_fc);
  dim_vect_fc.push_back(NcDim_x_fc);
  // create attributes
  createAttVector("u","x-component velocity","m s-1",dim_vect_fc,&(ugd->u));
  createAttVector("v","y-component velocity","m s-1",dim_vect_fc,&(ugd->v));
  createAttVector("w","z-component velocity","m s-1",dim_vect_fc,&(ugd->w));

  // set cell-centered data dimensions
  // space dimensions 
  NcDim NcDim_x_cc=addDimension("x_cc",ugd->nx-1);
  NcDim NcDim_y_cc=addDimension("y_cc",ugd->ny-1);
  NcDim NcDim_z_cc=addDimension("z_cc",ugd->nz-2);
  
  // create 2D vector (surface, indep of time)
  std::vector<NcDim> dim_vect_2d;
  dim_vect_2d.push_back(NcDim_y_cc);
  dim_vect_2d.push_back(NcDim_x_cc);
  // create attributes 
  createAttVector("terrain","terrain height","m",dim_vect_2d,&(ugd->terrain));
  createAttVector("terrain","terrain height","m",dim_vect_2d,&(ugd->terrain));

  // 3D vector dimension (time dep)
  std::vector<NcDim> dim_vect_cc;
  dim_vect_cc.push_back(NcDim_t);
  dim_vect_cc.push_back(NcDim_z_cc);
  dim_vect_cc.push_back(NcDim_y_cc);
  dim_vect_cc.push_back(NcDim_x_cc);
  // create attributes 
  createAttVector("icell","icell flag value","--",dim_vect_cc,&(ugd->icellflag));  

  // attributes for coefficients for SOR solver
  dim_vect_cc.clear();
  dim_vect_cc.push_back(NcDim_z_cc);
  dim_vect_cc.push_back(NcDim_y_cc);
  dim_vect_cc.push_back(NcDim_x_cc);
  createAttVector("e","e cut-cell coefficient","--",dim_vect_cc,&(ugd->e));  
  createAttVector("f","f cut-cell coefficient","--",dim_vect_cc,&(ugd->f)); 
  createAttVector("g","g cut-cell coefficient","--",dim_vect_cc,&(ugd->g)); 
  createAttVector("h","h cut-cell coefficient","--",dim_vect_cc,&(ugd->h)); 
  createAttVector("m","m cut-cell coefficient","--",dim_vect_cc,&(ugd->m)); 
  createAttVector("n","n cut-cell coefficient","--",dim_vect_cc,&(ugd->n)); 
  
  // adding building informations
  if (ugd->building_id.size()>0) {
    // building dimension
    NcDim NcDim_building=addDimension("building",ugd->building_id.size());
    // vector of dimension for building information 
    std::vector<NcDim> dim_vect_building;
    dim_vect_building.push_back(NcDim_building);
    // create attributes
    createAttVector("building_id","ID of building","--",dim_vect_building,&(ugd->building_id)); 
    // add to output fields
    output_fields.push_back("building_id");
    
    // vector of dimension for time dep building information 
    std::vector<NcDim> dim_vect_building_t;
    dim_vect_building_t.push_back(NcDim_t);
    dim_vect_building_t.push_back(NcDim_building);
    // create attributes
    createAttVector("effective_height","effective height of building","m",
		    dim_vect_building_t,&(ugd->effective_height)); 
    // add to output fields
    output_fields.push_back("effective_height");
  }
  
  // create output fields
  addOutputFields();
}

  
// Save output at cell-centered values
void URBOutput_TURBInputFile::save(URBGeneralData *ugd)
{
  
  // set time
  time = (double)output_counter;
  
  // save fields
  saveOutputFields();

  // remmove time indep from output array after first save
  if (output_counter==0) {
    rmTimeIndepFields();
  }
  
  // increment for next time insertion
  output_counter +=1;

  
};
