#include "URBOutput_TURBInputFile.h"

URBOutput_TURBInputFile::URBOutput_TURBInputFile(URBGeneralData *ugd,std::string output_file)
  : URBOutput_Generic(output_file)
{
  std::cout<<"Setting fields of TURBInputFile file"<<std::endl;
  
  // set list of fields to save, no option available for this file
  output_fields = {"t","u","v","w","icell","terrain",
		   "e","f","g","h","m","n"};
  
  // set time data dimensions
  std::vector<NcDim> dim_time;
  dim_time.push_back(addDimension("t"));
  // create attributes
  createAttScalar("t","time","s",dim_time,&time);

  // set face-centered data dimensions
  // scalar dimension 
  std::vector<NcDim> dim_vect_z_fc;
  dim_vect_z_fc.push_back(addDimension("z_fc",ugd->nz));
  std::vector<NcDim> dim_vect_y_fc;
  dim_vect_y_fc.push_back(addDimension("y_fc",ugd->ny));
  std::vector<NcDim> dim_vect_x_fc;
  dim_vect_x_fc.push_back(addDimension("x_fc",ugd->nx));
  
  // 3D vector dimension (time dep)
  std::vector<NcDim> dim_vect_fc;
  dim_vect_fc.push_back(dim_time[0]);
  dim_vect_fc.push_back(dim_vect_z_fc[0]);
  dim_vect_fc.push_back(dim_vect_y_fc[0]);
  dim_vect_fc.push_back(dim_vect_x_fc[0]);
  // create attributes
  createAttVector("u","x-component velocity","m s-1",dim_vect_fc,&(ugd->u));
  createAttVector("v","y-component velocity","m s-1",dim_vect_fc,&(ugd->v));
  createAttVector("w","z-component velocity","m s-1",dim_vect_fc,&(ugd->w));

  // set cell-centered data dimensions
  // scalar dimension
  std::vector<NcDim> dim_vect_z_cc;
  dim_vect_z_cc.push_back(addDimension("z_cc",ugd->nz-1));
  std::vector<NcDim> dim_vect_y_cc;
  dim_vect_y_cc.push_back(addDimension("y_cc",ugd->ny-1));
  std::vector<NcDim> dim_vect_x_cc;
  dim_vect_x_cc.push_back(addDimension("x_cc",ugd->nx-1));
  
  // 2D vector (surface, indep of time)
  std::vector<NcDim> dim_vect_2d;
  dim_vect_2d.push_back(dim_vect_y_cc[0]);
  dim_vect_2d.push_back(dim_vect_x_cc[0]);
  // create attributes 
  createAttVector("terrain","terrain height","m",dim_vect_2d,&(ugd->terrain));
  createAttVector("terrain","terrain height","m",dim_vect_2d,&(ugd->terrain));

  // 3D vector dimension (time dep)
  std::vector<NcDim> dim_vect_cc;
  dim_vect_cc.push_back(dim_time[0]);
  dim_vect_cc.push_back(dim_vect_z_cc[0]);
  dim_vect_cc.push_back(dim_vect_y_cc[0]);
  dim_vect_cc.push_back(dim_vect_x_cc[0]);
  // create attributes 
  createAttVector("icell","icell flag value","--",dim_vect_cc,&(ugd->icellflag));  

  // attributes for coefficients for SOR solver
  dim_vect_cc.clear();
  dim_vect_cc.push_back(dim_vect_z_cc[0]);
  dim_vect_cc.push_back(dim_vect_y_cc[0]);
  dim_vect_cc.push_back(dim_vect_x_cc[0]);
  createAttVector("e","e cut-cell coefficient","--",dim_vect_cc,&(ugd->e));  
  createAttVector("f","f cut-cell coefficient","--",dim_vect_cc,&(ugd->f)); 
  createAttVector("g","g cut-cell coefficient","--",dim_vect_cc,&(ugd->g)); 
  createAttVector("h","h cut-cell coefficient","--",dim_vect_cc,&(ugd->h)); 
  createAttVector("m","m cut-cell coefficient","--",dim_vect_cc,&(ugd->m)); 
  createAttVector("n","n cut-cell coefficient","--",dim_vect_cc,&(ugd->n)); 
  
  // adding building informations
  if (ugd->building_id.size()>0) {
    // add building dimension
    std::vector<NcDim> dim_building;
    dim_building.push_back(addDimension("building",ugd->building_id.size()));
    // vector of dimension for time dep building information 
    std::vector<NcDim> dim_building_time;
    dim_building_time.push_back(dim_time[0]);
    dim_building_time.push_back(dim_building[0]);
    
    createAttVector("building_id","ID of building","--",dim_building,&(ugd->building_id)); 
    output_fields.push_back("building_id");

    createAttVector("effective_height","effective height of building","m",
		    dim_building_time,&(ugd->effective_height)); 
    output_fields.push_back("effective_height");
  }
  
  // create output fields
  addOutputFields();
}

bool URBOutput_TURBInputFile::validateFileOtions()
{
  //check all fileoption specificed to make sure it's possible...
  return true;
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
