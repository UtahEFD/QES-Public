#include "TURBOutput.h"

TURBOutput::TURBOutput(TURBGeneralData *tgd,std::string output_file)
  : QESNetCDFOutput(output_file)
{
  std::cout<<"[Output] \t Setting output fields for Turbulence data"<<std::endl;

  output_fields = {"t","time","x","y","z","iturbflag",
		   "S11","S22","S33","S12","S13","S23","L",
		   "tau11","tau12","tau13","tau23","tau22","tau33","tke","CoEps"};

  tgd_=tgd;

  int nx = tgd_->nx;
  int ny = tgd_->ny;
  int nz = tgd_->nz;

  // unused: long numcell_cout = (nx-1)*(ny-1)*(nz-1);

  timestamp.resize( dateStrLen, '0' );
  // set time data dimensions
  NcDim NcDim_t=addDimension("t");
  NcDim NcDim_tstr=addDimension("dateStrLen",dateStrLen);
  // create attributes for time dimension
  std::vector<NcDim> dim_vect_t;
  dim_vect_t.push_back(NcDim_t);
  createAttScalar("t","time","s",dim_vect_t,&time);
  // create attributes for time dimension
  std::vector<NcDim> dim_vect_tstr;
  dim_vect_tstr.push_back(NcDim_t);
  dim_vect_tstr.push_back(NcDim_tstr);
  createAttVector("times","date time","-",dim_vect_tstr,&timestamp);
    
  // set cell-centered data dimensions
  // space dimensions
  NcDim NcDim_x_cc=addDimension("x",nx-1);
  NcDim NcDim_y_cc=addDimension("y",ny-1);
  NcDim NcDim_z_cc=addDimension("z",nz-1);

  // create attributes space dimensions
  std::vector<NcDim> dim_vect_x;
  dim_vect_x.push_back(NcDim_x_cc);
  createAttVector("x","x-distance","m",dim_vect_x,&(tgd_->x_cc));
  std::vector<NcDim> dim_vect_y;
  dim_vect_y.push_back(NcDim_y_cc);
  createAttVector("y","y-distance","m",dim_vect_y,&(tgd_->y_cc));
  std::vector<NcDim> dim_vect_z;
  dim_vect_z.push_back(NcDim_z_cc);
  createAttVector("z","z-distance","m",dim_vect_z,&(tgd_->z_cc));

  // 3D vector dimension (time dep)
  std::vector<NcDim> dim_vect_cc;
  dim_vect_cc.push_back(NcDim_t);
  dim_vect_cc.push_back(NcDim_z_cc);
  dim_vect_cc.push_back(NcDim_y_cc);
  dim_vect_cc.push_back(NcDim_x_cc);

  createAttVector("iturbflag","icell turb flag","--",dim_vect_cc,&(tgd_->iturbflag));

  // create attributes for strain-rate stress tensor
  createAttVector("S11","uu-component of strain-rate tensor","s-1",dim_vect_cc,&(tgd_->S11));
  createAttVector("S22","vv-component of strain-rate tensor","s-1",dim_vect_cc,&(tgd_->S22));
  createAttVector("S33","ww-component of strain-rate tensor","s-1",dim_vect_cc,&(tgd_->S33));
  createAttVector("S12","uv-component of strain-rate tensor","s-1",dim_vect_cc,&(tgd_->S12));
  createAttVector("S13","uw-component of strain-rate tensor","s-1",dim_vect_cc,&(tgd_->S13));
  createAttVector("S23","vw-component of strain-rate tensor","s-1",dim_vect_cc,&(tgd_->S23));

  // create attribute for mixing length
  createAttVector("L","mixing length","m",dim_vect_cc,&(tgd_->Lm));

  // create derived attributes
  createAttVector("tau11","uu-component of stress tensor","m2s-2",dim_vect_cc,&(tgd_->tau11));
  createAttVector("tau22","vv-component of stress tensor","m2s-2",dim_vect_cc,&(tgd_->tau22));
  createAttVector("tau33","ww-component of stress tensor","m2s-2",dim_vect_cc,&(tgd_->tau33));
  createAttVector("tau12","uv-component of stress tensor","m2s-2",dim_vect_cc,&(tgd_->tau12));
  createAttVector("tau13","uw-component of stress tensor","m2s-2",dim_vect_cc,&(tgd_->tau13));
  createAttVector("tau23","vw-component of stress tensor","m2s-2",dim_vect_cc,&(tgd_->tau23));
  createAttVector("tke","turbulent kinetic energy","m2s-2",dim_vect_cc,&(tgd_->tke));
  createAttVector("CoEps","dissipation rate","m2s-3",dim_vect_cc,&(tgd_->CoEps));

  // create output fields
  addOutputFields();

}

// Save output at cell-centered values
void TURBOutput::save(ptime timeOut)
{

    // set time
    time = (double)output_counter;
    
    std::string s=to_iso_extended_string(timeOut);
    std::copy(s.begin(), s.end(), timestamp.begin());
   
    // save fields
    saveOutputFields();

    // remmove time indep from output array after first save
    if (output_counter==0) {
        rmTimeIndepFields();
    }

    // increment for next time insertion
    output_counter +=1;

};
