#include "PlumeOutputLagrangian.h"

PlumeOutputLagrangian::PlumeOutputLagrangian(Dispersion* dis,PlumeInputData* PID,std::string output_file)
  : NetCDFOutputGeneric(output_file)
{
  
  std::cout<<"[PlumeOutputLagrangian] set up NetCDF file "<<output_file<<std::endl;
  
  /* --------------------------------------------------------
     setup the output times 
     -------------------------------------------------------- */
  
  // FM -> meed ot create dedicated input variables
  // time to start of Lag output
  stLagTime_ = PID->colParams->timeStart;
  // time to output (1st output -> updated each time)
  outLagTime_ =  stLagTime_+PID->colParams->timeAvg;
  // Copy of the input timeAvg and timeStep 
  outLagFreq_=PID->colParams->timeAvg;

  /* --------------------------------------------------------
     setup the paricle information 
     -------------------------------------------------------- */
  // copy of disp pointer
  disp_=dis;
  
  // get total number of particle to be released 
  int numPar_=0;
  for (auto sidx=0; sidx < disp_->allSources.size(); sidx++) {
    numPar_+=disp_->allSources.at(sidx)->getNumParticles();
  }

  std::cout<<"[PlumeOutputLagrangian] total number of particle to be save in file "<<numPar_<<std::endl;

  // initialization of the container
  ParID.resize(numPar_,0.0);
  for(auto k=0;k<numPar_;k++)
    ParID.at(k)=k;

  xPos.resize(numPar_,-1.0);
  yPos.resize(numPar_,-1.0);
  zPos.resize(numPar_,-1.0);
  
  /* --------------------------------------------------------
     setup the output information 
     -------------------------------------------------------- */
  
  // setup output fields (can be added in fileOpion at one point)
  output_fields = {"t","ParID","xPos","yPos","zPos"};

  // set cell-centered data dimensions
  // time dimension
  NcDim NcDim_t=addDimension("t");
  // particles dimensions
  NcDim NcDim_par=addDimension("parID",numPar_);
  
  // create attributes for time dimension
  std::vector<NcDim> dim_vect_t;
  dim_vect_t.push_back(NcDim_t);
  createAttScalar("t","time","s",dim_vect_t,&time);

  // create attributes space dimensions
  std::vector<NcDim> dim_vect_par;
  dim_vect_par.push_back(NcDim_par);
  createAttVector("ParID","Paricle ID","--",dim_vect_par,&ParID);
  
  // create 2D vector (time,par)
  std::vector<NcDim> dim_vect_2d;
  dim_vect_2d.push_back(NcDim_t);
  dim_vect_2d.push_back(NcDim_par);

  // create attributes particles position
  createAttVector("xPos","x-distance","m",dim_vect_2d,&xPos);
  createAttVector("yPos","y-distance","m",dim_vect_2d,&yPos);
  createAttVector("zPos","z-distance","m",dim_vect_2d,&zPos);

  // create output fields
  addOutputFields();

}

void PlumeOutputLagrangian::save(float currentTime)
{

  if( currentTime >= outLagTime_ ) {
    // copy particle positions
    for (int par=0; par<disp_->pointList.size(); par++) {
      yPos.at(par) = disp_->pointList.at(par).xPos;
      yPos.at(par) = disp_->pointList.at(par).yPos;
      zPos.at(par) = disp_->pointList.at(par).zPos;
    }
    
    // set output time
    time=currentTime;
  
    // save the fields to NetCDF files
    saveOutputFields();
    
    // remove not time dep variable
    // from output array after first save
    if (output_counter==0) {
      rmTimeIndepFields();
    }
    
    // increment for next time insertion
    output_counter +=1;
    
    // avgTime - updated to the time for the next average output
    outLagTime_ += outLagFreq_;   
  }
};
