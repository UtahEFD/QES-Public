#include "URBOutputData.h"

URBOutputData :: URBOutputData(URBGeneralData* UGD,URBInputData* UID,std::string outname)
{
  if (UID->fileOptions->outputFlag==1) {
    std::string fname=outname;
    fname.append("_VizFile.nc");
    output_viz = new URBOutput_VizFields(UGD,fname);
  }

  if (UID->fileOptions->outputFlag==1) {
    std::string fname=outname;
    fname.append("_TURBInputFile.nc");
    output_turb = new URBOutput_TURBInputFile(UGD,fname);
  }
}

void URBOutputData::save(URBGeneralData* UGD)
{
  std::cout<<"Output ..."<<std::endl;
  if (output_viz) {
    output_viz->save(UGD);
  }
  if (output_turb) {
    output_turb->save(UGD);
  } 
}
