#include "QUFileOptions.h"

quFileOptions::quFileOptions()
: quicDataFile()
{
  format_type = 1;
  uofield_flag   = false;
  uosensor_flag  = false;
	staggered_flag = false;
}


bool quFileOptions::readQUICFile(const std::string &filename)
{

  std::ifstream file(filename.c_str(), std::ifstream::in);

  if(!file.is_open())
  {
	  std::cerr << "urbParser could not open :: " << filename << "." << std::endl;
	  return false;
  }
  else
  {
	  file.close();
  }

  legacyFileParser* lfp = new legacyFileParser();

  intElement ie_output_format_type 
	  = intElement("output data file format flag (1=ascii, 2=binary, 3=both)");

  boolElement be_output_uofield_flag 
	  = boolElement("flag to write out non-mass conserved initial field (uofield.dat) (1=write,0=no write)");

  boolElement	be_output_uosensor_flag 
	  = boolElement("flag to write out the file uosensorfield.dat, the initial sensor velocity field (1=write,0=no write)");

  boolElement be_output_staggered_flag
	  = boolElement("flag to write out the file QU_staggered_velocity.bin used by QUIC-Pressure(1=write,0=no write)");

  lfp->commit(ie_output_format_type);
  lfp->commit(be_output_uofield_flag);
  lfp->commit(be_output_uosensor_flag);
  lfp->commit(be_output_staggered_flag);

  lfp->study(filename);

 // if(lfp->recall(ie_output_format_type))    {format_type    = (FILE_FORMAT_TYPE) ie_output_format_type.value;}
   assert(ie_output_format_type.value >0 && ie_output_format_type.value <4);
   if(lfp->recall(ie_output_format_type))    {format_type    = ie_output_format_type.value;}
  if(lfp->recall(be_output_uofield_flag))   {uofield_flag   = be_output_uofield_flag.value;}
  if(lfp->recall(be_output_uosensor_flag))  {uosensor_flag  = be_output_uosensor_flag.value;}
  if(lfp->recall(be_output_staggered_flag)) {staggered_flag = be_output_staggered_flag.value;}

  delete lfp;
  return true;
}
  
bool quFileOptions::writeQUICFile(const std::string &filename)
{
  std::ofstream qufile;
  qufile.open(filename.c_str());

  qufile << "!QUIC 5.72" << std::endl;

  if (qufile.is_open())
    {
      qufile << format_type << "\t\t\t!output data file format flag (1=ascii, 2=binary, 3=both)" << std::endl;
      qufile << uofield_flag << "\t\t\t!flag to write out non-mass conserved initial field (uofield.dat) (1=write,0=no write)" << std::endl;
      qufile << uosensor_flag << "\t\t\t!flag to write out the file uosensorfield.dat, the initial sensor velocity field (1=write,0=no write)" << std::endl;
      qufile << staggered_flag << "\t\t\t!flag to write out the file QU_staggered_velocity.bin used by QUIC-Pressure(1=write,0=no write)" << std::endl;

      return true;
    }

  return false;
}
