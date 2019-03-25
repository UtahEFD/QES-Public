#include "QUSimparams.h"
void quSimParams::build_map()
{


  var_addressMap["nx"]=&nx;
  var_addressMap["ny"]=&ny;
  var_addressMap["nz"]=&nz;

  var_addressMap["dx"]=&dx;
  var_addressMap["dy"]=&dy;
  var_addressMap["dz"]=&dz;

  var_addressMap["vstretch"]=&vstretch;

  var_addressMap["start_time"]=&start_time;
  var_addressMap["time_incr"]=&time_incr;
  var_addressMap["num_time_steps"]=&num_time_steps;
  var_addressMap["day_of_year"]=&day_of_year;
		
  var_addressMap["utc_conversion"]=&utc_conversion;

  var_addressMap["roof_type"]=&roof_type;
  var_addressMap["upwind_type"]=&upwind_type;
  var_addressMap["canyon_type"]=&canyon_type;
  var_addressMap["intersection_flag"]=&intersection_flag;
		
  var_addressMap["max_iterations"]=&max_iterations;
  var_addressMap["residual_reduction"]=&residual_reduction;
  var_addressMap["diffusion_flag"]=&diffusion_flag;
  var_addressMap["diffusion_step"]=&diffusion_step;
		
  var_addressMap["domain_rotation"]=&domain_rotation;
  var_addressMap["utmx"]=&utmx;
  var_addressMap["utmy"]=&utmy;
		
  var_addressMap["utm_zone"]=&utm_zone;
  var_addressMap["quic_cfd_type"]=&quic_cfd_type;
  var_addressMap["wake_type"]=&wake_type;

  var_addressMap["explosive_building_damage"]=&explosive_building_damage;
  var_addressMap["building_array_flag"]=&building_array_flag;




}


quSimParams& quSimParams::operator=(const quSimParams& other)
{

  //  std::cerr<<"operator ---------quSimParams---------"<<std::endl;
  if (this == &other)
    return *this;

  // Check for discovery and default if necessary.

		
  nx=other.nx;
  ny=other.ny;
  nz=other.nz;

  dx=other.dx;
  dy=other.dy;
  dz=other.dz;

  vstretch=other.vstretch;

  start_time=other.start_time;
  time_incr=other.time_incr;
  num_time_steps=other.num_time_steps;
  day_of_year=other.day_of_year;
		
  utc_conversion=other.utc_conversion;

  roof_type=other.roof_type;
  upwind_type=other.upwind_type;
  canyon_type=other.canyon_type;
  intersection_flag=other.intersection_flag;
		
  max_iterations=other.max_iterations;
  residual_reduction=other.residual_reduction;
  diffusion_flag=other.diffusion_flag;
  diffusion_step=other.diffusion_step;
		
  domain_rotation=other.domain_rotation;
  utmx=other.utmx;
  utmy=other.utmy;
		
  utm_zone=other.utm_zone;
  quic_cfd_type=other.quic_cfd_type;
  wake_type=other.wake_type;

  explosive_building_damage=other.explosive_building_damage;
  building_array_flag=other.building_array_flag;   

  return * this;
}

quSimParams::quSimParams()
  : quicDataFile()
{
  start_time     = 0.;
  time_incr      = 1.;
  num_time_steps = 1;

  roof_type = 0;
  upwind_type = 0;
  canyon_type = 0;
  intersection_flag = false;
		
  domain_rotation = 0.;
  utmx = 0;
  utmy = 0;
  utm_zone = 0;
  quic_cfd_type = 0;
  wake_type = 0;
}

bool quSimParams::readQUICFile(const std::string &filename)
{
  // create the legacy file parser to parse the QU_simparams.inp file.
  legacyFileParser* lfp = new legacyFileParser();  

  intElement ie_nx = intElement("nx - Domain Length(X) Grid Cells");
  intElement ie_ny = intElement("ny - Domain Width(Y) Grid Cells");
  intElement ie_nz = intElement("nz - Domain Height(Z) Grid Cells");
  lfp->commit(ie_nx);
  lfp->commit(ie_ny);
  lfp->commit(ie_nz);

  floatElement fe_dx = floatElement("dx (meters)");
  floatElement fe_dy = floatElement("dy (meters)");
  lfp->commit(fe_dx);
  lfp->commit(fe_dy);

  intElement ie_vstretch = intElement("Vertical stretching flag(0=uniform,1=custom,2=parabolic Z,3=parabolic DZ,4=exponential)");
  lfp->commit(ie_vstretch);

  floatElement fe_dz = floatElement("dz (meters)");
  lfp->commit(fe_dz);

  floatElement fe_start_time   = floatElement("decimal start time (hr)");
  floatElement fe_time_incr    = floatElement("time increment (hr)");
  intElement ie_num_time_steps = intElement("total time increments");
  intElement ie_day_of_year = intElement("day of the year");
  lfp->commit(fe_start_time);
  lfp->commit(fe_time_incr);
  lfp->commit(ie_num_time_steps);
  lfp->commit(ie_day_of_year);
		
  intElement ie_utc_conversion = intElement("UTC conversion");
  lfp->commit(ie_utc_conversion);

  intElement ie_roof_type   = intElement("rooftop flag (0-none, 1-log profile, 2-vortex)");
  intElement ie_upwind_type = intElement("upwind cavity flag (0-none, 1-Rockle, 2-MVP, 3-HMVP)");
  intElement ie_canyon_type = intElement("street canyon flag (0-none, 1-Roeckle, 2-CPB, 3-exp. param. PKK, 4-Roeckle w/ Fackrel)");
  boolElement be_intersection_flag = boolElement("street intersection flag (0-off, 1-on)");
  lfp->commit(ie_roof_type);
  lfp->commit(ie_upwind_type);
  lfp->commit(ie_canyon_type);
  lfp->commit(be_intersection_flag);
		
  intElement ie_max_iterations     = intElement("Maximum number of iterations");
  intElement ie_residual_reduction = intElement("Residual Reduction (Orders of Magnitude)");
  boolElement be_diffusion_flag    = boolElement("Use Diffusion Algorithm (1 = on)");
  intElement ie_diffusion_step     = intElement("Number of Diffusion iterations");
  lfp->commit(ie_max_iterations);
  lfp->commit(ie_residual_reduction);
  lfp->commit(be_diffusion_flag);
  lfp->commit(ie_diffusion_step);
		
  floatElement fe_domain_rotation = floatElement("Domain rotation relative to true north (cw = +)");
  intElement ie_utmx              = intElement("UTMX of domain origin (m)");
  intElement ie_utmy              = intElement("UTMY of domain origin (m)");
  lfp->commit(fe_domain_rotation);
  lfp->commit(ie_utmx);
  lfp->commit(ie_utmy);
		
  intElement ie_utm_zone      = intElement("UTM zone");
  intElement be_quic_cfd_type = intElement("QUIC-CFD Flag");
  intElement ie_wake_type     = intElement("wake flag (0-none, 1-Rockle, 2-Modified Rockle)");

  intElement ie_explosive_building_damage = intElement("Explosive building damage flag (1 = on)");

  intElement ie_building_array_flag = intElement("Building Array Flag (1 = on)");  // added by 5.72

  lfp->commit(ie_utm_zone);
  lfp->commit(be_quic_cfd_type);
  lfp->commit(ie_wake_type);
  lfp->commit(ie_explosive_building_damage);
  lfp->commit(ie_building_array_flag);
		
  if (beVerbose)
  {
    std::cout << "\tParsing QU_simparams.inp:" << filename << std::endl;
  }
  lfp->study(filename);

  // Check for discovery and default if necessary.		
  nx = (lfp->recall(ie_nx)) ? ie_nx.value : 0 ;
  ny = (lfp->recall(ie_ny)) ? ie_ny.value : 0 ;
  nz = (lfp->recall(ie_nz)) ? ie_nz.value : 0 ;
  // Why add 1 to all z?  Look at init.f90! The nz is stored 1 short.
  // It may be for the ground that gets "added later."
  nz++;

  if(nx == 0 || ny == 0) {
    std::cerr << "Error::quicLoader::one or more dimensions is zero." << std::endl;
    exit(EXIT_FAILURE);
  }

  dx = (lfp->recall(fe_dx)) ? fe_dx.value : 1.f ;
  dy = (lfp->recall(fe_dy)) ? fe_dy.value : 1.f ;
  dz = (lfp->recall(fe_dz)) ? fe_dz.value : 1.f ;

  vstretch = (lfp->recall(ie_vstretch))     ? ie_vstretch.value     : 0 ;
  // std::cout << vstretch << std::endl;
  start_time = (lfp->recall(fe_start_time))     ? fe_start_time.value     : 0. ;
  time_incr = (lfp->recall(fe_time_incr))      ? fe_time_incr.value      : 0. ;
  num_time_steps = (lfp->recall(ie_num_time_steps)) ? ie_num_time_steps.value : 1 ;

  day_of_year = (lfp->recall(ie_day_of_year)) ? ie_day_of_year.value : 0;
  utc_conversion = (lfp->recall(ie_utc_conversion)) ? ie_utc_conversion.value : 0;
		
  roof_type   = (lfp->recall(ie_roof_type))   ? ie_roof_type.value   : 0 ;
  upwind_type = (lfp->recall(ie_upwind_type)) ? ie_upwind_type.value : 0 ;
  canyon_type = (lfp->recall(ie_canyon_type)) ? ie_canyon_type.value : 0 ;
  intersection_flag = (lfp->recall(be_intersection_flag)) ? be_intersection_flag.value : false ;
		
  max_iterations     = (lfp->recall(ie_max_iterations))     ? ie_max_iterations.value     : 10000 ;
  residual_reduction = (lfp->recall(ie_residual_reduction)) ? ie_residual_reduction.value :     3 ;
  diffusion_flag     = (lfp->recall(be_diffusion_flag))     ? be_diffusion_flag.value     : false ;
  diffusion_step     = (lfp->recall(ie_diffusion_step))     ? ie_diffusion_step.value     :     1 ;
		
  domain_rotation = (lfp->recall(fe_domain_rotation)) ? fe_domain_rotation.value : 0. ;
  utmx = (lfp->recall(ie_utmx)) ? ie_utmx.value : 0 ;
  utmy = (lfp->recall(ie_utmy)) ? ie_utmy.value : 0 ;
		
  utm_zone      = (lfp->recall(ie_utm_zone))      ? ie_utm_zone.value      :     0 ;
  quic_cfd_type = (lfp->recall(be_quic_cfd_type)) ? be_quic_cfd_type.value : false ;
  wake_type     = (lfp->recall(ie_wake_type))     ? ie_wake_type.value     :     0 ;

  explosive_building_damage = (lfp->recall(ie_explosive_building_damage))     ? ie_explosive_building_damage.value     :     0 ;
  building_array_flag = (lfp->recall(ie_building_array_flag))     ? ie_building_array_flag.value     :     0 ;
		
  delete lfp;

  return true;
}

bool quSimParams::writeQUICFile(const std::string &filename)
{
  std::ofstream qufile;
  qufile.open(filename.c_str());

  qufile << "!QUIC 5.72" << std::endl;

  if (qufile.is_open())
    {
      qufile << nx << "\t\t\t!nx - Domain Length(X) Grid Cells" << std::endl;
      qufile << ny << "\t\t\t!ny - Domain Width(Y) Grid Cells" << std::endl;
      qufile << nz << "\t\t\t!nz - Domain Height(Z) Grid Cells" << std::endl;
      qufile << dx << "\t\t\t!dx (meters)" << std::endl;
      qufile << dy << "\t\t\t!dy (meters)" << std::endl;
      qufile << vstretch << "\t\t\t!Vertical stretching flag(0=uniform,1=custom,2=parabolic Z,3=parabolic DZ,4=exponential)" << std::endl;
      qufile << dz << "\t\t\t!dz (meters)" << std::endl;
      qufile << num_time_steps << "\t\t\t!total time increments" << std::endl;
      qufile << day_of_year << "\t\t\t!day of the year" << std::endl;
      qufile << utc_conversion << "\t\t\t!UTC conversion" << std::endl;
      qufile << "!Time(s) of simulations in decimal hours" << std::endl;
      qufile << "0" << std::endl;

      // ??? qufile << start_time << "\t\t\t!decimal start time (hr)" << std::endl;
      // qufile << time_incr << "\t\t\t!time increment (hr)" << std::endl;

      qufile << roof_type << "\t\t\t!rooftop flag (0-none, 1-log profile, 2-vortex)" << std::endl;
      qufile << upwind_type << "\t\t\t!upwind cavity flag (0-none, 1-Rockle, 2-MVP, 3-HMVP)" << std::endl;
      qufile << canyon_type << "\t\t\t!street canyon flag (0-none, 1-Roeckle, 2-CPB, 3-exp. param. PKK, 4-Roeckle w/ Fackrel)" << std::endl;
      qufile << intersection_flag << "\t\t\t!street intersection flag (0-off, 1-on)" << std::endl;
      qufile << wake_type << "\t\t\t!wake flag (0-none, 1-Rockle, 2-Modified Rockle)" << std::endl;
      qufile << max_iterations << "\t\t\t!Maximum number of iterations" << std::endl;
      qufile << residual_reduction << "\t\t\t!Residual Reduction (Orders of Magnitude)" << std::endl;
      qufile << diffusion_flag << "\t\t\t!Use Diffusion Algorithm (1 = on)" << std::endl;
      qufile << diffusion_step << "\t\t\t!Number of Diffusion iterations" << std::endl;
      qufile << domain_rotation << "\t\t\t!Domain rotation relative to true north (cw = +)" << std::endl;
      qufile << utmx << "\t\t\t!UTMX of domain origin (m)" << std::endl;
      qufile << utmy << "\t\t\t!UTMY of domain origin (m)" << std::endl;
      qufile << utm_zone << "\t\t\t!UTM zone" << std::endl;
      qufile << quic_cfd_type << "\t\t\t!QUIC-CFD Flag" << std::endl;
      qufile << explosive_building_damage << "\t\t\t!Explosive building damage flag (1 = on)" << std::endl;

      // added by 5.72
      qufile << building_array_flag << "\t\t\t!Building Array Flag (1 = on)" << std::endl;

      return true;
    }

  return false;
}
