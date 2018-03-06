
#include "QPParams.h"

void qpParams::build_map()
{



        var_addressMap["sourceFlag"]=&sourceFlag;
	 
	var_addressMap["westernEdge"]=&westernEdge;
	var_addressMap["southernEdge"]=&southernEdge;
	var_addressMap["z0"]=&z0;
	var_addressMap["rcl"]=&rcl;

	var_addressMap["boundaryLayerHeight"]=&boundaryLayerHeight;
	var_addressMap["numParticles"]=&numParticles;

	var_addressMap["partNumIncreaseFactor"]=&partNumIncreaseFactor;
	var_addressMap["partSplittingDosage"]=&partSplittingDosage;

	var_addressMap["taylorMicroscaleMin"]=&taylorMicroscaleMin;
	var_addressMap["randomNumberSeed"]=&randomNumberSeed;

	var_addressMap["timeStep"]=&timeStep;
	var_addressMap["duration"]=&duration;

	var_addressMap["concAvgTime"]=&concAvgTime;
	var_addressMap["concStartTime"]=&concStartTime;

	var_addressMap["partOutputPeriod"]=&partOutputPeriod;
	var_addressMap["nbx"]=&nbx;

	var_addressMap["nby"]=&nby;
	var_addressMap["nbz"]=&nbz;

	var_addressMap["xbl"]=&xbl;
	var_addressMap["xbu"]=&xbu;
	var_addressMap["ybl"]=&ybl;

	var_addressMap["ybu"]=&ybu;
	var_addressMap["zbl"]=&zbl;
	var_addressMap["zbu"]=&zbu;

	var_addressMap["isiteflag"]=&isiteflag;
	var_addressMap["iindoorflag"]=&iindoorflag;
	var_addressMap["inextgridflag"]=&inextgridflag;

	var_addressMap["nonLocalMixing"]=&nonLocalMixing;
	var_addressMap["useCFDTurbulence"]=&useCFDTurbulence;

	var_addressMap["particleDistFlag"]=&particleDistFlag;
	var_addressMap["particleSplitFlag"]=&particleSplitFlag;
	var_addressMap["particleRecyclingFlag"]=&particleRecyclingFlag;

	var_addressMap["numParticleSplit"]=&numParticleSplit;



}
qpParams& qpParams::operator=(const qpParams& other)
{

  // std::cerr<<"operator ---------qpParams---------"<<std::endl;
  if (this == &other)
    return *this;

	
  sourceFlag=other.sourceFlag;  
	
  isiteflag=other.isiteflag;   // !normal QUIC (isitefl=0) or sensor siting (=1) mode
  iindoorflag=other.iindoorflag; // !indoor calculations turned off (=0) or turned on (=1)
  inextgridflag=other.inextgridflag;      // !1 - inner grid, 2 - outer grid
  westernEdge=other.westernEdge;  // !Location of western edge of inner grid relative to outer grid (m)
  southernEdge=other.southernEdge; // !Location of southern edge of inner relative to outer grid (m)
  z0=other.rcl;  // wallSurfRoughness;    // !Wall Surface Roughness Length (m)
  rcl=other.rcl;  // !Reciprocal Monin-Obukhov length(1/m)
  boundaryLayerHeight=other.boundaryLayerHeight;  // !Boundary Layer height (m)
  nonLocalMixing=other.nonLocalMixing;        // !use 1 to enable non-local mixing
  useCFDTurbulence=other.useCFDTurbulence;        // !use 1 to enable use of QUIC-CFD turbulence
  numParticles=other.numParticles;          // !number of particles released over entire simulation
  particleDistFlag=other.particleDistFlag;     // !Number of particle distribution flag (1 = by mass, 2 = by source)
  particleSplitFlag=other.particleSplitFlag;     // !Particle splitting flag
  particleRecyclingFlag=other.particleRecyclingFlag; // !Particle recycling flag
  partNumIncreaseFactor=other.partNumIncreaseFactor;  // !Total particle number increase factor
  numParticleSplit=other.numParticleSplit;     // !Number of particles a particle is split into
  partSplittingDosage=other.partSplittingDosage; // !Particle splitting target dose (gs/m^3)
  taylorMicroscaleMin=other.taylorMicroscaleMin;  // !Enable Taylor microscale lower limit to sub-time steps
  randomNumberSeed=other.randomNumberSeed;  // !Random number seed
  timeStep=other.timeStep;       // !time step (s)
  duration=other.duration;       // !duration (s)
  concAvgTime=other.concAvgTime;   // !concentration averaging time (s)
  concStartTime=other.concStartTime; // !starting time for concentration averaging (s)
  partOutputPeriod=other.partOutputPeriod; // !particle output period (s)
  nbx=other.nbx;  // !in x direction, # of collecting boxes (concentration grid cells) 
  nby=other.nby;  // !in y direction, # of collecting boxes (concentration grid cells) 
  nbz=other.nbz;  // !in z direction, # of collecting boxes (concentration grid cells) 
  xbl=other.xbl;  // !lower limits for collecting boxes in x in meters
  xbu=other.xbu;  // !upper limits for collecting boxes in x direction in meters
  ybl=other.ybl;  // !lower limits for collecting boxes in y in meters
  ybu=other.ybu;  // !upper limits for collecting boxes in y direction in meters
  zbl=other.zbl;  // !lower limits for collecting boxes in z in meters
  zbu=other.zbu;  // !upper limits for collecting boxes in z direction in meters

  return * this;
}

bool qpParams::readQUICFile(const std::string &filename)
{
  // ///////////////////////////////////////////////////////////
  // 
  // create the legacy file parser to parse the QP_params.inp file.
  // 
  legacyFileParser *lfp = new legacyFileParser();  

  intElement ie_sourceTypeFlag = intElement("Source type flag (1 = Basic, 2 = Dense Gas, 3 = Distributed Particle Size, 4 = Explosive, 5 = ERAD source, 6 = Bio Slurry, 7 = 2-phase, 8 = Exfiltration)");
  boolElement ie_isiteflag = boolElement("normal QUIC (isitefl=0) or sensor siting (=1) mode");
  boolElement ie_iindoorflag = boolElement("indoor calculations turned off (=0) or turned on (=1)");
  intElement ie_inextgridflag = intElement("1 - inner grid, 2 - outer grid");
  floatElement ie_westernEdge = floatElement("Location of western edge of inner grid relative to outer grid (m)");
  floatElement ie_southernEdge = floatElement("Location of southern edge of inner relative to outer grid (m)");
  floatElement ie_z0 = floatElement("Wall Surface Roughness Length (m)");
  floatElement ie_rcl = floatElement("Reciprocal Monin-Obukhov length(1/m)");
  floatElement ie_boundaryLayerHeight = floatElement("Boundary Layer height (m)");
  boolElement ie_nonLocalMixing = boolElement("use 1 to enable non-local mixing");
  boolElement ie_useCFDTurbulence = boolElement("use 1 to enable use of QUIC-CFD turbulence");
  intElement ie_numParticles = intElement("number of particles released over entire simulation");
  intElement ie_particleDistFlag = intElement("Number of particle distribution flag (1 = by mass, 2 = by source)");
  boolElement ie_particleSplitFlag = boolElement("Particle splitting flag");
  boolElement ie_particleRecyclingFlag = boolElement("Particle recycling flag");
  intElement ie_partNumIncreaseFactor = intElement("Total particle number increase factor");
  intElement ie_numParticleSplit = intElement("Number of particles a particle is split into");
  floatElement ie_partSplittingDosage = floatElement("Particle splitting target dose (gs/m^3)");
  floatElement ie_taylorMicroscaleMin = floatElement("Enable Taylor microscale lower limit to sub-time steps");
  intElement ie_randomNumberSeed = intElement("Random number seed");
  floatElement ie_timeStep = floatElement("time step (s)");
  floatElement ie_duration = floatElement("duration (s)");
  floatElement ie_concAvgTime = floatElement("concentration averaging time (s)");
  floatElement ie_concStartTime = floatElement("starting time for concentration averaging (s)");
  floatElement ie_partOutputPeriod = floatElement("particle output period (s)");
  floatElement ie_nbx = floatElement("in x direction, # of collecting boxes (concentration grid cells) ");
  floatElement ie_nby = floatElement("in y direction, # of collecting boxes (concentration grid cells) ");
  floatElement ie_nbz = floatElement("in z direction, # of collecting boxes (concentration grid cells) ");
  floatElement ie_xbl = floatElement("lower limits for collecting boxes in x in meters");
  floatElement ie_xbu = floatElement("upper limits for collecting boxes in x direction in meters");
  floatElement ie_ybl = floatElement("lower limits for collecting boxes in y in meters");
  floatElement ie_ybu = floatElement("upper limits for collecting boxes in y direction in meters");
  floatElement ie_zbl = floatElement("lower limits for collecting boxes in z in meters");
  floatElement ie_zbu = floatElement("upper limits for collecting boxes in z direction in meters");

  lfp->commit(ie_sourceTypeFlag);
  lfp->commit(ie_isiteflag);
  lfp->commit(ie_iindoorflag);
  lfp->commit(ie_inextgridflag);
  lfp->commit(ie_westernEdge);
  lfp->commit(ie_southernEdge);
  lfp->commit(ie_z0);
  lfp->commit(ie_rcl);
  lfp->commit(ie_boundaryLayerHeight);
  lfp->commit(ie_nonLocalMixing);
  lfp->commit(ie_useCFDTurbulence);
  lfp->commit(ie_numParticles);
  lfp->commit(ie_particleDistFlag);
  lfp->commit(ie_particleSplitFlag);
  lfp->commit(ie_particleRecyclingFlag);
  lfp->commit(ie_partNumIncreaseFactor);
  lfp->commit(ie_numParticleSplit);
  lfp->commit(ie_partSplittingDosage);
  lfp->commit(ie_taylorMicroscaleMin);
  lfp->commit(ie_randomNumberSeed);
  lfp->commit(ie_timeStep);
  lfp->commit(ie_duration);
  lfp->commit(ie_concAvgTime);
  lfp->commit(ie_concStartTime);
  lfp->commit(ie_partOutputPeriod);
  lfp->commit(ie_nbx);
  lfp->commit(ie_nby);
  lfp->commit(ie_nbz);
  lfp->commit(ie_xbl);
  lfp->commit(ie_xbu);
  lfp->commit(ie_ybl);
  lfp->commit(ie_ybu);
  lfp->commit(ie_zbl);
  lfp->commit(ie_zbu);

  if (beVerbose)
  {
    std::cout << "\tParsing QP_Params file: " << filename << std::endl;
  }
  lfp->study(filename);

  // Check for discovery and default if necessary.		
  if (lfp->recall(ie_sourceTypeFlag))
  {
/*
         if (ie_sourceTypeFlag.value == 1) sourceFlag = qpParams::BASIC;
    else if (ie_sourceTypeFlag.value == 2) sourceFlag = qpParams::DENSEGAS;
    else if (ie_sourceTypeFlag.value == 3) sourceFlag = qpParams::DISTPARTSIZE;
    else if (ie_sourceTypeFlag.value == 4) sourceFlag = qpParams::EXPLOSIVE;
    else if (ie_sourceTypeFlag.value == 5) sourceFlag = qpParams::ERADSOURCE;
    else if (ie_sourceTypeFlag.value == 6) sourceFlag = qpParams::BIOSLURRY;
    else if (ie_sourceTypeFlag.value == 7) sourceFlag = qpParams::TWOPHASE;
    else if (ie_sourceTypeFlag.value == 8) sourceFlag = qpParams::EXFILTRATION;
    else                                	 sourceFlag = qpParams::BASIC;
 */
assert(ie_sourceTypeFlag.value >0 && ie_sourceTypeFlag.value < 9);
sourceFlag = ie_sourceTypeFlag.value;
 }

  isiteflag = (lfp->recall(ie_isiteflag)) ? ie_isiteflag.value : 0;
  iindoorflag = (lfp->recall(ie_iindoorflag)) ? ie_iindoorflag.value : false;
  inextgridflag = (lfp->recall(ie_inextgridflag)) ? ie_inextgridflag.value : 1;
  westernEdge = (lfp->recall(ie_westernEdge)) ? ie_westernEdge.value : 0;
  southernEdge = (lfp->recall(ie_southernEdge)) ? ie_southernEdge.value : 0;
  z0 = (lfp->recall(ie_z0)) ? ie_z0.value : 0;
  rcl = (lfp->recall(ie_rcl)) ? ie_rcl.value : 0;
  boundaryLayerHeight = (lfp->recall(ie_boundaryLayerHeight)) ? ie_boundaryLayerHeight.value : 0;
  nonLocalMixing = (lfp->recall(ie_nonLocalMixing)) ? ie_nonLocalMixing.value : 0;
  useCFDTurbulence = (lfp->recall(ie_useCFDTurbulence)) ? ie_useCFDTurbulence.value : 0;
  numParticles = (lfp->recall(ie_numParticles)) ? ie_numParticles.value : 0;
  particleDistFlag = (lfp->recall(ie_particleDistFlag)) ? ie_particleDistFlag.value : 0;
  particleSplitFlag = (lfp->recall(ie_particleSplitFlag)) ? ie_particleSplitFlag.value : 0;
  particleRecyclingFlag = (lfp->recall(ie_particleRecyclingFlag)) ? ie_particleRecyclingFlag.value : 0;
  partNumIncreaseFactor = (lfp->recall(ie_partNumIncreaseFactor)) ? ie_partNumIncreaseFactor.value : 0;
  numParticleSplit = (lfp->recall(ie_numParticleSplit)) ? ie_numParticleSplit.value : 0;
  partSplittingDosage = (lfp->recall(ie_partSplittingDosage)) ? ie_partSplittingDosage.value : 0;
  taylorMicroscaleMin = (lfp->recall(ie_taylorMicroscaleMin)) ? ie_taylorMicroscaleMin.value : 0;
  randomNumberSeed = (lfp->recall(ie_randomNumberSeed)) ? ie_randomNumberSeed.value : 0;
  timeStep = (lfp->recall(ie_timeStep)) ? ie_timeStep.value : 0;
  duration = (lfp->recall(ie_duration)) ? ie_duration.value : 0;
  concAvgTime = (lfp->recall(ie_concAvgTime)) ? ie_concAvgTime.value : 0;
  concStartTime = (lfp->recall(ie_concStartTime)) ? ie_concStartTime.value : 0;
  partOutputPeriod = (lfp->recall(ie_partOutputPeriod)) ? ie_partOutputPeriod.value : 0;
  nbx = (lfp->recall(ie_nbx)) ? ie_nbx.value : 0;
  nby = (lfp->recall(ie_nby)) ? ie_nby.value : 0;
  nbz = (lfp->recall(ie_nbz)) ? ie_nbz.value : 0;
  xbl = (lfp->recall(ie_xbl)) ? ie_xbl.value : 0;
  xbu = (lfp->recall(ie_xbu)) ? ie_xbu.value : 0;
  ybl = (lfp->recall(ie_ybl)) ? ie_ybl.value : 0;
  ybu = (lfp->recall(ie_ybu)) ? ie_ybu.value : 0;
  zbl = (lfp->recall(ie_zbl)) ? ie_zbl.value : 0;
  zbu = (lfp->recall(ie_zbu)) ? ie_zbu.value : 0;

  delete lfp;
  return true;
}

bool qpParams::writeQUICFile(const std::string &filename)
{
  std::ofstream qpfile;
  qpfile.open(filename.c_str());

  if (qpfile.is_open())
    {
      qpfile << "!QUIC 5.6" << std::endl;

      qpfile << sourceFlag << "\t\t\t!Source type flag (1 = Basic, 2 = Dense Gas, 3 = Distributed Particle Size, 4 = Explosive, 5 = ERAD source, 6 = Bio Slurry, 7 = 2-phase, 8 = Exfiltration)" << std::endl;
      qpfile << isiteflag << "\t\t\t!normal QUIC (isitefl=0) or sensor siting (=1) mode" << std::endl;
      qpfile << iindoorflag << "\t\t\t!indoor calculations turned off (=0) or turned on (=1)" << std::endl;
      qpfile << inextgridflag << "\t\t\t!1 - inner grid, 2 - outer grid" << std::endl;
      qpfile << westernEdge << "\t\t\t!Location of western edge of inner grid relative to outer grid (m)" << std::endl;
      qpfile << southernEdge << "\t\t\t!Location of southern edge of inner relative to outer grid (m)" << std::endl;
      qpfile << z0 << "\t\t\t!Wall Surface Roughness Length (m)" << std::endl;
      qpfile << rcl << "\t\t\t!Reciprocal Monin-Obukhov length(1/m)" << std::endl;
      qpfile << boundaryLayerHeight << "\t\t\t!Boundary Layer height (m)" << std::endl;
      qpfile << nonLocalMixing << "\t\t\t!use 1 to enable non-local mixing" << std::endl;
      qpfile << useCFDTurbulence << "\t\t\t!use 1 to enable use of QUIC-CFD turbulence" << std::endl;
      qpfile << numParticles << "\t\t\t!number of particles released over entire simulation" << std::endl;
      qpfile << particleDistFlag << "\t\t\t!Number of particle distribution flag (1 = by mass, 2 = by source)" << std::endl;
      qpfile << particleSplitFlag << "\t\t\t!Particle splitting flag" << std::endl;
      qpfile << particleRecyclingFlag << "\t\t\t!Particle recycling flag" << std::endl;
      qpfile << partNumIncreaseFactor << "\t\t\t!Total particle number increase factor" << std::endl;
      qpfile << numParticleSplit << "\t\t\t!Number of particles a particle is split into" << std::endl;
      qpfile << partSplittingDosage << "\t\t\t!Particle splitting target dose (gs/m^3)" << std::endl;
      qpfile << taylorMicroscaleMin << "\t\t\t!Enable Taylor microscale lower limit to sub-time steps" << std::endl;
      qpfile << randomNumberSeed << "\t\t\t!Random number seed" << std::endl;
      qpfile << timeStep << "\t\t\t!time step (s)" << std::endl;
      qpfile << duration << "\t\t\t!duration (s)" << std::endl;
      qpfile << concAvgTime << "\t\t\t!concentration averaging time (s)" << std::endl;
      qpfile << concStartTime << "\t\t\t!starting time for concentration averaging (s)" << std::endl;
      qpfile << partOutputPeriod << "\t\t\t!particle output period (s)" << std::endl;
      qpfile << nbx << "\t\t\t!in x direction, # of collecting boxes (concentration grid cells) " << std::endl;
      qpfile << nby << "\t\t\t!in y direction, # of collecting boxes (concentration grid cells) " << std::endl;
      qpfile << nbz << "\t\t\t!in z direction, # of collecting boxes (concentration grid cells) " << std::endl;
      qpfile << xbl << "\t\t\t!lower limits for collecting boxes in x in meters" << std::endl;
      qpfile << xbu << "\t\t\t!upper limits for collecting boxes in x direction in meters" << std::endl;
      qpfile << ybl << "\t\t\t!lower limits for collecting boxes in y in meters" << std::endl;
      qpfile << ybu << "\t\t\t!upper limits for collecting boxes in y direction in meters" << std::endl;
      qpfile << zbl << "\t\t\t!lower limits for collecting boxes in z in meters" << std::endl;
      qpfile << zbu << "\t\t\t!upper limits for collecting boxes in z direction in meters" << std::endl;

      return true;
    }

  return true;
}

