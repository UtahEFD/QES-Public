#include <iostream>
#include <fstream>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cassert>
#include <algorithm>
#include <signal.h>
#include <math.h>
#include <string>
#include <limits>
#include <sys/stat.h>
#include <sys/types.h>

#include <dirent.h>

#include "quicutil/QUMetParams.h"
#include "quicutil/QUBuildings.h"
#include "quicutil/QUSensor.h"
#include "quicutil/QUSimparams.h"
#include "quicutil/QPParams.h"
#include "quicutil/QPSource.h"

#include "util/ArgumentParsing.h"

class DomainData
{
public:
  DomainData() {}

  static int apronDim;
  static int canyonDim;
  static int bWidth;

};

using namespace std;

void signalHandler(int sig);
void augmentBuildingData( quBuildings &bData, const quSimParams &quSP, const int N, const int M, const int towerHeight );

void copyFile(const std::string &sourceFilename, const std::string &destFilename)
{
  int length;
  char *byteBuffer;

  std::ifstream is;
  is.open(sourceFilename.c_str(), ios::binary);

  if (is.good())
    {
      // get length of file:
      is.seekg(0, ios::end);
      length = is.tellg();
      is.seekg(0, ios::beg);

      // allocate memory:
      byteBuffer = new char[length];

      // read data as a block:
      is.read(byteBuffer,length);
      is.close();

      std::ofstream os;
      os.open(destFilename.c_str(), ios::binary);
      os.write(byteBuffer, length);
      os.close();
    }
  else 
    {
      std::cerr << "Cannot copyFile: unable to open \"" << sourceFilename << "\"." << std::endl;
    }

  return;
}

int generateProjectFiles( const std::string &baseProject, const std::string &newProject, const int bldN, const int bldM, const int towerHeight )
{
  mkdir(newProject.c_str(), S_IRUSR | S_IWUSR | S_IXUSR);

  size_t location = baseProject.find_last_of("/\\");
  std::cout << "baseProject folder: " << baseProject.substr(0, location) << std::endl;
  std::string baseProjectSuffix = baseProject.substr(location+1);
  
  location = newProject.find_last_of("/\\");
  std::cout << "newProject folder: " << newProject.substr(0, location) << std::endl;
  std::string newProjectSuffix = newProject.substr(location+1);
					     
  // create the inner directory
  std::ostringstream outputDirName;
  outputDirName << newProject << "/" << newProjectSuffix << "_inner";
  mkdir(outputDirName.str().c_str(), S_IRUSR | S_IWUSR | S_IXUSR);

  std::cout << "Writing files to " << outputDirName.str() << std::endl;

  std::string outputDir = outputDirName.str();

  // Copy the base project proj file to the new directory
  std::string baseProjName = baseProject + "/" + baseProjectSuffix + ".proj";
  std::string newProjName = newProject + "/" + newProjectSuffix + ".proj";
  copyFile(baseProjName, newProjName);

  std::string quicFilesPath = baseProject + "/" + baseProjectSuffix + "_inner/";

  // 
  // QU_fileoptions.inp...
  // 
  copyFile(quicFilesPath + "QU_fileoptions.inp", outputDir + "/QU_fileoptions.inp");

  // Landuse file
  copyFile(quicFilesPath + "QU_landuse.inp", outputDir + "/QU_landuse.inp");

  //
  // QU_metparams.inp"
  // 
  quMetParams quMetParamData;
  quMetParamData.readQUICFile(quicFilesPath + "QU_metparams.inp");

  // Write this file to the appropriate place.
  quMetParamData.writeQUICFile(outputDir + "/QU_metparams.inp");

  // 
  // QU_simparams.inp
  // 
  quSimParams quSimParamData;
  quSimParamData.readQUICFile(quicFilesPath + "QU_simparams.inp");

  quSimParamData.nx = DomainData::apronDim * 2 + bldN * DomainData::bWidth + (bldN-1) * DomainData::canyonDim;;
  quSimParamData.ny = quSimParamData.nx;

  // domain height should be 10-20 meters higher than the tallest tower...
  quSimParamData.nz = DomainData::bWidth + 10;
  for (int h=0; h<towerHeight; h++)
    quSimParamData.nz += DomainData::bWidth;
  
  // Write this file to the appropriate place.
  quSimParamData.writeQUICFile(outputDir + "/QU_simparams.inp");
  
  quBuildings quBuildingData;
  augmentBuildingData( quBuildingData, quSimParamData, bldN, bldM, towerHeight );
  quBuildingData.writeQUICFile(outputDir + "/QU_buildings.inp");

  // deal with changing wind...
  quSensorParams sensorData;
  sensorData.readQUICFile(quicFilesPath + "sensor1.inp");
  sensorData.writeQUICFile(outputDir + "/sensor1.inp");
  
  copyFile(quicFilesPath + baseProjectSuffix + ".info", outputDir + "/" + newProjectSuffix + ".info");
  copyFile(quicFilesPath + "QP_materials.inp", outputDir + "/QP_materials.inp");      
  copyFile(quicFilesPath + "QP_indoor.inp", outputDir + "/QP_indoor.inp");


  //
  // augment the emitters, if applicable
  //
  qpSource qpSourceData;
  qpSourceData.readQUICFile(quicFilesPath + "QP_source.inp");
  qpSourceData.writeQUICFile(outputDir + "/QP_source.inp");

  copyFile(quicFilesPath + "QP_fileoptions.inp", outputDir + "/QP_fileoptions.inp");

  qpParams qpParamData;
  qpParamData.readQUICFile(quicFilesPath + "QP_params.inp");
  qpParamData.writeQUICFile(outputDir + "/QP_params.inp");

  copyFile(quicFilesPath + "QP_particlesize.inp", outputDir + "/QP_particlesize.inp");

  return 1;
}

int main(int argc, char **argv)
{
  // Setup a signal handler to catch the ^C when program exits
  signal(SIGINT, signalHandler);

  sivelab::ArgumentParsing argParser;
  argParser.reg("version", 'v', no_argument);
  argParser.reg("baseProject", 'b', required_argument);
  argParser.reg("newProject", 'p', required_argument);
  argParser.reg("bldN", 'n', required_argument);
  argParser.reg("bldM", 'm', required_argument);
  argParser.reg("with-towers", 't', required_argument);
  argParser.reg("dx", 'x', required_argument);
  argParser.reg("dy", 'y', required_argument);
  argParser.reg("dz", 'z', required_argument);
  argParser.reg("random", 'r', no_argument);

  argParser.processCommandLineArgs(argc, argv);  

  bool verbose = false;
  if (argParser.isSet("version"))
    {
      std::cout << "autoGenTestCases: version 0.0.1" << std::endl;
      verbose = true;
    }

  std::string baseProject = "";
  if (argParser.isSet("baseProject", baseProject))
    {
      std::cout << "Base Project Name: " << baseProject << std::endl;
    }

  std::string newProject = "";
  if (argParser.isSet("newProject", newProject))
    {
      std::cout << "New Project Name: " << newProject << std::endl;
    }

  std::string argVal;
  int towerHeight = 0;
  if (argParser.isSet("with-towers", argVal))
    {
      // generate towers on buildings
      towerHeight = atoi(argVal.c_str());
    }

  if (baseProject == "" || newProject == "")
    {
      std::cerr << "Must supply a base project and a new project name." << std::endl;
      exit(EXIT_SUCCESS);
    }

  int bldN=1, bldM=1;
  if (argParser.isSet("bldN", argVal))
    {
      bldN = atoi(argVal.c_str());
      bldM = bldN;
    }
  if (argParser.isSet("bldM", argVal))
    {
      bldM = atoi(argVal.c_str());
    }

  float dx = 1.0, dy = 1.0, dz = 1.0;
  if (argParser.isSet("dx", argVal))
    {
      dx = atof(argVal.c_str());
    }
  if (argParser.isSet("dy", argVal))
    {
      dy = atof(argVal.c_str());
    }
  if (argParser.isSet("dz", argVal))
    {
      dz = atof(argVal.c_str());
    }

  std::cout << "Generating " << bldN << " X " << bldM << " buildings." << std::endl;
  if (towerHeight > 0)
    std::cout << "\tBuildings will have " << towerHeight+1 << " levels." << std::endl;

  generateProjectFiles( baseProject, newProject, bldN, bldM, towerHeight );

  exit(EXIT_SUCCESS);
}


void generateBuilding( quBuildings &bData, const int bldIdx, 
		       const float xfo, const float yfo, const float zfo,
		       const float width, const float bHeight, const float length,
		       const int groupNum,
		       int height)
{
  bData.buildings[bldIdx].bldNum = bldIdx + 1;

  bData.buildings[bldIdx].group = groupNum;
  bData.buildings[bldIdx].type = 1;
	
  bData.buildings[bldIdx].height = bHeight;
  bData.buildings[bldIdx].width = width;
  bData.buildings[bldIdx].length = length;

  bData.buildings[bldIdx].xfo = xfo;
  bData.buildings[bldIdx].yfo = yfo;
  bData.buildings[bldIdx].zfo = zfo;
	
  bData.buildings[bldIdx].gamma = 0;
  bData.buildings[bldIdx].supplementalData = 0;

  if (height == 0 || ((bHeight - 2) < 1))
    // done
    return;
  else 
    {
      // need to create space to hold another building--- this is
      // inefficient, but at the moment it doesn't really matter - Pete
      unsigned int currSize = bData.buildings.size();
      bData.buildings.resize(currSize+1);
            
      generateBuilding( bData, currSize, 
			xfo+1, yfo+1, zfo+bHeight, 
			width-2, bHeight-2, length-2,
			bldIdx+1, height-1 );
    }
}

void augmentBuildingData( quBuildings &bData, const quSimParams &quSP, const int N, const int M, const int towerHeight )
{
  // N * M buildings
  bData.buildings.resize(N * M);
  
  bData.x_subdomain_sw = 0;
  bData.y_subdomain_sw = 0;
  bData.x_subdomain_ne = quSP.nx;
  bData.y_subdomain_ne = quSP.ny;

  bData.zo = 0.1;

  for (int i=0; i<N; i++)
    for (int j=0; j<M; j++)
      {
	int bldIdx = j * N + i;

	generateBuilding( bData, bldIdx, 
			  DomainData::apronDim + i * (DomainData::bWidth + DomainData::canyonDim),
			  DomainData::apronDim + DomainData::bWidth/2 + j * (DomainData::bWidth + DomainData::canyonDim),
			  0,
			  DomainData::bWidth, DomainData::bWidth, DomainData::bWidth,
			  bldIdx + 1,
			  towerHeight );
      }
}


void cleanupDirectory( const std::string &dir )
{
  std::string dirn, dotDir = ".", dotdotDir = "..";

  std::cout << "Attempting to delete: " << dir << std::endl;

  DIR *directoryPtr = opendir(dir.c_str());
  if (directoryPtr == 0) 
    {
      std::cout << "Unable to open directory for deletion operations." << std::endl;
      return;
    }

  dirent *dirEntryPtr = 0;
  while ((dirEntryPtr = readdir(directoryPtr)) != 0)
    {
      dirn = dirEntryPtr->d_name;
      if (dirEntryPtr->d_type == DT_DIR)
	{
	  std::string fullDirName = dir + "/" + dirEntryPtr->d_name;

	  if ((dirn != dotDir) && (dirn != dotdotDir))
	    {
	      std::string fullDirName = dir + "/" + dirEntryPtr->d_name;
	      cleanupDirectory(fullDirName);

	      // then remove that directory
	      // std::cout << "Deleting directory: " << fullDirName << std::endl;
	      rmdir(fullDirName.c_str());
	    }
	}
      else 
	{
	  if ((dirn != dotDir) && (dirn != dotdotDir))
	    {
	      std::string fullFileName = dir + "/" + dirEntryPtr->d_name;
	      // std::cout << "Deleting file: " << fullFileName << std::endl;
	      remove( fullFileName.c_str() );
	    }
	}
    }
  closedir(directoryPtr);

  // then remove that directory
  rmdir(dir.c_str());
}

void cleanupFile( const std::string &f )
{
  std::cout << "Attempting to delete: " << f << std::endl;
  remove( f.c_str() );
}

std::string searchForPROJFile(const std::string &dir)
{
  DIR *directoryPtr = opendir(dir.c_str());
  if (directoryPtr == 0) 
    {
      std::cout << "Unable to open directory for search operations." << std::endl;
      return "";
    }

  dirent *dirEntryPtr = 0;
  while ((dirEntryPtr = readdir(directoryPtr)) != 0)
    {
      if (dirEntryPtr->d_type != DT_DIR)
	{
	  std::cout << "Checking file: " << dirEntryPtr->d_name << " for .proj extenstion..." << std::endl;
	  // only search in the last 5 characters
	  std::string dirEntry_name = dirEntryPtr->d_name;

	  size_t found = dirEntry_name.find( ".proj", dirEntry_name.length() - 5 );
	  if (found != std::string::npos)
	    {
	      closedir(directoryPtr);
	      return dirEntry_name;
	    }
	}
    }

  return "";
}


void signalHandler(int sig)
{
  std::cout << "Signal Caught! Cleaning up and exiting." << std::endl;
  exit(EXIT_SUCCESS);
}


int DomainData::apronDim = 30;  // dimension of outskirts around buildings in center
int DomainData::canyonDim = 6;  // dimension for the width of the canyons between buildings
int DomainData::bWidth = 10;

