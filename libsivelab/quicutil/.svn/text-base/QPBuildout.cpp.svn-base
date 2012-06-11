#include <cassert>

#include "QPBuildout.h"

bool qpBuildout::readQUICFile(const std::string &filename)
{
  // 
  // It's special...a common format is needed.
  //
  if (beVerbose)
  {
    std::cout << "\tParsing QP_buildout.inp file: " << filename << std::endl;
  }
  
  std::ifstream bldFile(filename.c_str(), std::ifstream::in);
  if(!bldFile.is_open())
    {
      std::cerr << "quicLoader could not open :: " << filename << "." << std::endl;
      exit(EXIT_FAILURE);
    }
		
  std::string line;
  std::stringstream ss(line, std::stringstream::in | std::stringstream::out);

  // There is no comment in the QP_buildout.inp files with version information... yet!
  // first thing in these files is now a comment with the version information
  // getline(bldFile, line);

  // total number of buildings
  int totalNumBuildings;
  getline(bldFile, line);
  ss.str(line);
  ss >> totalNumBuildings;
		
  // total number of vegetative canopies
  getline(bldFile, line);
  ss.str(line);
  ss >> numVegetativeCanopies;

  std::cerr << "QPBuildout.cpp WARNING: Currently not parsing the vegetative canopies from QP_buildout.inp!" << std::endl;

  // resize the building vector
  buildings.resize(totalNumBuildings);

  // buildings
  for(unsigned int i=0; i<buildings.size(); i++)
  {
    // Building Number =    4
    // Type =    1 gamma =    0.0000
    // Ht =   10.0000 W =   10.0000 L =   10.0000
    // xfo =   15.0000 yfo =   36.0000 zfo =    0.0000
    // Weff =   10.0000 Leff =   10.0000
    // Lf =    8.3333  Lr =   14.5161 Att =    0.0000
    // Sx = 9999.0000 Sy = 9999.0000
    // Building Damage =    0

    char c;
    int bldId;
    std::string token1, token2, token3;

    // reading the building number line...
    getline(bldFile, line);
    ss.str(line);
    ss >> token1 >> token2 >> c;
    if 
    (
      (token1.compare("Building") == 0) && 
      (token2.compare("Number") == 0) && 
      (c == '=')
    )
  	{
	    ss >> bldId;
	  }

    assert( (bldId-1) < totalNumBuildings && bldId >= 0 );
    ss.clear();

    unsigned int bldIdx = bldId - 1;

    // Type and gamma
    getline(bldFile, line);
    ss.str(line);
    ss >> token1 >> c >> buildings[bldIdx].type >> token2 >> c >> buildings[bldIdx].gamma;
    ss.clear();

    // Ht, W, L
    getline(bldFile, line);
    ss.str(line);
    ss >> token1 >> c >> buildings[bldIdx].height 
	     >> token2 >> c >> buildings[bldIdx].width
	     >> token3 >> c >> buildings[bldIdx].length;
    ss.clear();

      // Xfo, yfo, zfo
    getline(bldFile, line);
    ss.str(line);
    ss >> token1 >> c >> buildings[bldIdx].xfo 
	     >> token2 >> c >> buildings[bldIdx].yfo
	     >> token3 >> c >> buildings[bldIdx].zfo;
    ss.clear();

      // Weff, Leff
    getline(bldFile, line);
    ss.str(line);
    ss >> token1 >> c >> buildings[bldIdx].weff 
	     >> token2 >> c >> buildings[bldIdx].leff;
    ss.clear();

    // Lf, Lr, Att
    getline(bldFile, line);
    ss.str(line);
    ss >> token1 >> c >> buildings[bldIdx].lfr 
	     >> token2 >> c >> buildings[bldIdx].lr
	     >> token3 >> c >> buildings[bldIdx].att;
    ss.clear();

      // Sx, Sy
    getline(bldFile, line);
    ss.str(line);
    ss >> token1 >> c >> buildings[bldIdx].sx 
	     >> token2 >> c >> buildings[bldIdx].sy;
    ss.clear();

      // Building Damage
    getline(bldFile, line);
    ss.str(line);
    ss >> token1 >> token2 >> c;
    if ((token1.compare("Building") == 0) && (token2.compare("Damage") == 0) && (c == '='))
	  {
	    ss >> buildings[bldIdx].damage;
	  }
    ss.clear();
  }
  
  if (bldFile.is_open())
  {
    bldFile.close();
  }

  return true;
}

bool qpBuildout::writeQUICFile(const std::string &filename)
{
  std::ofstream qufile;
  qufile.open(filename.c_str());
  if (qufile.is_open())
  {
    for (unsigned int i=0; i<buildings.size(); i++)
  	{
  	  //qufile << buildings[i].bldNum << '\t' << buildings[i].group << '\t' << buildings[i].type << '\t' 
  	  //		 << buildings[i].height << '\t' << buildings[i].width << '\t' << buildings[i].length << '\t' 
  	  //		 << buildings[i].xfo << '\t' << buildings[i].yfo << '\t' << buildings[i].zfo << '\t' 
  	  //<< buildings[i].gamma << '\t' << buildings[i].supplementalData << std::endl;
  	}
      
    return true;
  }

  return true;
}

