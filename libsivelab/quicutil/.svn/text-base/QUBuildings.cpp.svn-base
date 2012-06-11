#include "QUBuildings.h"

#include <cassert>

quBuildings::quBuildings() 
: quicDataFile()
{
  x_subdomain_sw = 0;
  y_subdomain_sw = 0;
	x_subdomain_ne   = 0;
	y_subdomain_ne   = 0;
	zo = 0.;
	
	buildings.resize(0); // declared
}

bool quBuildings::readQUICFile(const std::string &filename)
{
  // 
  // It's special...a common format is needed.
  //
  if (beVerbose)
  { 
    std::cout << "\tParsing QU_buildings.inp file: " << filename << std::endl;
  }
  
  std::ifstream bldFile(filename.c_str(), std::ifstream::in);
  if(!bldFile.is_open())
  {
    std::cerr << "quicLoader could not open :: " << filename << "." << std::endl;
    exit(EXIT_FAILURE);
  }
		
  std::string line;
  std::stringstream ss(line, std::stringstream::in | std::stringstream::out);

  // first thing in these files is now a comment with the version information
  getline(bldFile, line);

  // x subdomain (southwest corner)
  getline(bldFile, line);
  ss.str(line);
  ss >> x_subdomain_sw;
		
  // y subdomain (southwest corner)
  getline(bldFile, line);
  ss.str(line);
  ss >> y_subdomain_sw;

  // x subdomain (northeast corner)
  getline(bldFile, line);
  ss.str(line);
  ss >> x_subdomain_ne;
		
  // y subdomain (northeast corner)
  getline(bldFile, line);
  ss.str(line);
  ss >> y_subdomain_ne;
		
  // wall roughness
  getline(bldFile, line);
  ss.str(line);
  ss >> zo;
		
  // number of buildings
  getline(bldFile, line);
  ss.str(line);
  int numbuilds = 0;
  ss >> numbuilds;

  // resize the building vector
  buildings.resize(numbuilds);

  // building description !Bld #	Group	Type	Height	Width	Length	Xfo	Yfo	Zfo	Gamma	Attenuation	Values in grid cell units
  //						!1	1	1	10	48	49	37	63	0	0	0
  getline(bldFile, line);
		
  // buildings
  int currBuildingType;
  for(int i = 0; i < numbuilds; i++)
    {
      getline(bldFile, line);
      ss.str(line);
      ss >> buildings[i].bldNum >> buildings[i].group >> currBuildingType;
      ss >> buildings[i].height >> buildings[i].width >> buildings[i].length;
      ss >> buildings[i].xfo >> buildings[i].yfo >> buildings[i].zfo;
      ss >> buildings[i].gamma >> buildings[i].supplementalData;
      ss.clear();
      
      switch(currBuildingType)
      {
        case REGULAR:     
          buildings[i].type = REGULAR; 
          break;
		    case CYLINDRICAL: 
		      buildings[i].type = CYLINDRICAL; 
		      break;
		    case PENTAGON:    
		      buildings[i].type = PENTAGON; 
		      break;
		    case VEGETATION:  
		      buildings[i].type = VEGETATION; 
		      break;
		    default:
		      std::cerr << "  Unknown building type: " << currBuildingType << std::endl;
		      assert(buildings[i].type == -1);
		      exit(EXIT_FAILURE);
		      break;
      }
    }
  
  bldFile.close();

  return true;
}

bool quBuildings::writeQUICFile(const std::string &filename)
{
  std::ofstream qufile;
  qufile.open(filename.c_str());
  if (qufile.is_open())
    {
      // !!!!! different versions here!!! qufile << "!QUIC 5.51" << std::endl;
      qufile << "!QUIC 5.72" << std::endl;

      qufile << x_subdomain_sw << "\t\t\t!x subdomain coordinate (southwest corner) (Cells)" << std::endl;
      qufile << y_subdomain_sw << "\t\t\t!y subdomain coordinate (southwest corner) (Cells)" << std::endl;
      qufile << x_subdomain_ne << "\t\t\t!x subdomain coordinate (northeast corner) (Cells)" << std::endl;
      qufile << y_subdomain_ne << "\t\t\t!y subdomain coordinate (northeast corner) (Cells)" << std::endl;
      qufile << zo << "\t\t\t!Wall roughness length (m)" << std::endl;

      qufile << buildings.size() << "\t\t\t!Number of Structures" << std::endl;
      qufile << "!Bld #	Group	Type	Height(m)	Width(m)	Length(m)	Xfo(m)	Yfo(m)	Zfo(m)	Gamma	Suplemental Data" << std::endl;
     for (unsigned int i=0; i<buildings.size(); i++)
	{
	  qufile << buildings[i].bldNum << '\t' << buildings[i].group << '\t' << buildings[i].type << '\t' 
		 << buildings[i].height << '\t' << buildings[i].width << '\t' << buildings[i].length << '\t' 
		 << buildings[i].xfo << '\t' << buildings[i].yfo << '\t' << buildings[i].zfo << '\t' 
		 << buildings[i].gamma << '\t' << buildings[i].supplementalData << std::endl;
	}
      
      return true;
    }

  return true;
}

int quBuildings::findIdxByBldNum(int n)
{
  for (unsigned int i=0; i<buildings.size(); i++)
    {
      if (buildings[i].bldNum == n)
	return i;
    }
  return -1;
}
