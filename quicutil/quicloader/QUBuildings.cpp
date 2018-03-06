#include "QUBuildings.h"

#include <cassert>
void quBuildings::build_map(){

  ///building the map
  //buildings.reserve(20);          ////this could cause problems
           
  var_addressMap["x_subdomain_sw"]=&x_subdomain_sw;
  var_addressMap["y_subdomain_sw"]=&y_subdomain_sw;
  var_addressMap["x_subdomain_ne"]=&x_subdomain_ne;
  var_addressMap["y_subdomain_ne"]=&y_subdomain_ne;
  var_addressMap["zo"]=&zo;
  var_addressMap["buildings[]"]=sizeof(buildingData);         //size of the structure 
  var_addressMap["buildings.bldNum"]=&buildings[0].bldNum;
  var_addressMap["buildings.group"]=&buildings[0].group;
	    
  var_addressMap["buildings.type"]=&buildings[0].type;
  var_addressMap["buildings.height"]=&buildings[0].height;
	
  //  std::cout<<"The address of the height varaible : --------------------------------------------------------------"<<&buildings[0].height<<"-----------------------------"<<std::endl;



	    
  var_addressMap["buildings.width"]=&buildings[0].width;
  var_addressMap["buildings.length"]=&buildings[0].length;
	    
  var_addressMap["buildings.xfo"]=&buildings[0].xfo;
  var_addressMap["buildings.yfo"]=&buildings[0].yfo;
	    
  var_addressMap["buildings.zfo"]=&buildings[0].zfo;
  var_addressMap["buildings.gamma"]=&buildings[0].gamma;
  var_addressMap["buildings.supplementalData"]=&buildings[0].supplementalData;
  // var_addressMap["test_string"]=&test_string; //test remove it


}

quBuildings& quBuildings::operator=(const quBuildings& other)
{

  //  std::cerr<<"operator ---------quBuildings---------"<<std::endl;
  if (this == &other)
    return *this;



  x_subdomain_sw=other.x_subdomain_sw;
  y_subdomain_sw=other.y_subdomain_sw;
  x_subdomain_ne=other.x_subdomain_ne;
  y_subdomain_ne=other.y_subdomain_ne;

  zo=other.zo;
  buildings.reserve( other.buildings.capacity());
  copy(other.buildings.begin(),other.buildings.end(), std::back_inserter(buildings));
  return * this;
}


quBuildings::quBuildings() 
  : quicDataFile()
{
  x_subdomain_sw = 0;
  y_subdomain_sw = 0;
  x_subdomain_ne   = 0;
  y_subdomain_ne   = 0;
  zo = 0.;
	
  //buildings.resize(0); // declared


  
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
  std::string comment, version;
  ss.str(line);
  ss >> comment >> version;

  // std::cout << "version: " << version << std::endl;

  if (version == "5.92" || version == "6.01" || version == "6.1")
    {
      // In 5.92, the building structures resemble:
      // 
      // !Start Building 1
      // 1			!Group ID
      // 1			!Geometry = Rectangular
      // 1			!Building Type = Solid
      // 12			!Height [m]
      // 0			!Base Height (Zfo) [m]
      // 702.5			!Centroid X [m]
      // 665			!Centroid Y [m]
      // 680			!Xfo [m]
      // 665			!Yfo [m]
      // 45			!Length [m]
      // 50			!Width [m]
      // 0			!Rotation [deg]
      // !End Building 1

      // First, need to read off the wall roughness, number of buildings, and number of polygon building nodes
      
      // wall roughness
      getline(bldFile, line);
      ss.str(line);
      ss >> wallRoughnessLength;
      ss.clear();

      // std::cout << "wall roughness: " << wallRoughnessLength << std::endl;

      // num buildings
      int numBuildings;
      getline(bldFile, line);
      ss.str(line);
      ss >> numBuildings;
      ss.clear();

      // std::cout << "num buildings: " << numBuildings << std::endl;

      // number of polygon building nodes
      getline(bldFile, line);
      ss.str(line);
      ss >> numPolygonBuildingNodes;
      ss.clear();

      // std::cout << "num polys: " << numPolygonBuildingNodes << std::endl;

      // resize the building vector
      buildings.resize(numBuildings);

      // std::cout << "qu buildings size: " << buildings.size() << std::endl;

      // buildings
      for(unsigned int i=0; i<buildings.size(); i++)
	{
	  buildings[i].bldNum = i + 1;

	  // First line for each building is something like this:
	  // !Start Building 1
	  getline(bldFile, line);
	  // std::cout << "Start: " << line << std::endl;

	  // group
	  getline(bldFile, line);
	  ss.str(line);
	  ss >> buildings[i].group;
	  ss.clear();

	  // geometry
	  getline(bldFile, line);
	  ss.str(line);
	  ss >> buildings[i].geometry;
	  ss.clear();

	  // type 
	  getline(bldFile, line);
	  ss.str(line);
	  ss >> buildings[i].type;
	  ss.clear();

	  // std::cout << "G: " << buildings[i].group << ", g: " << buildings[i].geometry << ", t:" << buildings[i].type << std::endl;

	  if (buildings[i].type == 2)  // 			!Building Type = Canopy
	    {
	      // Read the     2.68			!Attenuation Coefficient
	      getline(bldFile, line);
	      ss.str(line);
	      ss >> buildings[i].attenuationCoef;
	      ss.clear();
	    }

	  // height
	  getline(bldFile, line);
	  ss.str(line);
	  ss >> buildings[i].height;
	  ss.clear();

	  // zfo
	  getline(bldFile, line);
	  ss.str(line);
	  ss >> buildings[i].zfo;
	  ss.clear();

	  // centroid x
	  getline(bldFile, line);
	  ss.str(line);
	  ss >> buildings[i].centroidX;
	  ss.clear();

	  // centroid y
	  getline(bldFile, line);
	  ss.str(line);
	  ss >> buildings[i].centroidY;
	  ss.clear();

	  // std::cout << "H: " << buildings[i].height << ", zfo: " << buildings[i].zfo << ", cX:" << buildings[i].centroidX << ", cY:" << buildings[i].centroidY << std::endl;

	// Geometry determines how the next elements are parsed!
	if (buildings[i].geometry == 1 ||
	    buildings[i].geometry == 2) 
	  {
	    // Xfo
	    getline(bldFile, line);
	    ss.str(line);
	    ss >> buildings[i].xfo;
	    ss.clear();
	    
	    // yfo
	    getline(bldFile, line);
	    ss.str(line);
	    ss >> buildings[i].yfo;
	    ss.clear();
	    
	    // length
	    getline(bldFile, line);
	    ss.str(line);
	    ss >> buildings[i].length;
	    ss.clear();

	    // width
	    getline(bldFile, line);
	    ss.str(line);
	    ss >> buildings[i].width;
	    ss.clear();

	    // rotation
	    getline(bldFile, line);
	    ss.str(line);
	    ss >> buildings[i].rotation; 
	    ss.clear();

	    // std::cout << "xfo: " << buildings[i].xfo 
	    // << ", yfo: " << buildings[i].yfo 
	    // << ", length:" << buildings[i].length 
	    // << ", width:" << buildings[i].width 
	    // << ", rotation:" << buildings[i].rotation << std::endl;
	  }
	else if (buildings[i].geometry == 6)
	  {
	    // 1			!Number of Polygons
	    // !Start Polygon 1
	    // 5			!Number of Nodes
	    // !X [m]   Y [m]
	    // 645	560
	    // 645	530
	    // 630	535
	    // 630	560
	    // 645	560
	    // !End Polygon 1

	    // num polys
	    getline(bldFile, line);
	    ss.str(line);
	    ss >> buildings[i].numPolys;
	    ss.clear();	    

	    for (unsigned int pCount=0; pCount<buildings[i].numPolys; pCount++) 
	      {

		// Read over the "!Start Polygon" line
		getline(bldFile, line);

		// Read number of nodes
		unsigned int nNodes;
		getline(bldFile, line);
		ss.str(line);
		ss >> nNodes;
		ss.clear();	    

		// Read over the "!X [m]   Y [m]" line
		getline(bldFile, line);

		for (unsigned int nCount=0; nCount<nNodes; nCount++) 
		  {
		    getline(bldFile, line);
		    ss.str(line);
		
		    float xC, yC;
		    ss >> xC >> yC;

		    ss.clear();	    		
		  }

		// Read over the "!End Polygon" line
		getline(bldFile, line);
	      }
	  }

	// Read over the "!End Building" line
	getline(bldFile, line);
	//	std::cout << "End: " << line << std::endl;
	}
      
      // -------------------------
      // END version 5.92/6.1
      // -------------------------
    }
  else {
    
    // x subdomain (southwest corner)
    getline(bldFile, line);
    ss.str(line);
    ss >> x_subdomain_sw;
    ss.clear();	    
		
    // y subdomain (southwest corner)
    getline(bldFile, line);
    ss.str(line);
    ss >> y_subdomain_sw;
    ss.clear();	    

    // x subdomain (northeast corner)
    getline(bldFile, line);
    ss.str(line);
    ss >> x_subdomain_ne;
    ss.clear();	    
		
    // y subdomain (northeast corner)
    getline(bldFile, line);
    ss.str(line);
    ss >> y_subdomain_ne;
    ss.clear();	    
		
    // wall roughness
    getline(bldFile, line);
    ss.str(line);
    ss >> zo;
    ss.clear();	    
		
    // number of buildings
    getline(bldFile, line);
    ss.str(line);
    int numbuilds = 0;
    ss >> numbuilds;
    ss.clear();	    

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
	ss >> buildings[i].bldNum >> buildings[i].group >> buildings[i].type;
	ss >> buildings[i].height >> buildings[i].width >> buildings[i].length;
	ss >> buildings[i].xfo >> buildings[i].yfo >> buildings[i].zfo;
	ss >> buildings[i].gamma >> buildings[i].supplementalData;
	ss.clear();

	if( buildings[i].type == 6 ){ buildings[i].type = 1; }

	assert(buildings[i].type ==1 || buildings[i].type == 2 || buildings[i].type == 3 ||buildings[i].type == 9);
      
      }
  
    bldFile.close();
  }

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
