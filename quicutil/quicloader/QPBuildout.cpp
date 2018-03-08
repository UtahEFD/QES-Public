#include <cassert>

#include "QPBuildout.h"

qpBuildout& qpBuildout::operator=(const qpBuildout& other)
{

  //  std::cerr<<"operator ---------qpBuildout---------"<<std::endl;
  if (this == &other)
    return *this;

  numVegetativeCanopies=other.numVegetativeCanopies;




   
  buildings.reserve( other.buildings.capacity());
  copy(other.buildings.begin(),other.buildings.end(), std::back_inserter(buildings));
  return * this;
}
void qpBuildout::build_map()
{

	 var_addressMap["numVegetativeCanopies"]=&numVegetativeCanopies;  //integer

	  var_addressMap["buildings[]"]=sizeof(buildingOutData);      
	  var_addressMap["buildings.type"]=&buildings[0].type;
	  var_addressMap["buildings.gamma"]=&buildings[0].gamma;
	
  	  var_addressMap["buildings.height"]=&buildings[0].height;
	  var_addressMap["buildings.width"]=&buildings[0].width;
  	  var_addressMap["buildings.length"]=&buildings[0].length;

	  var_addressMap["buildings.xfo"]=&buildings[0].xfo;
    	  var_addressMap["buildings.yfo"]=&buildings[0].yfo;
	  var_addressMap["buildings.zfo"]=&buildings[0].zfo;

	  var_addressMap["buildings.weff"]=&buildings[0].weff;
	  var_addressMap["buildings.leff"]=&buildings[0].leff;

	  var_addressMap["buildings.lfr"]=&buildings[0].lfr;
	  var_addressMap["buildings.lr"]=&buildings[0].lr;
	  var_addressMap["buildings.att"]=&buildings[0].att;

	  var_addressMap["buildings.sx"]=&buildings[0].sx;
	  var_addressMap["buildings.sy"]=&buildings[0].sy;

	  var_addressMap["buildings.damage"]=&buildings[0].damage;
	 
   

}
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

  // If quicVersionString == "5.92"
  // need to read 
  //     36  ! total number of polygon nodes
  if (quicVersionString == "5.92") 
    {
      getline(bldFile, line);
      ss.str(line);
      ss >> numPolygonNodes;
    }

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

    int bldId;

    // And 6.1 needs to be added here!!!!
    if (quicVersionString == "5.92") 
      {
	// For version 5.92, the format is
	//              292  !number
	//            1  !geometry
	//            1  !type
	//    3.000000      !height
	//    0.000000      !zfo
	//            0  !damage
	//    5.320889      !Weff
	//    10.64178      !Leff
	//   -999.0000      !Lf
	//    3.323769      !Lr
	//    640.0000      !xfo
	//    575.0000      !yfo
	//    5.000000      !length
	//    10.00000      !width
	//    0.000000      !gamma
	
	// reading the building number line...
	getline(bldFile, line);
	ss.str(line);
	ss >> bldId;
	ss.clear();

	unsigned int bldIdx = bldId - 1;

	// geometry
	getline(bldFile, line);
	ss.str(line);
	ss >> buildings[bldIdx].geometry;
	ss.clear();

	// type 
	getline(bldFile, line);
	ss.str(line);
	ss >> buildings[bldIdx].type;
	ss.clear();

	// height
	getline(bldFile, line);
	ss.str(line);
	ss >> buildings[bldIdx].height;
	ss.clear();

	// zfo
	getline(bldFile, line);
	ss.str(line);
	ss >> buildings[bldIdx].zfo;

	// damage
	getline(bldFile, line);
	ss.str(line);
	ss >> buildings[bldIdx].damage;
	ss.clear();
	
	// Weff
	getline(bldFile, line);
	ss.str(line);
	ss >> buildings[bldIdx].weff;
	ss.clear();

	// Leff
	getline(bldFile, line);
	ss.str(line);
	ss >> buildings[bldIdx].leff;
	ss.clear();

	// Lf
	getline(bldFile, line);
	ss.str(line);
	ss >> buildings[bldIdx].lfr;
	ss.clear();

	// Lr
	getline(bldFile, line);
	ss.str(line);
	ss >> buildings[bldIdx].lr;
	ss.clear();

	// Geometry determines how the next elements are parsed!
	if (buildings[bldIdx].geometry == 1 ||
	    buildings[bldIdx].geometry == 2) 
	  {
	    // Xfo
	    getline(bldFile, line);
	    ss.str(line);
	    ss >> buildings[bldIdx].xfo;
	    ss.clear();
	    
	    // yfo
	    getline(bldFile, line);
	    ss.str(line);
	    ss >> buildings[bldIdx].yfo;
	    ss.clear();
	    
	    // length
	    getline(bldFile, line);
	    ss.str(line);
	    ss >> buildings[bldIdx].length;
	    ss.clear();

	    // width
	    getline(bldFile, line);
	    ss.str(line);
	    ss >> buildings[bldIdx].width;
	    ss.clear();

	    // gamma
	    getline(bldFile, line);
	    ss.str(line);
	    ss >> buildings[bldIdx].gamma; 
	    ss.clear();
	  }
	else if (buildings[bldIdx].geometry == 6)
	  {
	    //    709.8910      !xc
	    //520.9290      !yc
	    //      19  !start
	    // 24  !stop
            // 1  !num polygons
	    // 745.0000        550.0000    
	    // 745.0000        495.0000    
	    // 670.0000        495.0000    
	    // 670.0000        525.0000    
	    // 695.0000        550.0000    
	    // 745.0000        550.0000    
	    
	    // xc
	    getline(bldFile, line);
	    ss.str(line);
	    ss >> buildings[bldIdx].xc;
	    ss.clear();
	    
	    // yc
	    getline(bldFile, line);
	    ss.str(line);
	    ss >> buildings[bldIdx].yc;
	    ss.clear();
	    
	    // start
	    getline(bldFile, line);
	    ss.str(line);
	    ss >> buildings[bldIdx].start;
	    ss.clear();

	    // stop
	    getline(bldFile, line);
	    ss.str(line);
	    ss >> buildings[bldIdx].stop;
	    ss.clear();

	    // num polys
	    getline(bldFile, line);
	    ss.str(line);
	    ss >> buildings[bldIdx].numPolys;
	    ss.clear();	    

	    // Apparently, the difference between stop and start + 1
	    // determines the number of 2D vertices to be read.
	    int maxVertices = buildings[bldIdx].stop - buildings[bldIdx].start + 1;
	    for (unsigned int vCount=0; vCount<maxVertices; vCount++) 
	      {
		getline(bldFile, line);
		ss.str(line);
		
		float xC, yC;
		ss >> xC >> yC;

		ss.clear();	    		
	      }
	  }

	// -------------------------
	// END version 5.92
	// -------------------------
      }
    else 
      {
	// Non 5.92 versions, likely prior to 5.92

    char c;

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

    //    assert( (bldId-1) < totalNumBuildings && bldId >= 0 );
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
	qufile << '\t' << buildings.size() << "\t! total number of buildings" << std::endl;
        qufile << '\t' << numVegetativeCanopies << "\t! total number of vegitative canopies" << std::endl;
        qufile << '\t' << numPolygonNodes << "\t! total number of polygon nodes" << std::endl;
      
        for (int i = buildings.size()-1; i>=0; i--)
	{
            qufile << '\t' << i+1 << "\t!number" << std::endl;
            qufile << '\t' << buildings[i].geometry << "\t!geometry" << std::endl;
            qufile << '\t' << buildings[i].type << "\t!type" << std::endl;
            qufile << '\t' << buildings[i].height << "\t!height" << std::endl;
            qufile << '\t' << buildings[i].zfo << "\t!zfo" << std::endl;
            qufile << '\t' << buildings[i].damage << "\t!damage" << std::endl;
            qufile << '\t' << buildings[i].weff << "\t!Weff" << std::endl;
            qufile << '\t' << buildings[i].leff << "\t!Leff" << std::endl;
            qufile << '\t' << buildings[i].lfr << "\t!Lf" << std::endl;
            qufile << '\t' << buildings[i].lr << "\t!Lr" << std::endl;
            qufile << '\t' << buildings[i].xc << "\t!xc" << std::endl;
            qufile << '\t' << buildings[i].yc << "\t!yc" << std::endl;
            qufile << '\t' << buildings[i].start << "\t!start" << std::endl;
            qufile << '\t' << buildings[i].stop << "\t!stop" << std::endl;
            qufile << '\t' << buildings[i].numPolys << "\t!num polygons" << std::endl;
	}
        
 
      return true;
    }

  return true;

}

