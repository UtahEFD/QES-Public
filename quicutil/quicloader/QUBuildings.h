#ifndef __QUICDATAFILE_QUBUILDINGS_H__
#define __QUICDATAFILE_QUBUILDINGS_H__ 1

#include <iostream>
#include <fstream>
#include <sstream>
#include <cstdlib>
#include <vector>

#include "QUICDataFile.h"

// //////////////////////////////////////////////////////////////////
// 
// Class for holding the QU_buildings.inp file
// 
// //////////////////////////////////////////////////////////////////
class quBuildings : public quicDataFile
{

	
 public:
  quBuildings();
  ~quBuildings() {}
 
  //copy constructor
  quBuildings(const quBuildings& other)
    {
      std::cerr<<"Copy constructor called"<<std::endl;
      *this = other;

    }

  //overloaded assignment
  quBuildings& operator= (const quBuildings& other);
  bool readQUICFile(const std::string &filename);
  bool writeQUICFile(const std::string &filename);

  // know that x_subdomain_sw is an INTType

  int x_subdomain_sw;
  int y_subdomain_sw;
  int x_subdomain_ne;
  int y_subdomain_ne;

  float zo;

  struct buildingData
  {
    int bldNum;
    int group; //                           				     *********TYPE**********
    int type;                                                                  //REGULAR     = 1, 
    float height;	                                                      //CYLINDRICAL = 2, 
    float width;	                                                      //PENTAGON    = 3, 
    float length;	                                                      //VEGETATION  = 9
    float xfo;
    float yfo;
    float zfo;
    float gamma;
    float supplementalData;

    int geometry;  // 5.92 addition
    float centroidX;  // 5.92
    float centroidY;  // 5.92
    float rotation; // 5.92
    float attenuationCoef; // 5.92
      unsigned int numPolys;
  };

  float wallRoughnessLength;   // 5.92 information
    unsigned int numPolygonBuildingNodes;


  std::vector<buildingData> buildings;
 	void build_map();          ///languageMap
  int findIdxByBldNum(int n);
  
private:
};

// How do we want to use this?  Given a name in a file, refer to something in the data structure
// above.
// For example:   From QU_simparams.inp, we might want to modify the domain width... nx
//
// It might nice to reference based on file name and data name... so in the OPT file we might have
//     
// QU_simparams.nx would refer to the nx parameter in QU_simparams.inp
// 
// First, we need a map between filenames and the larger scale data structures for the file
//    std::map<std::string, quicDataFile *>  QUICMaps;
//    upon creation of these types, we'd
//              QUICMaps["QU_simparams"] = new quSimparams( "filename.inp" );
//              ...
// and then when we need nx, we might say
//     quicDataFile *qusimParamPtr;
//     ...
//     qusimParamPtr->retrieveData("nx")   this would get the thing's data for us...   what does this return to us???
//    

#endif // #define __QUICDATAFILE_QUBUILDINGS_H__
