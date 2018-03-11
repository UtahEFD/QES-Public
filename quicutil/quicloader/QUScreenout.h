#ifndef __QUICDATAFILE_QUSCREENOUT_H__
#define __QUICDATAFILE_QUSCREENOUT_H__ 1

#include "QUICDataFile.h"

class quScreenout : public quicDataFile
{
public:
  quScreenout()
  : quicDataFile()
  {}
  
  ~quScreenout() {}

  bool writeQUICFile(const std::string &filename);

  float Lx;
	float Ly;
	float Lz;

  float dx; 
	float dy; 
	float dz;

  int x_subdomain_start;
	int y_subdomain_start;
	int x_subdomain_end;
	int y_subdomain_end;
	
	
	//buildingData is in QUBuildings
	//std::vector<buildingData> buildings;
	
	
	
private:
}
#endif
