#pragma once

// currently not in use

#include "ParseInterface.h"
#include "Triangle.h"
#include <vector>

class Terrain : public ParseInterface
{
private:
public:
  std::vector<Triangle *> tris;


  virtual void parseValues()
  {
    parseMultiElements<Triangle>(true, tris, "tri");
  }
};
