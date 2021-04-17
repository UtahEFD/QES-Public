#pragma once

#include "ParseInterface.h"

namespace pt = boost::property_tree;

class GeneralInputData : public ParseInterface
{
public:
  GeneralInputData(pt::ptree t)
  {
    tree = t;
    treeParents = "root";
  }


  virtual void parseValues() = 0;
};
