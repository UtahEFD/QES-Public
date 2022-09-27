//
//  PlumeInputData.hpp
//
//  This class represents all xml settings
//
//  Created by Jeremy Gibbs on 03/28/19.
//

#ifndef PLUMEINPUTDATA_HPP
#define PLUMEINPUTDATA_HPP


#include "PlumeParameters.hpp"
#include "CollectionParameters.hpp"
#include "ParticleOutputParameters.hpp"
#include "Sources.hpp"
#include "BoundaryConditions.hpp"

#include "util/ParseInterface.h"

#include <boost/foreach.hpp>
#include <boost/property_tree/xml_parser.hpp>
#include <boost/property_tree/ptree.hpp>

class PlumeInputData : public ParseInterface
{

public:
  PlumeParameters *plumeParams = nullptr;
  CollectionParameters *colParams = nullptr;
  ParticleOutputParameters *partOutParams = nullptr;
  Sources *sources = nullptr;
  BoundaryConditions *BCs = nullptr;


  PlumeInputData()
  {
    plumeParams = 0;
    colParams = 0;
    partOutParams = 0;
    sources = 0;
  }

  PlumeInputData(const std::string fileName)
  {
    plumeParams = 0;
    colParams = 0;
    partOutParams = 0;
    sources = 0;

    // read and parse the XML
    parseXML(fileName, "QESPlumeParameters");
  }

  virtual void parseValues()
  {
    parseElement<PlumeParameters>(true, plumeParams, "plumeParameters");
    parseElement<CollectionParameters>(true, colParams, "collectionParameters");
    parseElement<ParticleOutputParameters>(false, partOutParams, "particleOutputParameters");
    parseElement<Sources>(false, sources, "sources");
    parseElement<BoundaryConditions>(true, BCs, "boundaryConditions");
  }
};
#endif
