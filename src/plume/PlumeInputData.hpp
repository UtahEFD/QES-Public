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
  PlumeParameters *plumeParams;
  CollectionParameters *colParams;
  ParticleOutputParameters *partOutParams;
  Sources *sources;
  BoundaryConditions *BCs;


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

    parseXML(fileName, "QESPlumeParameters");
    /*
    pt::ptree tree;

    try {
      pt::read_xml(fileName, tree);
    } catch (boost::property_tree::xml_parser::xml_parser_error &e) {
      std::cerr << "Error reading tree in" << fileName << "\n";
      std::cerr << "QES-Plume input file was not able to be read successfully." << std::endl;
      exit(EXIT_FAILURE);
    }

    parseTree(tree);
    */
  }

  virtual void parseValues()
  {
    parseElement<PlumeParameters>(true, plumeParams, "plumeParameters");
    parseElement<CollectionParameters>(true, colParams, "collectionParameters");
    parseElement<ParticleOutputParameters>(false, partOutParams, "particleOutputParameters");

    parseElement<Sources>(false, sources, "sources");
    parseElement<BoundaryConditions>(true, BCs, "boundaryConditions");
  }
  /**
     * This function takes in an URBInputData variable and uses it
     * as the base to parse the ptree
     * @param UID the object that will serve as the base level of the xml parser
     */
  void parseTree(pt::ptree t)
  {
    //setTree(t);
    //setParents("root");
    auto child = t.get_child_optional("QESPlumeParameters");
    setTree(*child);
    setParents("root::QESPlumeParameters");
    parseValues();
  }
};
#endif
