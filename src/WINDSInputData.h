#pragma once

/*
 * A collection of data read from an XML. This contains
 * all root level information extracted from the xml.
 */

#include "util/ParseInterface.h"

#include "SimulationParameters.h"
#include "FileOptions.h"
#include "MetParams.h"
#include "Buildings.h"
#include "Canopies.h"
#include "TURBParams.h"

class WINDSInputData : public ParseInterface
{
public:
    SimulationParameters* simParams;
    FileOptions* fileOptions;
    MetParams* metParams;
    TURBParams* turbParams;
    Buildings* buildings;
    Canopies* canopies;


    WINDSInputData()
    {
	      fileOptions = 0;
	      metParams = 0;
          turbParams=0;
	      buildings = 0;
	      canopies = 0;
    }

    virtual void parseValues()
    {
	     parseElement<SimulationParameters>(true, simParams, "simulationParameters");
	     parseElement<FileOptions>(false, fileOptions, "fileOptions");
	     parseElement<MetParams>(false, metParams, "metParams");
	     parseElement<TURBParams>(false, turbParams,"turbParams");
       parseElement<Buildings>(false, buildings, "buildings");
	     parseElement<Canopies>(false, canopies, "canopies");
    }

    /**
     * @brief Parses the main XML for our QUIC projects.
     *
     * This function initializes the XML structure and parses the main
     * XML file used to represent projects in the QUIC system.
     */
    void parseTree(pt::ptree t)
    {
        setTree(t);
        setParents("root");
        parseValues();
    }
};
