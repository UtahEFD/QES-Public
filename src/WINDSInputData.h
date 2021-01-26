/*
 * QES-Winds
 *
 * Copyright (c) 2021 University of Utah
 * Copyright (c) 2021 University of Minnesota Duluth
 *
 * Copyright (c) 2021 Behnam Bozorgmehr
 * Copyright (c) 2021 Jeremy A. Gibbs
 * Copyright (c) 2021 Fabien Margairaz
 * Copyright (c) 2021 Eric R. Pardyjak
 * Copyright (c) 2021 Zachary Patterson
 * Copyright (c) 2021 Rob Stoll
 * Copyright (c) 2021 Pete Willemsen
 *
 * This file is part of QES-Winds
 *
 * GPL-3.0 License
 *
 * QES-Winds is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, version 3 of the License.
 *
 * QES-Winds is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with QES-Winds. If not, see <https://www.gnu.org/licenses/>.
 *
 */


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

class WINDSInputData : public ParseInterface
{
public:
    SimulationParameters* simParams;
    FileOptions* fileOptions;
    MetParams* metParams;
    Buildings* buildings;
    Canopies* canopies;


    WINDSInputData()
    {
	      fileOptions = 0;
	      metParams = 0;
	      buildings = 0;
	      canopies = 0;
    }

    virtual void parseValues()
    {
	     parseElement<SimulationParameters>(true, simParams, "simulationParameters");
	     parseElement<FileOptions>(false, fileOptions, "fileOptions");
	     parseElement<MetParams>(false, metParams, "metParams");
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
