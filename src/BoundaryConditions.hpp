//
//  BoundaryConditions.hpp
//  
//  This class represents boundary condition types
//
//  Created by Loren Atwood on 01/02/20.
//  Modified by Loren Atwood on 03/24/2020.
//

#ifndef BOUNDARYCONDITIONS_HPP
#define BOUNDARYCONDITIONS_HPP


#include <string>

#include "util/ParseInterface.h"


class BoundaryConditions : public ParseInterface
{
    
    private:

    public:

        // current possible domainBCtypes are:
        // "exiting", "periodic", "reflection"
        // where this is assumed to always occur using the cell face normal
        // these are checked in the Plume constructor 
        // as pointer functions for the boundary conditions are assigned
        // if "periodic" is chosen for one domain direction BC type, it has to be chosen for the opposing too
        std::string xDomainStartBCtype;
        std::string yDomainStartBCtype;
        std::string zDomainStartBCtype;
        std::string xDomainEndBCtype;
        std::string yDomainEndBCtype;
        std::string zDomainEndBCtype;

        // current possible icellflag BC types are
        // 0  "Building"      == "simpleStairStep"
        // 1  "Fluid"         == "passthrough"
        // 2  "Terrain"       == "simpleStairStep"
        // 3  "Upwind cavity" == "passthrough"
        // 4  "Cavity"        == "passthrough"
        // 5  "Farwake"       == "passthrough"
        // 6  "Street canyon" == "passthrough"
        // 7  "Building cut-cells" == "simpleStairStep", // not yet but soon: "simpleCutCell", "normalCutCell"
        // 8  "Terrain cut-cells"  == "simpleStairStep", // not yet but soon: "simpleCutCell", "normalCutCell"
        // 9  "Sidewall"      == "passthrough"
        // 10 "Rooftop"       == "passthrough"
        // 11 "Canopy vegetation"  == "passthrough", // when adding in depositions, will be passthrough or canopy deposition
        // 12 "Fire"          == "passthrough"
        // to do this, the following variables need set, with the following options:
        // "buildingCutCell_reflectionType" == "simpleStairStep", // not yet but soon: "simpleCutCell", "normalCutCell"
        // "terrainCutCell_reflectionType"  == "simpleStairStep", // not yet but soon: "simpleCutCell", "normalCutCell"
        std::string buildingCutCell_reflectionType;
        std::string terrainCutCell_reflectionType;

        // whether to do depositions or no
        bool doDepositions;
        
        virtual void parseValues()
        {
            // first parse all the values
    		parsePrimitive<std::string>(true, xDomainStartBCtype, "xDomainStartBCtype");
    		parsePrimitive<std::string>(true, yDomainStartBCtype, "yDomainStartBCtype");
            parsePrimitive<std::string>(true, zDomainStartBCtype, "zDomainStartBCtype");
            parsePrimitive<std::string>(true, xDomainEndBCtype,   "xDomainEndBCtype");
    		parsePrimitive<std::string>(true, yDomainEndBCtype,   "yDomainEndBCtype");
            parsePrimitive<std::string>(true, zDomainEndBCtype,   "zDomainEndBCtype");

            parsePrimitive<std::string>(true, buildingCutCell_reflectionType, "buildingCutCell_reflectionType");
            parsePrimitive<std::string>(true, terrainCutCell_reflectionType,  "terrainCutCell_reflectionType");

            parsePrimitive<bool>(true, doDepositions, "doDepositions");

            // now do a few checks on the values
            checkParsedValues();
    	}

        // LA-note: because the use is actually done in the dispersion class when setting the boundary condition function pointers,
        //  care needs taken to make sure the specific strings to check here match those of the dispersion class methods
        void checkParsedValues()
        {
            // make sure domainBCtypes that are set to periodic on one end are set to periodic on both ends
            if( xDomainStartBCtype == "periodic" && xDomainEndBCtype != "periodic" )
            {
                std::cerr << "(BoundaryConditions::checkParsedValues): input xDomainEndBCtype must be set to \"periodic\" if input xDomainStartBCtype is set to \"periodic\"!";
                std::cerr << " xDomainStartBCtype = \"" << xDomainStartBCtype << "\", xDomainEndBCtype = \"" << xDomainEndBCtype << "\"" << std::endl;
                exit(EXIT_FAILURE);
            }
            if( xDomainEndBCtype == "periodic" && xDomainStartBCtype != "periodic" )
            {
                std::cerr << "(BoundaryConditions::checkParsedValues): input xDomainStartBCtype must be set to \"periodic\" if input xDomainEndBCtype is set to \"periodic\"!";
                std::cerr << " xDomainStartBCtype = \"" << xDomainStartBCtype << "\", xDomainEndBCtype = \"" << xDomainEndBCtype << "\"" << std::endl;
                exit(EXIT_FAILURE);
            }
            if( yDomainStartBCtype == "periodic" && yDomainEndBCtype != "periodic" )
            {
                std::cerr << "(BoundaryConditions::checkParsedValues): input yDomainEndBCtype must be set to \"periodic\" if input yDomainStartBCtype is set to \"periodic\"!";
                std::cerr << " yDomainStartBCtype = \"" << yDomainStartBCtype << "\", yDomainEndBCtype = \"" << yDomainEndBCtype << "\"" << std::endl;
                exit(EXIT_FAILURE);
            }
            if( yDomainEndBCtype == "periodic" && yDomainStartBCtype != "periodic" )
            {
                std::cerr << "(BoundaryConditions::checkParsedValues): input yDomainStartBCtype must be set to \"periodic\" if input yDomainEndBCtype is set to \"periodic\"!";
                std::cerr << " yDomainStartBCtype = \"" << yDomainStartBCtype << "\", yDomainEndBCtype = \"" << yDomainEndBCtype << "\"" << std::endl;
                exit(EXIT_FAILURE);
            }
            if( zDomainStartBCtype == "periodic" && zDomainEndBCtype != "periodic" )
            {
                std::cerr << "(BoundaryConditions::checkParsedValues): input zDomainEndBCtype must be set to \"periodic\" if input zDomainStartBCtype is set to \"periodic\"!";
                std::cerr << " zDomainStartBCtype = \"" << zDomainStartBCtype << "\", zDomainEndBCtype = \"" << zDomainEndBCtype << "\"" << std::endl;
                exit(EXIT_FAILURE);
            }
            if( zDomainEndBCtype == "periodic" && zDomainStartBCtype != "periodic" )
            {
                std::cerr << "(BoundaryConditions::checkParsedValues): input zDomainStartBCtype must be set to \"periodic\" if input zDomainEndBCtype is set to \"periodic\"!";
                std::cerr << " zDomainStartBCtype = \"" << zDomainStartBCtype << "\", zDomainEndBCtype = \"" << zDomainEndBCtype << "\"" << std::endl;
                exit(EXIT_FAILURE);
            }
        }

};

#endif