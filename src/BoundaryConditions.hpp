//
//  BoundaryConditions.hpp
//  
//  This class represents boundary condition types
//
//  Created by Loren Atwood on 01/02/20.
//

#ifndef BOUNDARYCONDITIONS_HPP
#define BOUNDARYCONDITIONS_HPP


#include <string>

#include "util/ParseInterface.h"


class BoundaryConditions : public ParseInterface
{
    
    private:

    public:

        // current possible BCtypes are:
        // "exiting", "periodic", "reflection"

        std::string xBCtype;
        std::string yBCtype;
        std::string zBCtype;
        
        virtual void parseValues() {
    		parsePrimitive<std::string>(true, xBCtype, "xBCtype");
    		parsePrimitive<std::string>(true, yBCtype, "yBCtype");
            parsePrimitive<std::string>(true, zBCtype, "zBCtype");
    	}

};

#endif