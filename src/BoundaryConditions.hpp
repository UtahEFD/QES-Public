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
  
    // possible reflection methods:
    /*
     * "doNothing"             - nothing happen when particle enter wall
     * "setInactive" (default) - particle is set to inactive when entering a wall
     * "stairstepReflection"   - particle use full stair step reflection when entering a wall
     */
    
    std::string wallReflection;
    
    virtual void parseValues() {
        parsePrimitive<std::string>(true, xBCtype, "xBCtype");
        parsePrimitive<std::string>(true, yBCtype, "yBCtype");
        parsePrimitive<std::string>(true, zBCtype, "zBCtype");
        
        
        wallReflection = "";
        parsePrimitive<std::string>(false, wallReflection, "wallReflection");
        
        if( wallReflection == "" ) {
            wallReflection = "setInactive";
        }
    }
    
};

#endif
