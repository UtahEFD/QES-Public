//
//  Source.hpp
//  
//  This class represents a generic source
//
//  Created by Jeremy Gibbs on 03/28/19.
//

#ifndef SOURCEKIND_HPP
#define SOURCEKIND_HPP

class SourceKind : public ParseInterface {
    
    protected:
    
    public:
    	virtual void parseValues() = 0;
};
#endif