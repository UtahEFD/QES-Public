//
//  Source.hpp
//  
//  This class represents a generic source
//
//  Created by Jeremy Gibbs on 03/28/19.
//

#ifndef SOURCEKIND_HPP
#define SOURCEKIND_HPP


#include "pointInfo.hpp"


class SourceKind : public ParseInterface {
    
    protected:
    
    public:

        //initializer
        SourceKind()
        {
        }

        // destructor
        virtual ~SourceKind()
        {
        }

    
    	virtual void parseValues() = 0;

        // hm, I just wanted a copy of the data, but I'm not sure if I can do that, I might have to deal with pointer stuff
        // see the idea is I don't want this list except temporarily for the sources, but I want it permanently for the dispersion class
        // but is this a valid construct or do I have to output a pointer to the class?
        // I thought about getting around this by putting a pointInfoList storage in to each source by putting said storage to here
        // but again I run into similar problems, unless I delete the values of the list after the output function call
        // but the problem there is that the whole point is to get the output from the source, deleting the values of the list after the
        // output function call has to be done AFTER the function call, not during. Would mean a manual call of a delete function
        // which isn't gauranteed to be done every time. Would be smarter just to deal with pointers correctly
        //  So probably need to output a pointer to a pointInfoList class, which the caller of this function just has to be smart about deleting
        // after they are done. More confusing logic than calling a delete function, but this would mean passing around fewer values so more efficient
        // oh what the heck, I'm just going to do the easier way for now, a list of values for each source that is carefully cleaned up each time
        // actually, can completely avoid an external function call to clean if it is called every time by 
        virtual std::vector<pointInfo> outputPointInfo(const double& dt,const double& simDur)
        {
            // just output the value, but empty so it doesn't get used in the user functions
            std::vector<pointInfo> outputVal;
            return outputVal;
        }

};
#endif