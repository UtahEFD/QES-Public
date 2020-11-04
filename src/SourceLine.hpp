
#pragma once


#include "SourceKind.hpp"


class SourceLine : public SourceKind
{
private:

    // note that this also inherits public data members ReleaseType* m_rType and SourceShape m_sShape.
    // guidelines for how to set these variables within an inherited source are given in SourceKind.

    double posX_0;
    double posY_0;
    double posZ_0;
    double posX_1;
    double posY_1;
    double posZ_1;
    
protected:
    
public:

    // Default constructor
    SourceLine()
    {
    }
    
    // destructor
    ~SourceLine()
    {
    }


    virtual void parseValues()
    {
        m_sShape = SourceShape::line;

        setReleaseType();
        
        parsePrimitive<double>(true, posX_0, "posX_0");
        parsePrimitive<double>(true, posY_0, "posY_0");
        parsePrimitive<double>(true, posZ_0, "posZ_0");
        parsePrimitive<double>(true, posX_1, "posX_1");
        parsePrimitive<double>(true, posY_1, "posY_1");
        parsePrimitive<double>(true, posZ_1, "posZ_1");
        
    }


    void checkPosInfo( const double& domainXstart, const double& domainXend, 
                       const double& domainYstart, const double& domainYend,
                       const double& domainZstart, const double& domainZend);

    
    int emitParticles(const float dt, const float currTime,
                      std::list<Particle*>& emittedParticles);
    
};
