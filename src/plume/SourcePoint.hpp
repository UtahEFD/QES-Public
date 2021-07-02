
#pragma once


#include "SourceKind.hpp"


class SourcePoint : public SourceKind
{
private:

    // note that this also inherits public data members ReleaseType* m_rType and SourceShape m_sShape.
    // guidelines for how to set these variables within an inherited source are given in SourceKind.

    double posX;
    double posY;
    double posZ;
    
protected:
    
public:

    // Default constructor
    SourcePoint()
    {
    }

    // destructor
    ~SourcePoint()
    {
    }


    virtual void parseValues()
    {
        m_sShape = SourceShape::point;

        setReleaseType();

        parsePrimitive<double>(true, posX, "posX");
        parsePrimitive<double>(true, posY, "posY");
        parsePrimitive<double>(true, posZ, "posZ");
    }


    void checkPosInfo( const double& domainXstart, const double& domainXend, 
                       const double& domainYstart, const double& domainYend,
                       const double& domainZstart, const double& domainZend);
    
    
    int emitParticles(const float dt, const float currTime,
                      std::list<Particle*>& emittedParticles);
    
};
