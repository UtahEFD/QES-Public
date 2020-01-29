#pragma once


#include "SourceKind.hpp"


class SourceCircle : public SourceKind
{
private:

    // note that this also inherits public data members ReleaseType* m_rType and SourceShape m_sShape.
    //  this also inherits protected data members ReleaseType* rType_instantaneous and ReleaseType* rType_perTimeStep from SourceKind.
    // guidelines for how to set these variables within an inherited source are given in SourceKind.

    double posX;
    double posY;
    double posZ;
    double radius;
    
protected:
    
public:

    // Default constructor
    SourceCircle()
    {
    }
    
    // destructor
    ~SourceCircle()
    {
    }


    virtual void parseValues()
    {
        m_sShape = SourceShape::circle;

        parsePolymorph(false, rType_instantaneous, Polymorph<ReleaseType, ReleaseType_instantaneous>("ReleaseType_instantaneous"));
        //parsePolymorph(false, rType_perSecond, Polymorph<ReleaseType, ReleaseType_rType_perSecond>("ReleaseType_rType_perSecond"));
        parsePolymorph(false, rType_perTimeStep, Polymorph<ReleaseType, ReleaseType_perTimeStep>("ReleaseType_perTimeStep"));
        setReleaseType();
        
        parsePrimitive<double>(true, posX, "posX");
        parsePrimitive<double>(true, posY, "posY");
        parsePrimitive<double>(true, posZ, "posZ");
        parsePrimitive<double>(true, radius, "radius");

    }


    void checkMetaData(const double& domainXstart, const double& domainXend, 
                       const double& domainYstart, const double& domainYend,
                       const double& domainZstart, const double& domainZend);
    
    
    int emitParticles(const float dt,
                      const float currTime,
                      std::vector<particle>& emittedParticles);
    
};
