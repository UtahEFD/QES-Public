#pragma once


#include "SourceKind.hpp"


class SourceLine : public SourceKind
{
private:

    // note that this also inherits public data members ReleaseType* m_rType and SourceShape m_sShape.
    //  this also inherits protected data members ReleaseType* rType_instantaneous and ReleaseType* rType_perTimeStep from SourceKind.
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

        parsePolymorph(false, rType_instantaneous, Polymorph<ReleaseType, ReleaseType_instantaneous>("ReleaseType_instantaneous"));
        //parsePolymorph(false, rType_perSecond, Polymorph<ReleaseType, ReleaseType_perSecond>("ReleaseType_perSecond"));
        parsePolymorph(false, rType_perTimeStep, Polymorph<ReleaseType, ReleaseType_perTimeStep>("ReleaseType_perTimeStep"));
        setReleaseType();
        
        parsePrimitive<double>(true, posX_0, "posX_0");
        parsePrimitive<double>(true, posY_0, "posY_0");
        parsePrimitive<double>(true, posZ_0, "posZ_0");
        parsePrimitive<double>(true, posX_1, "posX_1");
        parsePrimitive<double>(true, posY_1, "posY_1");
        parsePrimitive<double>(true, posZ_1, "posZ_1");
        
    }


    void checkMetaData(const double& domainXstart, const double& domainXend, 
                       const double& domainYstart, const double& domainYend,
                       const double& domainZstart, const double& domainZend);

    
    int emitParticles(const float dt,
                      const float currTime,
                      std::vector<particle>& emittedParticles);
    
};
