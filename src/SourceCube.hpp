
#pragma once


#include "SourceKind.hpp"


class SourceCube : public SourceKind
{
private:

    // note that this also inherits public data members ReleaseType* m_rType and SourceShape m_sShape.
    // guidelines for how to set these variables within an inherited source are given in SourceKind.

    double m_minX;
    double m_minY;
    double m_minZ;
    double m_maxX;
    double m_maxY;
    double m_maxZ;
    
protected:
    
public:

    // Default constructor
    SourceCube()
    {
    }

    // destructor
    ~SourceCube()
    {
    }


    virtual void parseValues()
    {
        m_sShape = SourceShape::cube;
        
        setReleaseType();

        parsePrimitive<double>(true, m_minX, "minX");
        parsePrimitive<double>(true, m_minY, "minY");
        parsePrimitive<double>(true, m_minZ, "minZ");
        parsePrimitive<double>(true, m_maxX, "maxX");
        parsePrimitive<double>(true, m_maxY, "maxY");
        parsePrimitive<double>(true, m_maxZ, "maxZ");

    }


    void checkPosInfo( const double& domainXstart, const double& domainXend, 
                       const double& domainYstart, const double& domainYend,
                       const double& domainZstart, const double& domainZend);

    
    int emitParticles(const float dt, const float currTime,
                      std::vector<Particle>& emittedParticles);
    
};
