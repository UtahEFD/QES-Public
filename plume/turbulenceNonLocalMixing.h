#ifndef NONLOCALMIXING
#define NONLOCALMIXING 1

#include <cmath>
#include <vector>
#include <cutil_math.h>

// #include "../../Common/Structures/StopWatch.h"

namespace QUIC
{
  class nonLocalMixing
  {
  public:

    nonLocalMixing(float *windFieldVel) 
    { 
      // set wind field here 
    }

    ~nonLocalMixing();
    
    //non-local mixing
    std::vector<float> dutotdxi,dutotdyi,dutotdzi,dutotdni,dutotdsi;
    std::vector<float> alph1ij,alph2ij,alph3ij,bet1ij,bet2ij,bet3ij,gam1ij,gam2ij,gam3ij,alphn1ij;
    std::vector<float> alphn2ij,alphn3ij,betn1ij,betn2ij,betn3ij,gamn1ij,gamn2ij,gamn3ij,ani,bni,cni;
    std::vector<float> ufsqgi,vfsqgi,wfsqgi,ufvfgi,ufwfgi,vfwfgi;
    //non-localmixing ends
  



    void mix(unsigned int,unsigned int, unsigned int, unsigned int,unsigned int,unsigned int);

    bool calcSinOmegaCosOmega( float& snmg, float& csmg, float& dutotdn, int const& iomega, 
			       float const& sinpsi,  float const& cospsi,
			       float const& sinphiw, float const& cosphiw,
			       float3 const& dutot );

    void detang(const int, const int, float&, float&, int&,int&,int&);
      
    void rotu3psq(const int&,const float&,const float&,const float&,const float&,
		  const float&,const float& ,const float& ,float&,float&,float&,
		  float&,float&,float&);
    
    void rotufsq(const int&,float&,float&,float&,float&,const float&,const float&,
		 const float&,const float&,const float&,const float&);
      
    void rotate2d
    (
     const int&,const float&,const float&,const float&,const float&,
     const float&,const float&,const float&,const float&
     );
      
    float sign(const float&,const float&);
    
    int nint(const int t1) { return int(t1 + .5f); }
    int nint(const float t1) { return int(t1 + .5f); }

  protected:
    nonLocalMixing() {}

  private:
    float4 *wind_vel;
    
  };
}

#endif
