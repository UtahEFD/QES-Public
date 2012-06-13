#ifndef NONLOCALMIXING
#define NONLOCALMIXING 1

#include <cmath>
#include <vector>
#include <cutil_inline.h>
#include <cutil_math.h>

namespace QUIC
{
  class nonLocalMixing
  {
  public:

    nonLocalMixing(QUICProject *qpProj, float *windFieldVel) 
    { 
      // set wind field here 

      // set quic pointer here too
      // instead of m_util_ptr->, we want m_QPPtr->nx  or
      m_QPPtr = qpProj;
    }

    ~nonLocalMixing();
    
    std::vector<float> dutotdxi,dutotdyi,dutotdzi,dutotdni,dutotdsi;
    std::vector<float> alph1ij,alph2ij,alph3ij,bet1ij,bet2ij,bet3ij,gam1ij,gam2ij,gam3ij,alphn1ij;
    std::vector<float> alphn2ij,alphn3ij,betn1ij,betn2ij,betn3ij,gamn1ij,gamn2ij,gamn3ij,ani,bni,cni;
    std::vector<float> ufsqgi,vfsqgi,wfsqgi,ufvfgi,ufwfgi,vfwfgi;

    void mix(unsigned int,unsigned int, unsigned int, unsigned int,unsigned int,unsigned int);

    bool calcSinOmegaCosOmega( float& snmg, float& csmg, float& dutotdn, int const& iomega, 
			       float const& sinpsi,  float const& cospsi,
			       float const& sinphiw, float const& cosphiw,
			       float3 const& dutot );

    void detang(const int iomega,const int &id, float &dutotds,float &dutotdn,int &i,int &j,int &k);
      
    void rotu3psq(const int &id,const float &u3psq,const float &utot,const float &upvp,const float &upwp,
		  const float &vpwp,const float &v3psq,const float &w3psq,float &ufsqb,float &wfsqb,float &vfsqb,
		  float &ufvf,float &ufwf,float &vfwf);

    void rotufsq(const int &id,float &u3psq,float &upwp,
		 float &v3psq,float &w3psq,const float &ufsq,const float &ufvf,
		 const float &ufwf,const float &vfsq,const float &vfwf,const float &wfsq);

    void rotate2d(const int &id,const float &cosphi,const float &sinphi,const float &upsqg,
		  const float &upvpg,const float &vpsqg,const float &wpsqg,const float &upwpg,const float &vpwpg );

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
