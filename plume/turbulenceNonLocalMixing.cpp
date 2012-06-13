#include "turbulenceNonLocalMixing.h"

namespace QUIC
{
  /*
  float nonLocalMixing::sign(float const& A,float const &B)
  {
    return (B >= 0.f) ? fabs(A) : -fabs(A);	
  }
  */
  
  bool nonLocalMixing::calcSinOmegaCosOmega( float& snmg, float& csmg, float& dutotdn, int const& iomega, 
					     float const& sinpsi,  float const& cospsi,
					     float const& sinphiw, float const& cosphiw,
					     float3 const& dutot )
  {
    float cosomeg, sinomeg, omeg;
    
    if(iomega == 0)
    {
      float omeden =   dutot.x*cospsi*sinphiw 
                     + dutot.y*sinpsi*sinphiw 
                     - dutot.z*cosphiw;
    
      if(fabs(omeden) < 1.e-10)
      {
        cosomeg = 0.f;
        sinomeg = 1.f;
        dutotdn =   dutot.x*(sinpsi*sinomeg - cospsi*sinphiw*cosomeg) 
                  - dutot.y*(cospsi*sinomeg + sinpsi*sinphiw*cosomeg) 
                  + dutot.z*cosomeg*cosphiw;

        if(dutotdn < 0.f) {sinomeg = -1.f;}
      }
      else
      {
        float omenum = -dutot.x*sinpsi + dutot.y*cospsi;
        
        // Tangent of omega
		    float omeg1    = atan(omenum / omeden);
        float dutotdn1 =   dutot.x*(sinpsi*sin(omeg1) - cospsi*sinphiw*cos(omeg1)) 
                         - dutot.y*(cospsi*sin(omeg1) + sinpsi*sinphiw*cos(omeg1)) 
                         + dutot.z*cos(omeg1)*cosphiw;
        
        float omeg2 = omeg1 + M_PI;
        float dutotdn2 =   dutot.x*(sinpsi*sin(omeg2) - cospsi*sinphiw*cos(omeg2)) 
                         - dutot.y*(cospsi*sin(omeg2) + sinpsi*sinphiw*cos(omeg2)) 
                         + dutot.z*cos(omeg2)*cosphiw;
        
        dutotdn = (dutotdn2 > dutotdn1) ? dutotdn2   : dutotdn1 ;
        omeg    = (dutotdn2 > dutotdn1) ? omeg2      : omeg1 ;
        
        cosomeg = cos(omeg);
        sinomeg = sin(omeg);
      }
    }
    else
    {
      if(iomega != 1 && fabs(cospsi) <= .5f) {return false;}

      omeg    = (iomega == 1) ?      0.f : 
                (iomega == 3) ?    -omeg : 
                                M_PI / 2.f ;
                                
      cosomeg = (iomega == 1) ? 1.f : 0.f ;
      
      sinomeg = (iomega == 1) ?           0.f : 
                (iomega == 3) ?  fabs(cospsi) :
                                -fabs(cospsi) ;
    }
    
    return true;
  }



  void nonLocalMixing::detang(const int iomega,const int &id, float &dutotds,float &dutotdn,int &i,int &j,int &k)
  {
    float  e11,e12,e13,e21,e22,e23,e31,e32,e33;
    float cospsi,sinpsi,sinphiw,cosphiw,omenum,omeden,cosomeg,sinomeg,tanomeg,omeg1,cosomeg1,sinomeg1;
    float omeg2,cosomeg2,sinomeg2,dutotdn2,dutotdn1,pi,omeg;

	pi=4.*atan(1.0);
    
    if(sqrt(wind_vel[id].u*wind_vel[id].u+wind_vel[id].v*wind_vel[id].v)>1.e-05){
        cospsi=wind_vel[id].u/(sqrt(wind_vel[id].u*wind_vel[id].u+wind_vel[id].v*wind_vel[id].v));
        sinpsi=wind_vel[id].v/(sqrt(wind_vel[id].u*wind_vel[id].u+wind_vel[id].v*wind_vel[id].v));
        sinphiw=wind_vel[id].w/(sqrt(wind_vel[id].u*wind_vel[id].u+wind_vel[id].v*wind_vel[id].v+wind_vel[id].w*wind_vel[id].w)+1.e-10);
        cosphiw=sqrt(wind_vel[id].u*wind_vel[id].u+wind_vel[id].v*wind_vel[id].v)/(sqrt(wind_vel[id].u*wind_vel[id].u+wind_vel[id].v*wind_vel[id].v+wind_vel[id].w*wind_vel[id].w)+1.e-10);
        omenum=-dutotdxi.at(id)*sinpsi+dutotdyi.at(id)*cospsi;
        omeden=dutotdxi.at(id)*cospsi*sinphiw+dutotdyi.at(id)*sinpsi*sinphiw-dutotdzi.at(id)*cosphiw;
        if(iomega==0){
            if(fabs(omeden)<1.e-10){
                cosomeg=0.;
                sinomeg=1.;
                  dutotdn=dutotdxi.at(id)*(sinpsi*sinomeg-cospsi*sinphiw*cosomeg) 
                       -dutotdyi.at(id)*(cospsi*sinomeg+sinpsi*sinphiw*cosomeg) 
                      +dutotdzi.at(id)*cosomeg*cosphiw;
                  if(dutotdn<0.)sinomeg=-1.;
            }
            else{
                tanomeg=omenum/omeden;
				omeg1=atan(tanomeg);
                cosomeg1=cos(omeg1);
                sinomeg1=sin(omeg1);
                dutotdn1=dutotdxi.at(id)*(sinpsi*sinomeg1-cospsi*sinphiw*cosomeg1) 
                    -dutotdyi.at(id)*(cospsi*sinomeg1+sinpsi*sinphiw*cosomeg1) 
                    +dutotdzi.at(id)*cosomeg1*cosphiw;
                omeg2=omeg1+pi;
                cosomeg2=cos(omeg2);
                sinomeg2=sin(omeg2);
                dutotdn2=dutotdxi.at(id)*(sinpsi*sinomeg2-cospsi*sinphiw*cosomeg2) 
                    -dutotdyi.at(id)*(cospsi*sinomeg2+sinpsi*sinphiw*cosomeg2) 
                    +dutotdzi.at(id)*cosomeg2*cosphiw;
                if(dutotdn2>dutotdn1){
                    dutotdn=dutotdn2;
                    omeg=omeg2;
                    cosomeg=cosomeg2;
                    sinomeg=sinomeg2;
                }
                else{
                    dutotdn=dutotdn1;
                    omeg=omeg1;
                    cosomeg=cosomeg1;
                    sinomeg=sinomeg1;
                }
                

            }
        }
        else{
            if(iomega==1){
                omeg=0.;
                cosomeg=1.;
                sinomeg=0.;
            }
            else{
                if(fabs(cospsi)>0.5){
                    omeg=pi/2.;
                    cosomeg=0.;
					sinomeg=-sign(cospsi,1.0);
                    if(iomega==3){
                        sinomeg=sign(cospsi,1.0);
                        omeg=-omeg;
                    }
                }
                else{
                    return;
                }
            }
        }
        alph1ij.at(id)=cospsi*cosphiw;
        alph2ij.at(id)=-sinpsi*cosomeg-cospsi*sinphiw*sinomeg;
        alph3ij.at(id)=sinpsi*sinomeg-cospsi*sinphiw*cosomeg;
        bet1ij.at(id)=sinpsi*cosphiw;
        bet2ij.at(id)=cospsi*cosomeg-sinpsi*sinphiw*sinomeg;
        bet3ij.at(id)=-cospsi*sinomeg-sinpsi*sinphiw*cosomeg;
        gam1ij.at(id)=sinphiw;
        gam2ij.at(id)=cosphiw*sinomeg;
        gam3ij.at(id)=cosphiw*cosomeg;
        alphn1ij.at(id)=cospsi*cosphiw;
        alphn2ij.at(id)=sinpsi*cosphiw;
        alphn3ij.at(id)=sinphiw;
        betn1ij.at(id)=-sinpsi*cosomeg-cospsi*sinphiw*sinomeg;
        betn2ij.at(id)=cospsi*cosomeg-sinpsi*sinphiw*sinomeg;
        betn3ij.at(id)=cosphiw*sinomeg;
        gamn1ij.at(id)=sinpsi*sinomeg-cospsi*sinphiw*cosomeg;
        gamn2ij.at(id)=-cospsi*sinomeg-sinpsi*sinphiw*cosomeg;
        gamn3ij.at(id)=cosphiw*cosomeg;
        dutotdn=dutotdxi.at(id)*(sinpsi*sinomeg-cospsi*sinphiw*cosomeg) 
            -dutotdyi.at(id)*(cospsi*sinomeg+sinpsi*sinphiw*cosomeg) 
            +dutotdzi.at(id)*cosomeg*cosphiw;
        
        dutotds=dutotdxi.at(id)*cospsi*cosphiw+dutotdyi.at(id)*
            sinpsi*cosphiw+dutotdzi.at(id)*sinphiw;
        dutotdni.at(id)=dutotdn;
        dutotdsi.at(id)=dutotds;
        ani.at(id)=(sinpsi*sinomeg-cospsi*sinphiw*cosomeg);
        bni.at(id)=-(cospsi*sinomeg+sinpsi*sinphiw*cosomeg);
        cni.at(id)=cosomeg*cosphiw;
    }
    else{
        if(fabs(wind_vel[id].w)<1.e-05){
            if(sqrt(dutotdxi.at(id)*dutotdxi.at(id)+dutotdyi.at(id)*dutotdyi.at(id))>0.){
                cospsi=dutotdxi.at(id)/(sqrt(dutotdxi.at(id)*dutotdxi.at(id)+dutotdyi.at(id)*dutotdyi.at(id)));
                sinpsi=dutotdyi.at(id)/(sqrt(dutotdxi.at(id)*dutotdxi.at(id)+dutotdyi.at(id)*dutotdyi.at(id)));
            }
            else{
                cospsi=1.;
                sinpsi=0.;
            }
            if(sqrt(dutotdxi.at(id)*dutotdxi.at(id)+dutotdyi.at(id)*dutotdyi.at(id)+dutotdzi.at(id)*dutotdzi.at(id))>0){
                cosphiw=dutotdzi.at(id)/(sqrt(dutotdxi.at(id)*dutotdxi.at(id)
                                              +dutotdyi.at(id)*dutotdyi.at(id)+dutotdzi.at(id)*dutotdzi.at(id)));
                  sinphiw=sqrt(dutotdxi.at(id)*dutotdxi.at(id)+dutotdyi.at(id)*dutotdyi.at(id))
                      /(sqrt(dutotdxi.at(id)*dutotdxi.at(id)+dutotdyi.at(id)*dutotdyi.at(id)+dutotdzi.at(id)*dutotdzi.at(id)));
            }
            else{
                cosphiw=1.;
                sinphiw=0.;
            }
            alphn1ij.at(id)=cospsi*cosphiw;
            alphn2ij.at(id)=sinpsi*cosphiw;
            alphn3ij.at(id)=-sinphiw;
            betn1ij.at(id)=-sinpsi;
            betn2ij.at(id)=cospsi;
            betn3ij.at(id)=0.;
            gamn1ij.at(id)=sinphiw*cospsi;
            gamn2ij.at(id)=sinphiw*sinpsi;
            gamn3ij.at(id)=cosphiw;
            alph1ij.at(id)=cospsi*cosphiw;
            alph2ij.at(id)=-sinpsi;
            alph3ij.at(id)=sinphiw*cospsi;
            bet1ij.at(id)=sinpsi*cosphiw;
            bet2ij.at(id)=cospsi;
            bet3ij.at(id)=sinphiw*sinpsi;
            gam1ij.at(id)=-sinphiw;
            gam2ij.at(id)=0.;
            gam3ij.at(id)=cosphiw;
            dutotdni.at(id)=sqrt(dutotdxi.at(id)*dutotdxi.at(id)+dutotdyi.at(id)*dutotdyi.at(id)+dutotdzi.at(id)*dutotdzi.at(id));
            dutotdsi.at(id)=0.;
            ani.at(id)=sinphiw*cospsi;
            bni.at(id)=sinphiw*sinpsi;
            cni.at(id)=cosphiw;
        }
        else{
            if(wind_vel[id].w>0.){
                if(sqrt(dutotdxi.at(id)*dutotdxi.at(id)+dutotdyi.at(id)*dutotdyi.at(id))>0.){
                    cospsi=dutotdxi.at(id)/sqrt(dutotdxi.at(id)*dutotdxi.at(id)+dutotdyi.at(id)*dutotdyi.at(id));
                    sinpsi=dutotdyi.at(id)/sqrt(dutotdxi.at(id)*dutotdxi.at(id)+dutotdyi.at(id)*dutotdyi.at(id));
                }
                else{
                    cospsi=1.;
                    sinpsi=0.;
                }
                alph1ij.at(id)=0.;
                alph2ij.at(id)=sinpsi;
                alph3ij.at(id)=cospsi;
                bet1ij.at(id)=0.;
                bet2ij.at(id)=-cospsi;
                bet3ij.at(id)=sinpsi;
                gam1ij.at(id)=1.;
                gam2ij.at(id)=0.;
                gam3ij.at(id)=0.;
                alphn1ij.at(id)=0.;
                alphn2ij.at(id)=0.;
                alphn3ij.at(id)=1.;
                betn1ij.at(id)=sinpsi;
                betn2ij.at(id)=-cospsi;
                betn3ij.at(id)=0.;
                gamn1ij.at(id)=cospsi;
                gamn2ij.at(id)=sinpsi;
                gamn3ij.at(id)=0.;
                dutotdni.at(id)=sqrt(dutotdxi.at(id)*dutotdxi.at(id)+dutotdyi.at(id)*dutotdyi.at(id));
                dutotdni.at(id)=std::max(1.e-12f,dutotdni.at(id));
                
                dutotdsi.at(id)=dutotdzi.at(id);
                ani.at(id)=cospsi;
                bni.at(id)=sinpsi;
                cni.at(id)=0.;
            }
            else{
                if(sqrt(dutotdxi.at(id)*dutotdxi.at(id)+dutotdyi.at(id)*dutotdyi.at(id))>0.){
                    cospsi=dutotdxi.at(id)/sqrt(dutotdxi.at(id)*dutotdxi.at(id)+dutotdyi.at(id)*dutotdyi.at(id));
                    sinpsi=dutotdyi.at(id)/sqrt(dutotdxi.at(id)*dutotdxi.at(id)+dutotdyi.at(id)*dutotdyi.at(id));
                }
                else{
                    cospsi=1.;
                    sinpsi=0.;
                  }
                alphn1ij.at(id)=0.;
                alphn2ij.at(id)=0.;
                alphn3ij.at(id)=-1.;
                betn1ij.at(id)=-sinpsi;
                betn2ij.at(id)=cospsi;
                betn3ij.at(id)=0.;
                gamn1ij.at(id)=cospsi;
                gamn2ij.at(id)=sinpsi;
                gamn3ij.at(id)=0.;
                alph1ij.at(id)=0.;
                alph2ij.at(id)=-sinpsi;
                alph3ij.at(id)=cospsi;
                bet1ij.at(id)=0.;
                bet2ij.at(id)=cospsi;
                bet3ij.at(id)=sinpsi;
                gam1ij.at(id)=-1.;
                gam2ij.at(id)=0.;
                gam3ij.at(id)=0.;
                dutotdni.at(id)=sqrt(dutotdxi.at(id)*dutotdxi.at(id)+dutotdyi.at(id)*dutotdyi.at(id));
                
                dutotdsi.at(id)=-dutotdzi.at(id);
            }
        }
    }
    e11=alph1ij.at(id)*alphn1ij.at(id)+alph2ij.at(id)*betn1ij.at(id)+alph3ij.at(id)*gamn1ij.at(id);
    e12=alph1ij.at(id)*alphn2ij.at(id)+alph2ij.at(id)*betn2ij.at(id)+alph3ij.at(id)*gamn2ij.at(id);
    e13=alph1ij.at(id)*alphn3ij.at(id)+alph2ij.at(id)*betn3ij.at(id)+alph3ij.at(id)*gamn3ij.at(id);
    e21=bet1ij.at(id)*alphn1ij.at(id)+bet2ij.at(id)*betn1ij.at(id)+bet3ij.at(id)*gamn1ij.at(id);
    e22=bet1ij.at(id)*alphn2ij.at(id)+bet2ij.at(id)*betn2ij.at(id)+bet3ij.at(id)*gamn2ij.at(id);
    e23=bet1ij.at(id)*alphn3ij.at(id)+bet2ij.at(id)*betn3ij.at(id)+bet3ij.at(id)*gamn3ij.at(id);
    e31=gam1ij.at(id)*alphn1ij.at(id)+gam2ij.at(id)*betn1ij.at(id)+gam3ij.at(id)*gamn1ij.at(id);
    e32=gam1ij.at(id)*alphn2ij.at(id)+gam2ij.at(id)*betn2ij.at(id)+gam3ij.at(id)*gamn2ij.at(id);
    e33=gam1ij.at(id)*alphn3ij.at(id)+gam2ij.at(id)*betn3ij.at(id)+gam3ij.at(id)*gamn3ij.at(id);
    
}


#if 0 // Andrew's version
  // Orignally 253 - 16 = 237 lines
  void nonLocalMixing::detang(const int iomega, const int id, 
			      float& dutotds, float& dutotdn, int& i, int& j, int& k)
  {
    float3 dutot = {dutotdxi.at(id), dutotdyi.at(id), dutotdzi.at(id)};

    // Lateral Magnitude of wind velocity.
    float ltrl_mgntd = sqrt(wind_vel[id].x*wind_vel[id].x + wind_vel[id].y*wind_vel[id].y);
    // TODO Give this a better name. // Total Lateral magnitude of change in position???
    float dnmntr = sqrt(dutot.x*dutot.x + dutot.y*dutot.y);

    float cospsi = (lMgZero) ? wind_vel[id].x / ltrl_mgntd : 
                          (dnmntr > 0.f) ? dutot.x / dnmntr : 
                                           1.f ;
                                     
    float sinpsi = (lMgZero) ? wind_vel[id].y / ltrl_mgntd ;
                          (dnmntr > 0.f) ? dutot.y / dnmntr : 
                                           0.f ; 

    /////////////////////////////////////////////////////////////////////////////
    float mgntd_dutotd = mag(dutot);

    float sinphiw = (lMgZero) ? wind_vel[id].z / (mag(wind_vel[id]) + 1.e-10) :
                     (mgntd_dutotd > 0.f) ? dnmntr / mgntd_dutotd : 
                                            0.f ;
                     
    float cosphiw = (lMgZero) ? ltrl_mgntd / (mag(wind_vel[id]) + 1.e-10) : 
                     (mgntd_dutotd > 0.f) ? dutot.z / mgntd_dutotd : 
                                            1.f ;

     
    bool calcOkay = calcSinOmegaCosOmega
                    (
                      sinomeg, cosomeg, dutotdn, 
                      iomega, sinpsi, cospsi, 
                      sinphiw, cosphiw, dutot
                    );
    if(!calcOkay) {return};
    
    bool wIsPstv = (wind_vel[id].z > 0.f);
    bool wIsZero = (fabs(wind_vel[id].z) < 1.e-05); 
    bool lMgZero = (ltrl_mgntd > 1.e-05);
    
    float w_sgn = (wIsPstv) ? 1.f : -1.f ;
      
    alph1ij.at(id) =   (lMgZero || wIsZero) ?  cospsi*cosphiw : 0.f ;
    alph2ij.at(id) =              (lMgZero) ? -sinpsi*cosomeg - cospsi*sinphiw*sinomeg :
                      (wIsZero || !wIsPstv) ? -sinpsi : sinpsi ;
    alph3ij.at(id) =              (lMgZero) ?  sinpsi*sinomeg - cospsi*sinphiw*cosomeg :
                                  (wIsZero) ?  sinphiw*cospsi : 
                                               cospsi ;
    
    bet1ij.at(id) =  (lMgZero || wIsZero) ?  sinpsi*cosphiw : 0.f ;
    bet2ij.at(id) =             (lMgZero) ?  cospsi*cosomeg - sinpsi*sinphiw*sinomeg :
                    (wIsZero || !wIsPstv) ?  cospsi : -cospsi ;
    bet3ij.at(id) =             (lMgZero) ? -cospsi*sinomeg - sinpsi*sinphiw*cosomeg :
                                (wIsZero) ?  sinphiw*sinpsi : 
                                             sinpsi ;
    
    gam1ij.at(id) = (lMgZero) ?  sinphiw : 
                    (wIsZero) ? -sinphiw :
                                 sgn_w;
    gam2ij.at(id) = (lMgZero) ?  cosphiw*sinomeg : 0.f ;
    gam3ij.at(id) = (lMgZero) ?  cosphiw*cosomeg : 
                    (wIsZero) ?  cosphiw : 
                                 0.f ;
    
    alphn1ij.at(id) = (lMgZero || wIsZero) ?  cospsi*cosphiw : 0.f ;
    alphn2ij.at(id) = (lMgZero || wIsZero) ?  sinpsi*cosphiw : 0.f ;
    alphn3ij.at(id) =            (lMgZero) ?  sinphiw :
                                 (wIsZero) ? -sinphiw : 
                                              sgn_w;
                                              
    // TODO Clean up this horrid shit...                                          
    float3 alphnij = {alphn1ij.at(id), alphn2ij.at(id), alphn3ij.at(id)};
    
    betn1ij.at(id) =             (lMgZero) ? -sinpsi*cosomeg - cospsi*sinphiw*sinomeg :
                     (wIsZero || !wIsPstv) ? -sinpsi : sinpsi ;
    betn2ij.at(id) =             (lMgZero) ?  cospsi*cosomeg - sinpsi*sinphiw*sinomeg :
                     (wIsZero || !wIsPstv) ?  cospsi : -cospsi ;
    betn3ij.at(id) =             (lMgZero) ? cosphiw*sinomeg : 0.f ;
    
    
    gamn1ij.at(id) = (lMgZero) ?  sinpsi*sinomeg - cospsi*sinphiw*cosomeg :
                     (wIsZero) ?  sinphiw*cospsi :
                                  cospsi ;
    gamn2ij.at(id) = (lMgZero) ? -cospsi*sinomeg - sinpsi*sinphiw*cosomeg :
                     (wIsZero) ?  sinphiw*sinpsi :
                                  sinpsi ;
    gamn3ij.at(id) = (lMgZero) ?  cosphiw*cosomeg : 
                     (wIsZero) ?  cosphiw : 
                                  0.f ;

    // TODO Make all this shit float3...
    float3 gamnij = {gamn1ij.at(id), gamn2ij.at(id), gamn3ij.at(id)};                      
    
    ani.at(id) = (lMgZero || wIsZero || wIsPstv) ? gamnij.x : ani.at(id) ;
    bni.at(id) = (lMgZero || wIsZero || wIsPstv) ? gamnij.y : bni.at(id) ;
    cni.at(id) = (lMgZero || wIsZero || wIsPstv) ? gamnij.z : cni.at(id) ;
                          
    // Passed by reference so let them be modified                                                 
    dutotdn = (lMgZero) ? dutot*gamnij :
              (wIsZero) ? mgntd_dutotd : 
              (wIsPstv) ? std::max(1.e-12f, dnmntr) :
                          dnmntr ;
    
    dutotds = (lMgZero) ? dutot*alphnij :
              (wIsZero) ? 0. :
                          dutot.z*sgn_w ; 
                           
    dutotdni.at(id) = dutotdn;
    dutotdsi.at(id) = dutotds;
  }
#endif

  void nonLocalMixing::rotu3psq(const int &id,const float &u3psq,const float &utot,const float &upvp,const float &upwp,
				const float &vpwp,const float &v3psq,const float &w3psq,float &ufsqb,float &wfsqb,float &vfsqb,
				float &ufvf,float &ufwf,float &vfwf)
  {
    // this subroutine rotates the fluctuating quanitities back into the normal
    // coordinate sytem
								   
    ufsqb=u3psq*alph1ij.at(id)*alph1ij.at(id)+utot*utot*alph1ij.at(id)*alph1ij.at(id)+2.*upvp*alph1ij.at(id)*alph2ij.at(id) 
        +2.*upwp*alph1ij.at(id)*alph3ij.at(id)+v3psq*alph2ij.at(id)*alph2ij.at(id) 
        +2.*vpwp*alph2ij.at(id)*alph3ij.at(id)+w3psq*alph3ij.at(id)*alph3ij.at(id)-wind_vel[id].u*wind_vel[id].u;
    wfsqb=u3psq*gam1ij.at(id)*gam1ij.at(id)+utot*utot*gam1ij.at(id)*gam1ij.at(id)+2.*upvp*gam1ij.at(id)*gam2ij.at(id) 
        +2.*upwp*gam1ij.at(id)*gam3ij.at(id)+v3psq*gam2ij.at(id)*gam2ij.at(id)+ 
        2.*vpwp*gam2ij.at(id)*gam3ij.at(id)+w3psq*gam3ij.at(id)*gam3ij.at(id)-wind_vel[id].w*wind_vel[id].w;
    vfsqb=u3psq*bet1ij.at(id)*bet1ij.at(id)+utot*utot*bet1ij.at(id)*bet1ij.at(id)+2.*upvp*bet1ij.at(id)*bet2ij.at(id) 
        +2.*upwp*bet1ij.at(id)*bet3ij.at(id)+v3psq*bet2ij.at(id)*bet2ij.at(id) 
        +2.*vpwp*bet2ij.at(id)*bet3ij.at(id)+w3psq*bet3ij.at(id)*bet3ij.at(id)-wind_vel[id].v*wind_vel[id].v;
    ufvf=u3psq*alph1ij.at(id)*bet1ij.at(id)+utot*utot*alph1ij.at(id)*bet1ij.at(id) 
        +upvp*(alph1ij.at(id)*bet2ij.at(id)+alph2ij.at(id)*bet1ij.at(id)) 
        +upwp*(alph1ij.at(id)*bet3ij.at(id)+alph3ij.at(id)*bet1ij.at(id)) 
        +v3psq*alph2ij.at(id)*bet2ij.at(id)+vpwp*(alph2ij.at(id)*bet3ij.at(id) 
        +alph3ij.at(id)*bet2ij.at(id))+w3psq*alph3ij.at(id)*bet3ij.at(id) 
        -wind_vel[id].u*wind_vel[id].v;
    ufwf=u3psq*alph1ij.at(id)*gam1ij.at(id)+utot*utot*alph1ij.at(id)*gam1ij.at(id) 
        +upvp*(alph1ij.at(id)*gam2ij.at(id)+alph2ij.at(id)*gam1ij.at(id)) 
        +upwp*(alph1ij.at(id)*gam3ij.at(id)+alph3ij.at(id)*gam1ij.at(id)) 
        +v3psq*alph2ij.at(id)*gam2ij.at(id)+vpwp*(alph2ij.at(id)*gam3ij.at(id) 
        +alph3ij.at(id)*gam2ij.at(id))+w3psq*alph3ij.at(id)*gam3ij.at(id) 
        -wind_vel[id].u*wind_vel[id].w;
    vfwf=u3psq*bet1ij.at(id)*gam1ij.at(id)+utot*utot*bet1ij.at(id)*gam1ij.at(id)+upvp* 
        (bet1ij.at(id)*gam2ij.at(id)+bet2ij.at(id)*gam1ij.at(id))+upwp*(bet1ij.at(id)*gam3ij.at(id) 
        +bet3ij.at(id)*gam1ij.at(id))+v3psq*bet2ij.at(id)*gam2ij.at(id) 
        +vpwp*(bet2ij.at(id)*gam3ij.at(id)+bet3ij.at(id)*gam2ij.at(id)) 
        +w3psq*bet3ij.at(id)*gam3ij.at(id)-wind_vel[id].v*wind_vel[id].w;
}


#if 0 // Andrew's version

  void nonLocalMixing::rotu3psq
  (
    int const& id, 
    float const& u3psq, float const& utot,  
    float const& upvp,  float const& upwp, float const& vpwp, 
    float const& v3psq, float const& w3psq, 
    float &ufsqb, float& wfsqb, float& vfsqb, 
    float& ufvf, float& ufwf, float& vfwf
  )
  {
    //Why don't any of these matrix rotations have a dedicated function?
  
    //this subroutine rotates the fluctuating quanitities back into the normal
    // coordinate sytem
    float u = wind_vel[id].x;
    float v = wind_vel[id].y;
    float w = wind_vel[id].z;
    
    float a1 = alph1ij.at(id);
    float a2 = alph2ij.at(id);
    float a3 = alph3ij.at(id);
    
    ufsqb = utot*utot*a1*a1 - u*u 
            +
            u3psq*a1*a1 + v3psq*a2*a2 + w3psq*a3*a3 
            +
            2.*(upvp*a1*a2 + upwp*a1*a3 + vpwp*a2*a3);   
        
    float g1 = gam1ij.at(id);
    float g2 = gam2ij.at(id);
    float g3 = gam3ij.at(id);
    
    wfsqb = utot*utot*g1*g1 - w*w 
            + 
            u3psq*g1*g1 + v3psq*g2*g2 + w3psq*g3*g3
            + 
            2.*(upvp*g1*g2 + upwp*g1*g3 + vpwp*g2*g3);
         
    float b1 = bet1ij.at(id);
    float b2 = bet2ij.at(id);
    float b3 = bet3ij.at(id);

    vfsqb = utot*utot*b1*b1 - v*v
            +
            u3psq*b1*b1 + v3psq*b2*b2 + w3psq*b3*b3 
            + 
            2.*(upvp*b1*b2 + upwp*b1*b3 + vpwp*b2*b3);
    
    ////////////////////////////////////////////////////////////////////////////
    ufvf = utot*utot*a1*b1 - u*v 
           +
           u3psq*a1*b1 + v3psq*a2*b2 + w3psq*a3*b3 
           + 
           upvp*(a1*b2 + a2*b1) + upwp*(a1*b3 + a3*b1) + vpwp*(a2*b3 + a3*b2);
        
    ufwf = utot*utot*a1*g1 - u*w
           +
           u3psq*a1*g1 + v3psq*a2*g2 + w3psq*a3*g3 
           +
           vpwp*(a2*g3 + a3*g2) + upvp*(a1*g2 + a2*g1) + upwp*(a1*g3 + a3*g1);
        
    vfwf = utot*utot*b1*g1 - v*w
           +
           u3psq*b1*g1 + v3psq*b2*g2 + w3psq*b3*g3 
           + 
           upvp*(b1*g2 + b2*g1) + upwp*(b1*g3 + b3*g1) + vpwp*(b2*g3 + b3*g2);
  }
#endif


  void nonLocalMixing::rotufsq(const int &id,float &u3psq,float &upwp,
			       float &v3psq,float &w3psq,const float &ufsq,const float &ufvf,
			       const float &ufwf,const float &vfsq,const float &vfwf,const float &wfsq)
  {
    float  e11,e12,e13,e21,e22,e23,e31,e32,e33;    
    
    u3psq=ufsq*alphn1ij.at(id)*alphn1ij.at(id)+wind_vel[id].u*wind_vel[id].u*alphn1ij.at(id)*alphn1ij.at(id) 
        +2.*ufvf*alphn1ij.at(id)*alphn2ij.at(id)+2.*wind_vel[id].u*wind_vel[id].v*alphn1ij.at(id) 
        *alphn2ij.at(id)+2.*ufwf*alphn1ij.at(id)*alphn3ij.at(id)+2.*wind_vel[id].u*wind_vel[id].w 
        *alphn1ij.at(id)*alphn3ij.at(id)+vfsq*alphn2ij.at(id)*alphn2ij.at(id)+wind_vel[id].v*wind_vel[id].v 
        *alphn2ij.at(id)*alphn2ij.at(id)+2.*vfwf*alphn2ij.at(id)*alphn3ij.at(id)+ 
        2.*wind_vel[id].v*wind_vel[id].w*alphn2ij.at(id)*alphn3ij.at(id)+wfsq 
        *alphn3ij.at(id)*alphn3ij.at(id)+wind_vel[id].w*wind_vel[id].w*alphn3ij.at(id)*alphn3ij.at(id)
        -(wind_vel[id].u*wind_vel[id].u+wind_vel[id].v*wind_vel[id].v +wind_vel[id].w*wind_vel[id].w);
    
    v3psq=ufsq*betn1ij.at(id)*betn1ij.at(id)+wind_vel[id].u*wind_vel[id].u*betn1ij.at(id)*betn1ij.at(id)+2.*ufvf 
        *betn1ij.at(id)*betn2ij.at(id)+2.*wind_vel[id].u*wind_vel[id].v*betn1ij.at(id) 
        *betn2ij.at(id)+2.*ufwf*betn1ij.at(id)*betn3ij.at(id)+2.*wind_vel[id].u 
        *wind_vel[id].w*betn1ij.at(id)*betn3ij.at(id)+vfsq*betn2ij.at(id)*betn2ij.at(id) 
        +wind_vel[id].v*wind_vel[id].v*betn2ij.at(id)*betn2ij.at(id)+2.*vfwf*betn2ij.at(id)*betn3ij.at(id) 
        +2.*wind_vel[id].v*wind_vel[id].w*betn2ij.at(id)*betn3ij.at(id)+wfsq*betn3ij.at(id)*betn3ij.at(id) 
        +wind_vel[id].w*wind_vel[id].w*betn3ij.at(id)*betn3ij.at(id);
    w3psq=ufsq*gamn1ij.at(id)*gamn1ij.at(id)+wind_vel[id].u*wind_vel[id].u*gamn1ij.at(id)*gamn1ij.at(id)+2.*ufvf 
        *gamn1ij.at(id)*gamn2ij.at(id)+2.*wind_vel[id].u*wind_vel[id].v*gamn1ij.at(id) 
        *gamn2ij.at(id)+2.*ufwf*gamn1ij.at(id)*gamn3ij.at(id)+2.*wind_vel[id].u 
        *wind_vel[id].w*gamn1ij.at(id)*gamn3ij.at(id)+vfsq*gamn2ij.at(id)*gamn2ij.at(id) 
        +wind_vel[id].v*wind_vel[id].v*gamn2ij.at(id)*gamn2ij.at(id)+2.*vfwf*gamn2ij.at(id)*gamn3ij.at(id) 
        +2.*wind_vel[id].v*wind_vel[id].w*gamn2ij.at(id)*gamn3ij.at(id)+wfsq*gamn3ij.at(id)*gamn3ij.at(id) 
        +wind_vel[id].w*wind_vel[id].w*gamn3ij.at(id)*gamn3ij.at(id);
    upwp=ufsq*alphn1ij.at(id)*gamn1ij.at(id)+wind_vel[id].u*wind_vel[id].u*alphn1ij.at(id)* 
        gamn1ij.at(id)+ufvf*(alphn1ij.at(id)*gamn2ij.at(id)+alphn2ij.at(id)*gamn1ij.at(id))
        +wind_vel[id].u*wind_vel[id].v*(alphn1ij.at(id)*gamn2ij.at(id)+ alphn2ij.at(id)*gamn1ij.at(id))
        +ufwf*(alphn1ij.at(id)*gamn3ij.at(id)+alphn3ij.at(id)*gamn1ij.at(id))
        +wind_vel[id].u*wind_vel[id].w*(alphn1ij.at(id)*gamn3ij.at(id)+alphn3ij.at(id)*gamn1ij.at(id))+vfsq*alphn2ij.at(id) 
        *gamn2ij.at(id)+wind_vel[id].v*wind_vel[id].v*alphn2ij.at(id)*gamn2ij.at(id) 
        +vfwf*(alphn2ij.at(id)*gamn3ij.at(id)+alphn3ij.at(id)*gamn2ij.at(id)) 
        +wind_vel[id].v*wind_vel[id].w*(alphn2ij.at(id)*gamn3ij.at(id)+alphn3ij.at(id)*gamn2ij.at(id))+wfsq*alphn3ij.at(id)
        *gamn3ij.at(id)+wind_vel[id].w*wind_vel[id].w *alphn3ij.at(id)*gamn3ij.at(id);

    e11=alph1ij.at(id)*alphn1ij.at(id)+alph2ij.at(id)*betn1ij.at(id)+alph3ij.at(id)*gamn1ij.at(id);
    e12=alph1ij.at(id)*alphn2ij.at(id)+alph2ij.at(id)*betn2ij.at(id)+alph3ij.at(id)*gamn2ij.at(id);
    e13=alph1ij.at(id)*alphn3ij.at(id)+alph2ij.at(id)*betn3ij.at(id)+alph3ij.at(id)*gamn3ij.at(id);
    e21=bet1ij.at(id)*alphn1ij.at(id)+bet2ij.at(id)*betn1ij.at(id)+bet3ij.at(id)*gamn1ij.at(id);
    e22=bet1ij.at(id)*alphn2ij.at(id)+bet2ij.at(id)*betn2ij.at(id)+bet3ij.at(id)*gamn2ij.at(id);
    e23=bet1ij.at(id)*alphn3ij.at(id)+bet2ij.at(id)*betn3ij.at(id)+bet3ij.at(id)*gamn3ij.at(id);
    e31=gam1ij.at(id)*alphn1ij.at(id)+gam2ij.at(id)*betn1ij.at(id)+gam3ij.at(id)*gamn1ij.at(id);
    e32=gam1ij.at(id)*alphn2ij.at(id)+gam2ij.at(id)*betn2ij.at(id)+gam3ij.at(id)*gamn2ij.at(id);
    e33=gam1ij.at(id)*alphn3ij.at(id)+gam2ij.at(id)*betn3ij.at(id)+gam3ij.at(id)*gamn3ij.at(id);
    
}

#if 0 // Andrew's version
  void nonLocalMixing::rotufsq
  (
    int const& id, float& u3psq, float& upwp, float& v3psq, float& w3psq, 
    float const& ufsq, float const& ufvf, float const& ufwf, 
    float const& vfsq, float const& vfwf, float const& wfsq,
  )
  {
    
    float u = wind_vel[id].u;
    float v = wind_vel[id].v;
    float w = wind_vel[id].w;
    
    float an1 = alphn1ij.at(id);
    float an2 = alphn2ij.at(id);
    float an3 = alphn3ij.at(id);
    
    float bn1 = betn1ij.at(id);
    float bn2 = betn2ij.at(id);
    float bn3 = betn3ij.at(id);
    
    float gn1 = gamn1ij.at(id);
    float gn2 = gamn2ij.at(id);
    float gn3 = gamn3ij.at(id);

    u3psq =    ufsq*an1*an1 +    u*u*an1*an1 +
               vfsq*an2*an2 +    v*v*an2*an2 + 
               wfsq*an3*an3 +    w*w*an3*an3 -
            2.*ufvf*an1*an2 + 2.*u*v*an1*an2 + 
            2.*ufwf*an1*an3 + 2.*u*w*an1*an3 + 
            2.*vfwf*an2*an3 + 2.*v*w*an2*an3 + 
               mag(wind_vel[id]);

    v3psq =    ufsq*bn1*bn1 +    u*u*bn1*bn1 + 
               vfsq*bn2*bn2 +    v*v*bn2*bn2 + 
               wfsq*bn3*bn3 +    w*w*bn3*bn3;
            2.*ufvf*bn1*bn2 + 2.*u*v*bn1*bn2 + 
            2.*ufwf*bn1*bn3 + 2.*u*w*bn1*bn3 + 
            2.*vfwf*bn2*bn3 + 2.*v*w*bn2*bn3 + 
    
    w3psq =    ufsq*gn1*gn1 +    u*u*gn1*gn1 + 
               vfsq*gn2*gn2 +    v*v*gn2*gn2 + 
               wfsq*gn3*gn3 +    w*w*gn3*gn3;
            2.*ufvf*gn1*gn2 + 2.*u*v*gn1*gn2 + 
            2.*ufwf*gn1*gn3 + 2.*u*w*gn1*gn3 + 
            2.*vfwf*gn2*gn3 + 2.*v*w*gn2*gn3 + 
    
    upwp = ufsq* an1*gn1 +            u*u* an1*gn1 + 
           vfsq* an2*gn2 +            v*v* an2*gn2 + 
           wfsq* an3*gn3 +            w*w* an3*gn3;
           ufvf*(an1*gn2 + an2*gn1) + u*v*(an1*gn2 + an2*gn1) +
           ufwf*(an1*gn3 + an3*gn1) + u*w*(an1*gn3 + an3*gn1) + 
           vfwf*(an2*gn3 + an3*gn2) + v*w*(an2*gn3 + an3*gn2) + 
  }
#endif


  void nonLocalMixing::rotate2d(const int &id,const float &cosphi,const float &sinphi,const float &upsqg,
				const float &upvpg,const float &vpsqg,const float &wpsqg,const float &upwpg,const float &vpwpg )
  {
    // this subroutine rotates variables from primed system aligned with the overall wind
    // into the regular grid system
    float ub,vb,wb,upb,vpb;
    ub=wind_vel[id].u;
    vb=wind_vel[id].v;
    wb=wind_vel[id].w;
    upb=ub*cosphi+vb*sinphi;
    vpb=-ub*sinphi+vb*cosphi;
    ufsqgi.at(id)=upsqg*cosphi*cosphi+upb*upb*cosphi*cosphi-2.f*cosphi*sinphi*upvpg 
        -2.f*cosphi*sinphi*upb*vpb+vpsqg*sinphi*sinphi+sinphi*sinphi*vpb*vpb-ub*ub;

    vfsqgi.at(id)=upsqg*sinphi*sinphi+upb*upb*sinphi*sinphi+2.f*upvpg*sinphi*cosphi 
        +2.f*upb*vpb*sinphi*cosphi+vpsqg*cosphi*cosphi+vpb*vpb*cosphi*cosphi-vb*vb;
    wfsqgi.at(id)=wpsqg;
    ufvfgi.at(id)=upsqg*cosphi*sinphi+upb*upb*cosphi*sinphi+upvpg
        *(cosphi*cosphi-sinphi*sinphi)+upb*vpb*(cosphi*cosphi-sinphi*sinphi)-vpsqg*sinphi*cosphi
        -vpb*vpb*sinphi*cosphi-ub*vb;
    ufwfgi.at(id)=upwpg*cosphi+upb*wb*cosphi-vpwpg*sinphi-vpb*wb*sinphi-ub*wb;
    vfwfgi.at(id)=upwpg*sinphi+upb*wb*sinphi+vpwpg*cosphi+vpb*wb*cosphi-vb*wb;
}


  //void nonLocalMixing::nonLocalMixing
  //(
  //  unsigned int windField, unsigned int lambda,    unsigned int tau_dz, 
  //  unsigned int duvw_dz,   unsigned int dxyz_wall, unsigned int tauTex
  //)
  void nonLocalMixing::mix()
  {
    std::cout << "Running with Non-Local Mixing Calculations" << std::endl;

    GLfloat *data     = new GLfloat[ width * height * 4 ];
    GLfloat *dataWind = new GLfloat[ width * height * 4 ];
    GLfloat *dataTwo  = new GLfloat[ width * height * 4 ];
    GLfloat *dataTau  = new GLfloat[width*height*4];
    GLfloat *data3    = new GLfloat[width*height*4];
    GLfloat *data4    = new GLfloat[width*height*4];

    //Balli's new additions this is essentially a direct copy of the FORTRAN
    initCellType();
    
    //Balli: Substracting 1 from nzdz as it is increased by 1  after reading from QU_simparams.inp in Util.cpp
    // This is no longer true... we need to make sure we treat the variable correctly.
    nzdz=nzdz-1;
    std::string s;

    float nxnynz=nxdx*nydy*nzdz;

    float dx = m_util_ptr->dx;
    float dy = m_util_ptr->dy;
    float dz = m_util_ptr->dz;

    std::vector<float> dz_array,z,zm;
    dz_array.resize(nzdz,dz);
    z.resize(nzdz,0.0f);
    zm.resize(nzdz,0.0f);

    //Balli: Initialized first element of z and zm array before begining the loop as in GPU Plume
    //we do not store values below the ground, which are zero anyways
    z.at(0)  = dz_array.at(0);
    zm.at(0) = z.at(0) - 0.5*dz_array.at(0);
    
    for(int k=1;k<nzdz;k++)
      {
	z.at(k)  = z.at(k-1)+dz_array.at(k);
	zm.at(k) = z.at(k)-0.5*dz_array.at(k);
      }

    int roofflag = m_util_ptr->quSimParamData.roof_type;
    float rcl = m_util_ptr->qpParamData.rcl;
    float z0 = m_util_ptr->qpParamData.z0;
    float h = m_util_ptr->qpParamData.boundaryLayerHeight; //Boundary Layer Height
  
    //Balli: declaring few constants
    //NLM: non-local mixing
    const float kkar = 0.4f;           //von karman constant
    const float pi   = 4.f*atan(1.0f);
    const float knlc   = 0.113f;      //non-local mixing constant   
    const float ctau13 = 1.f;           
    const float cusq   = 2.5f*2.5f;     
    const float cvsq   = 2.f*2.f;         
    const float cwsq   = 1.3f*1.3f;    


    //Balli: "theta" is never read into or initilized in the FORTRAN code, but used for calculating ualoft and valoft.
    float theta  = 0.f;          // This varibale is not used anymore in QP but the legacy code still uses it-Balli-06/10/09
    float ualoft = 0.f;          
    float valoft = 0.f;             

    int time_idx=1;
    int check1=0;
    int check2=0;
    int check3=0;
    int check=0;

    //Balli: For writing turbulence data- Can be removed later
    std::ofstream turbfield;
    turbfield.open("GPU_turbfield.dat");
    
    //Balli : Declaring local vectors
    std::vector<float> elz,ustarz,sigwi,sigvi,ustarij,xi,yi,zi,hgt,hgtveg,eleff,xcb,ycb,icb,jcb,phib,zcorf;
    std::vector<float>uref,urefu,urefv,urefw, utotktp,uktop,vktop,wktop,deluc,ustargz,elzg,ustarg;
    std::vector<float>utotcl1,utotmax;
    
    // no longer needed
    // std::vector<int>bldtype;
    // std::vector<float> gamma, atten, Sx,Sy, weff, leff, lfr, lr
    
    //Balli : Vectors needs to be resized before they are  used otherwise they sometime give runtime errors
    eleff.resize(nxnynz,0.0); // efective turbulent length scale-initialized with zero values.
    ustarg.resize(nxnynz,0.0); // ustarg is a non-local stress velocity scale, horizontal direction - see Williams et al. 2004

    // Rather than re-read QP_buildout.inp here, we instead use the
    // information previously obtained frmo the quicloaders used by
    // gpuplume.
    // 
    //Balli : Reading "QU_buildout.inp"; THis should be handled in
    //Util.cpp - This file is required for getiign effective lenths of
    //downstream and upstream(along with other parameters) cavities
    //from QUICURB

    // grab the number of buildings read from QP_buildout.inp
    unsigned int numBuild = m_util_ptr->qpBuildoutData.buildings.size();
    unsigned int inumveg = m_util_ptr->qpBuildoutData.numVegetativeCanopies;

    // completed
    // bldtype.resize(numBuild);
    // gamma.resize(numBuild);
    // atten.resize(numBuild);
    // Sx.resize(numBuild);
    // Sy.resize(numBuild);
    // weff.resize(numBuild);
    // leff.resize(numBuild);
    // lfr.resize(numBuild);
    // lr.resize(numBuild);

    //Balli: IMPORTANT!!
    //QP differs from GPU in the indices i,j,k of all the arrays (u,v,w etc.)
    //Following calculations are for Boundary layer case only and it provides an insight into the coordinate sytem differences
    //QP velocity vectors, for example u(i,j,k), i goes from 1(0.5) to nx(9.5), j goes from 1(0.5) to ny(9.5)
    //and k goes from 1(-0.5) to nz+1(29.5)
    //[Note: nx,ny,nz above are what QP reads from input file, QP adds 1 to nx and ny, and 2 to nz after reading them from input file]
    
    //QP's k goes from 1 to nz+2, therefore for dz=1, zi goes from -0.5(k=1) to 30.5(k=32)
    //GPU k goes from 0 to nz-1, therefore for dz=1, zi goes from 0.5(k=0) to 29.5 (k=29)

    zi.resize(nzdz);
    for(int k=0;k<nzdz;k++){ 
      zi.at(k)=.5*dz+dz*k; //this expression is different from that used in the QP, but values are same
    }
    //QP's j goes from 1 to ny+1, therefore for dy=1, yi goes from 0.5(j=1) to 10.5(j=11)
    //GPU j goes from 0 to ny-1, therefore for dy=1, yi goes from 0.5(j=0) to 9.5 (j=9)
    
    yi.resize(nydy);
    for(int j=0;j<nydy;j++){
      yi.at(j)=.5*dy+dy*j;
    }
    //QP's i goes from 1 to nx+1, therefore for dx=1, xi goes from 0.5(i=1) to 10.5(i=11)
    //GPU i goes from 0 to nx-1, therefore for dx=1, xi goes from 0.5(i=0) to 9.5 (i=9)
    
    xi.resize(nxdx);
    for(int i=0;i<nxdx;i++){
      xi.at(i)=.5*dx+dx*i;
    }
    
    float ht_avg = 0.0;
    int k=0;
    if(numBuild > 0 && numBuild != inumveg){
      for(int  i_b=0;i_b<numBuild;i_b++){
	if(m_util_ptr->qpBuildoutData.buildings[i_b].type == 9){
	  continue;
	}
	ht_avg=ht[i_b]+zfo[i_b]+ht_avg;
      }
      ht_avg=ht_avg/float(numBuild-inumveg);
      float temp=ht_avg/dz;
      for(int kk=0;kk<nzdz;kk++){
	k=kk;
	if(ht_avg<z.at(kk))break;
      }
    }
    else{
      //BL Flow case: Control comes here as we have no buildings for this test case
      k=0; //altered to comply with GPU
    }
    dz=dz_array.at(k);
    
    //Obtain avg. velocity from the boundary of the domain at above cal avg ht of the buildings
    int i=0;//altered to comply with GPU
    int j=0;
    float u_left=0;
    for(j=0;j<nydy;j++){
      int p2idx = k*nxdx*nydy + j*nxdx + i;
      u_left=sqrt(wind_vel[p2idx].x*wind_vel[p2idx].x + wind_vel[p2idx].y*wind_vel[p2idx].y + wind_vel[p2idx].z*wind_vel[p2idx].z) +u_left;
    }
    u_left=u_left/(nydy);//altered to comply GPU
    // in QP, total number of cells in y is read from input file as ny and then QP adds 1to ny, therefore, 1 is substracted from ny above in QP.
    
    j=nydy-1;// substracted 1 as edge of the domain in y is nydy-1
    float u_top=0;
    for(i=0;i<nxdx;i++){
      int p2idx = k*nxdx*nydy + j*nxdx + i;
      u_top=sqrt(wind_vel[p2idx].u*wind_vel[p2idx].u + wind_vel[p2idx].v*wind_vel[p2idx].v + wind_vel[p2idx].w*wind_vel[p2idx].w) +u_top;
    }
    u_top=u_top/nxdx;
    i=nxdx-1;//same explanation as in case of j above
    float u_right=0;
    for(j=0;j<nydy;j++){
      int p2idx = k*nxdx*nydy + j*nxdx + i;
      u_right=sqrt(wind_vel[p2idx].u*wind_vel[p2idx].u + wind_vel[p2idx].v*wind_vel[p2idx].v + wind_vel[p2idx].w*wind_vel[p2idx].w) +u_right;
    }
    u_right=u_right/nydy;
    j=0;//alterted for GPU
    
    float u_bottom=0;
    for(i=0;i<nxdx;i++){
      int p2idx = k*nxdx*nydy + j*nxdx + i;
      u_bottom=sqrt(wind_vel[p2idx].u*wind_vel[p2idx].u + wind_vel[p2idx].v*wind_vel[p2idx].v + wind_vel[p2idx].w*wind_vel[p2idx].w) +u_bottom;
    }
    u_bottom=u_bottom/(nxdx);
    
    float u_b=0.25*(u_left+ u_top+ u_right+ u_bottom);//average velocity
    float nu_b=1.5e-5; //nu for air
    float del_b=(0.328* pow(nu_b/u_b,.2f) ) * pow(ht_avg,.8f);//expression for BL layer thickness (growth) (i think)
    // above expression is used for obtaining turbulence close to the walls of the buildings
    
    hgt.resize(nxnynz,0.);
    hgtveg.resize(nxnynz,0.);
    if(time_idx == 1){
      for(int k=0; k<nzdz;k++){
	for(int j=0; j<nydy;j++){
	  for(int i=0; i<nxdx;i++){
	    int p2idx = k*nxdx*nydy + j*nxdx + i;
	    int ij = j*nxdx + i;
	    if(retrieveCellTypeFromArray(p2idx) == 0)hgt.at(ij)=std::max(hgt.at(ij),z.at(k));
	    if(retrieveCellTypeFromArray(p2idx) == 8)hgtveg.at(ij)=std::max(hgtveg.at(ij),z.at(k));
	  }
	}
      }
    }    
    
    elz.resize(nxnynz);
    ustarz.resize(nxnynz);
    sigwi.resize(nxnynz);
    sigvi.resize(nxnynz);
    ustarij.resize(nxnynz);
    ustarz.resize(nxnynz);

    
    //Balli:Allocating global arrays, declared in header file
    dutotdxi.resize(nxnynz);
    dutotdyi.resize(nxnynz);
    dutotdzi.resize(nxnynz,0.0);
    dutotdni.resize(nxnynz);
    dutotdsi.resize(nxnynz);
    alph1ij.resize(nxnynz);
    alph2ij.resize(nxnynz);
    alph3ij.resize(nxnynz);
    bet1ij.resize(nxnynz);
    bet2ij.resize(nxnynz);
    bet3ij.resize(nxnynz);
    gam1ij.resize(nxnynz);
    gam2ij.resize(nxnynz);
    gam3ij.resize(nxnynz);
    alphn1ij.resize(nxnynz);
    alphn2ij.resize(nxnynz);
    alphn3ij.resize(nxnynz);
    betn1ij.resize(nxnynz);
    betn2ij.resize(nxnynz);
    betn3ij.resize(nxnynz);
    gamn1ij.resize(nxnynz);
    gamn2ij.resize(nxnynz);
    gamn3ij.resize(nxnynz);
    ani.resize(nxnynz);
    bni.resize(nxnynz);
    cni.resize(nxnynz);
    ufsqgi.resize(nxnynz);
    vfsqgi.resize(nxnynz);
    wfsqgi.resize(nxnynz);
    ufvfgi.resize(nxnynz);
    ufwfgi.resize(nxnynz);
    vfwfgi.resize(nxnynz);

    
    std::vector<float> dzm,dzp,dym,dyp,dxm,dxp,ufwfi,ufvfi,vfwfi,sigui,upwpi,epsi;
    
    dzm.resize(nzdz*nydy*nxdx);
    dzp.resize(nzdz*nydy*nxdx);
    dxm.resize(nzdz*nydy*nxdx);
    dxp.resize(nzdz*nydy*nxdx);
    dym.resize(nzdz*nydy*nxdx);
    dyp.resize(nzdz*nydy*nxdx);
    ufwfi.resize(nzdz*nydy*nxdx);
    ufvfi.resize(nzdz*nydy*nxdx);
    vfwfi.resize(nzdz*nydy*nxdx);
    sigui.resize(nzdz*nydy*nxdx);
    upwpi.resize(nzdz*nydy*nxdx);
    epsi.resize(nzdz*nydy*nxdx);
    
    int kcantop=0;
    float ucantop=0.;
    float elcanopy=0.;
    float utotl=0.;
    float utotu=0.;
    float phim=0.;
    float psim=0.;
    float dutotl=0.;
    float dutotu=0.f;
    float dutot=0.f;
    float utot=0.f;
    float dutotdzc=0.;
    float dutotdzp=0.;
    float dutotdzm=0.;
    float dutotdza=0.;
   
    //
    // Beginning of NonLocalMixing 
    // 
    bool useNonLocalMixing = true;
    if (useNonLocalMixing == true)
      {

	for(int j=0;j<nydy;j++){
	  for(int i=0;i<nxdx;i++){
            int ij = j*nxdx + i;
            if(hgtveg.at(ij) > 0.){
	      for(int kk=0;kk<nzdz;k++){
		kcantop=kk;
		if(hgtveg.at(ij) <= z.at(kk))break;
	      }
	      int idcan  = kcantop*nxdx*nydy + j*nxdx + i;
	      int id1can = (kcantop+1)*nxdx*nydy + j*nxdx + i;
                
	      ucantop=.5*sqrt(wind_vel[idcan].u*wind_vel[idcan].u+wind_vel[idcan].v*wind_vel[idcan].v+wind_vel[idcan].w*wind_vel[idcan].w)
		+ .5*sqrt(wind_vel[id1can].u*wind_vel[id1can].u+wind_vel[id1can].v*wind_vel[id1can].v+wind_vel[id1can].w*wind_vel[id1can].w); 
            }
            for(int k=0;k<nzdz;k++){

	      dz=dz_array.at(k);
	      int km1   = (k-1)*nxdx*nydy + j*nxdx + i;
	      int kp1   = (k+1)*nxdx*nydy + j*nxdx + i;
	      int knz1 = (nzdz-1)*nxdx*nydy + j*nxdx + i;
	      int p2idx = k*nxdx*nydy + j*nxdx + i;
	      int ij = j*nxdx + i;
	      int idklow=0;
	      //new changes from QUIC
	      if(retrieveCellTypeFromArray(p2idx) != 0){
		dzm.at(p2idx)=zm.at(k)-hgt.at(ij);
		eleff.at(p2idx)=dzm.at(p2idx);
	      }
	      else{
		dzm.at(p2idx)=0.f;
		eleff.at(p2idx)=0.f;
	      }
	      elcanopy=0.f;


#if 0
	      // *** Pete
	      if (km1 < 0)
		std::cout << "KM1 = " << km1 << ", cellQuic = " << (int)cellQuic[km1].c << std::endl;
#endif

	      if(((retrieveCellTypeFromArray(km1) == 0) || (retrieveCellTypeFromArray(km1)==8)) && 
		 (retrieveCellTypeFromArray(p2idx) != 0 && retrieveCellTypeFromArray(p2idx) != 8) || k == 0){//altered k
		utotl=0.f;
		utotu=sqrt(wind_vel[p2idx].u*wind_vel[p2idx].u+wind_vel[p2idx].v*wind_vel[p2idx].v+wind_vel[p2idx].w*wind_vel[p2idx].w);
		//MDW 7-01-2005 changed the way vertical gradients are calculated to avoid inaccuracies
		// in the representation of the gradients of a log-law term
                    
		if(rcl>0){
		  phim=1.f+4.7f*rcl*.5f*dz;
		  psim=-4.7f*rcl*.5f*dz;
		}
		else{
		  phim=pow( (1.f-15.f*rcl*.5f*dz),(-.25f));
		  psim=2.f*log((1.f+1.f/phim)/2.f)+log((1.f+1.f/pow(phim,2.f))/2.f)-2.f*atan(1.f/phim)+pi/2.f;
		}
		if(hgtveg.at(ij)>zi.at(k)){                           
		  phim=1.f;
		  psim=0.f;
		  elz.at(p2idx)=elcanopy*std::min(1.f,(zi.at(k)-z0)/(.3f*hgtveg.at(ij)));
		  ustar=elz.at(p2idx)*utotu/(.5f*dz);
		  dutotdzi.at(p2idx)=utotu/(.5f*dz);
		  ustarz.at(p2idx)=ustar;
		}
		else{
		  if(retrieveCellTypeFromArray(km1)!=8){
		    ustar=kkar*utotu/(log(.5f*dz/z0)-psim);
		    elz.at(p2idx)=kkar*.5f*dz;
		    ustarz.at(p2idx)=ustar;
		    dutotdzi.at(p2idx)=ustar*phim/elz.at(p2idx);
		  }
		  else{
		    utotl=sqrt(wind_vel[km1].u*wind_vel[km1].u+wind_vel[km1].v*wind_vel[km1].v+wind_vel[km1].w*wind_vel[km1].w);
		    dutotdzi.at(p2idx)=2.f*(utotu-utotl)/(dz_array.at(k-1)+dz_array.at(k));
		    elz.at(p2idx)=kkar*.5f*dz;
		    ustar=elz.at(p2idx)*dutotdzi.at(p2idx);
		    ustarz.at(p2idx)=ustar;
		  }
		}
		if(retrieveCellTypeFromArray(km1)!=8 && k!=0){
		  sigwi.at(km1)=0.f;
		  sigvi.at(km1)=0.f;
		  ustarij.at(km1)=0.f;
		  ustarz.at(km1)=0.f;
		}
	      }
	      else{
		if(k==nzdz-1){ // find gradient using a non-CDD approach
		  utotl=sqrt(wind_vel[km1].u*wind_vel[km1].u+wind_vel[km1].v*wind_vel[km1].v+wind_vel[km1].w*wind_vel[km1].w);
		  utotu=sqrt(wind_vel[p2idx].u*wind_vel[p2idx].u+wind_vel[p2idx].v*wind_vel[p2idx].v+wind_vel[p2idx].w*wind_vel[p2idx].w);
		  dutotdzi.at(knz1)=dutotdzi.at(km1)*zm.at(k-1)/zm.at(k);
		  elz.at(p2idx)=kkar*(eleff.at(p2idx)-hgtveg.at(ij));
		}
		else{ // find gradient using a CDD approach
		  utotl=sqrt(wind_vel[km1].u*wind_vel[km1].u+wind_vel[km1].v*wind_vel[km1].v+wind_vel[km1].w*wind_vel[km1].w);
		  utotu=sqrt(wind_vel[kp1].u*wind_vel[kp1].u+wind_vel[kp1].v*wind_vel[kp1].v+wind_vel[kp1].w*wind_vel[kp1].w);
		  // mdw 7-08-2005 changed the way vertical gradients are calculated to better represent
		  // log-law behavior
		  if(retrieveCellTypeFromArray(p2idx)==8){
		    dutotdzi.at(p2idx)=(utotu-utotl)/(dz_array.at(k)+.5*dz_array.at(k-1)+.5*dz_array.at(k+1));
		    ustarz.at(p2idx)=elcanopy*dutotdzi.at(p2idx);
		    elz.at(p2idx)=elcanopy*std::min(1.f,(zi.at(k)-z0)/(0.3f*hgtveg.at(ij)));
		  }
		  else{
		    int klow=0;
		    for (int kk=0;kk<nzdz;kk++){
		      klow=kk;
		      if(std::max(hgt.at(ij),hgtveg.at(ij))<z.at(kk))break;
		    }

		    idklow = klow*nxdx*nydy + j*nxdx + i;
		    if(rcl>0){
		      phim=1.f+4.7f*rcl*(eleff.at(km1)-std::max((hgtveg.at(ij)-elcanopy/kkar),0.f));
		      psim=-4.7f*rcl*(eleff.at(km1)-std::max((hgtveg.at(ij)-elcanopy/kkar),0.f));
		    }
		    else{
		      phim=pow( (1.f-15.f*rcl*(eleff.at(km1)-std::max((hgtveg.at(ij)-elcanopy/kkar),0.f))),(-.25f));
		      psim=2.f*log((1.f+1.f/phim)/2.f)+log((1.f+1.f/pow(phim,2.f))/2.f)-2.f*atan(1.f/phim)+pi/2.f;
		    }
		    dutotl=utotl-ustarz.at(idklow)*(log((zi.at(k-1)-std::max((hgtveg.at(ij)-elcanopy/kkar),0.f))/z0)-psim)/kkar;
		    if(rcl>0){
		      phim=1.f+4.7f*rcl*(eleff.at(kp1)-std::max((hgtveg.at(ij)-elcanopy/kkar),0.f));
		      psim=-4.7f*rcl*(eleff.at(kp1)-std::max((hgtveg.at(ij)-elcanopy/kkar),0.f));
		    }
		    else{
		      phim=pow( (1.f-15.f*rcl*(eleff.at(kp1)-std::max((hgtveg.at(ij)-elcanopy/kkar),0.f))),(-.25f));
		      psim=2.f*log((1.f+1.f/phim)/2.f)+log((1.f+1.f/pow(phim,2.f))/2.f)-2.f*atan(1.f/phim)+pi/2.f;
		    }
		    dutotu=utotu-ustarz.at(idklow)*(log((zi.at(k+1)-std::max((hgtveg.at(ij)-elcanopy/kkar),0.f))/z0)-psim)/kkar;
		    dutotdzi.at(p2idx)=(dutotu-dutotl)/(dz_array.at(k)+.5*dz_array.at(k-1)+.5*dz_array.at(k+1))
		      +ustarz.at(idklow)*psim/(kkar*zi.at(k));
		    elz.at(p2idx)=kkar*(eleff.at(p2idx)-std::max((hgtveg.at(ij)-elcanopy/kkar),0.f));
		    if(retrieveCellTypeFromArray(kp1) != 0 && retrieveCellTypeFromArray(p2idx) != 0  && retrieveCellTypeFromArray(km1) != 0){
		      // mdw 7-01-2005 centered around k instead of k-1 and ajusted for log-law behavior
		      utot=sqrt(wind_vel[p2idx].u*wind_vel[p2idx].u+wind_vel[p2idx].v*wind_vel[p2idx].v+wind_vel[p2idx].w*wind_vel[p2idx].w);
		      utotl=sqrt(wind_vel[km1].u*wind_vel[km1].u+wind_vel[km1].v*wind_vel[km1].v+wind_vel[km1].w*wind_vel[km1].w);
		      utotu=sqrt(wind_vel[kp1].u*wind_vel[kp1].u+wind_vel[kp1].v*wind_vel[kp1].v+wind_vel[kp1].w*wind_vel[kp1].w);
		      if(rcl>0){
			phim=1.f+4.7f*rcl*(eleff.at(km1)-std::max((hgtveg.at(ij)-elcanopy/kkar),0.f));
			psim=-4.7f*rcl*(eleff.at(km1)-std::max((hgtveg.at(ij)-elcanopy/kkar),0.f));
		      }
		      else{
			phim=pow( (1.-15.*rcl*(eleff.at(km1)-std::max((hgtveg.at(ij)-elcanopy/kkar),0.f))),(-.25));
			psim=2.*log((1.+1./phim)/2.)+log((1.+1./pow(phim,2.f))/2.)-2.*atan(1./phim)+pi/2.;
		      }
		      dutotl=utotl-ustarz.at(idklow)*(log((zi.at(k-1)-std::max((hgtveg.at(ij)-elcanopy/kkar),0.f))/z0)-psim)/kkar;
		      if(rcl>0){
			phim=1.+4.7*rcl*(eleff.at(kp1)-std::max((hgtveg.at(ij)-elcanopy/kkar),0.f));
			psim=-4.7*rcl*(eleff.at(kp1)-std::max((hgtveg.at(ij)-elcanopy/kkar),0.f));
		      }
		      else{
			phim=pow( (1.-15.*rcl*(eleff.at(kp1)-std::max((hgtveg.at(ij)-elcanopy/kkar),0.f))),(-.25));
			psim=2.*log((1.+1./phim)/2.)+log((1.+1./pow(phim,2.f))/2.)-2.*atan(1./phim)+pi/2.;
		      }
		      dutotu=utotu-ustarz.at(idklow)*(log((zi.at(k+1)-std::max((hgtveg.at(ij)-elcanopy/kkar),0.f))/z0)-psim)/kkar;
		      // mdw 3-08-2004 begin changes for highest gradient rather than centered diff gradient
		      if(rcl>0){
			phim=1.+4.7*rcl*(eleff.at(p2idx)-std::max((hgtveg.at(ij)-elcanopy/kkar),0.f));
			psim=-4.7*rcl*(eleff.at(p2idx)-std::max((hgtveg.at(ij)-elcanopy/kkar),0.f));
		      }
		      else{
			phim=pow( (1.-15.*rcl*(eleff.at(p2idx)-std::max((hgtveg.at(ij)-elcanopy/kkar),0.f))),(-.25) );
			psim=2.*log((1.+1./phim)/2.)+log((1.+1./pow(phim,2.f))/2.)-2.*atan(1./phim)+pi/2.;
		      }
		      dutot=utot-ustarz.at(idklow)*(log((zi.at(k)-std::max((hgtveg.at(ij)-elcanopy/kkar),0.f))/z0)-psim)/kkar;
		      dutotdzc=(dutotu-dutotl)/(dz_array.at(k)+.5*dz_array.at(k+1)+.5*dz_array.at(k-1));
		      dutotdzp=(dutotu-dutot)/(.5*dz_array.at(k+1)+.5*dz_array.at(k));
		      dutotdzm=(dutot-dutotl)/(.5*dz_array.at(k)+.5*dz_array.at(k-1));
		      dutotdza=0.5*(fabs(dutotdzp+ustarz.at(idklow)*phim/(kkar*(zi.at(k) 
										-std::max((hgtveg.at(ij)-elcanopy/kkar),0.f)))) 
				    +fabs(dutotdzm+ustarz.at(idklow)*phim/(kkar*(zi.at(k) 
										 -std::max((hgtveg.at(ij)-elcanopy/kkar),0.f)))));
		      if(abs(dutotdzp+ustarz.at(idklow)*phim/(kkar*(zi.at(k)-std::max((hgtveg.at(ij)-elcanopy/kkar),0.f))))> 
			 fabs(dutotdzm+ustarz.at(idklow)*phim/(kkar*(zi.at(k)-std::max((hgtveg.at(ij)-elcanopy/kkar),0.f))))){
			dutotdzi.at(p2idx)=dutotdzp+ustarz.at(idklow)*phim/(kkar*(zi.at(k) 
										  -std::max((hgtveg.at(ij)-elcanopy/kkar),0.f)));
		      }
		      else{
			dutotdzi.at(p2idx)=dutotdzm+ustarz.at(idklow)*phim/(kkar*(zi.at(k) 
										  -std::max((hgtveg.at(ij)-elcanopy/kkar),0.f)));
                                      
                                      
		      }
		    }
		    // use centered differences away from the boundaries
		  }
		}
	      }
            }
	  }
	}//end for loops

	//Balli: Above loop is working as expected. I have matched every variable value with QP (Balli-06/14/09)
	// IMPORTANT!!! Theta is never read from input file. so its values will be zero always.
	//Therefore following few lines do not effect the final solution at all
	// phi will be calculated again by taking into account the actual wind angle at each building.

	// float phi = m_util_ptr->quMetParamData.quSensorData.direction.degrees() - theta;
	float phi = m_util_ptr->quMetParamData.quSensorData.direction - theta;
	// Was -->> float phi = 270.-theta;
	// phi = 270.0;  ???

	phi=phi*pi/180.;
	float cosphi=cos(phi);
	int iupc=0;
	if(cosphi>=0)
	  iupc=0;//altered for GPU
	else
	  iupc=nxdx;
    
	float sinphi=sin(phi);
	int jupc=0;
	if(sinphi>=0.f)
	  jupc=0;//altered for GPU
	else
	  jupc=nydy;
    
	float phit=phi+0.5*pi;
	float cosphit=cos(phit);
	float sinphit=sin(phit);
    
	//Following variables are required for non-local mixing.
	float xcelt=0.f;
	float ycelt=0.f;
	int icelt=0;
	int jcelt=0;
    
	float xceln=0.f;
	float yceln=0.f;
	int iceln=0;
	int jceln=0;
	float utott=0.f;
	float delut=0.f;
	float delutz=0.f;
	xcb.resize(numBuild);
	ycb.resize(numBuild);
	icb.resize(numBuild);
	jcb.resize(numBuild);
	phib.resize(numBuild);
	zcorf.resize(nxnynz);
	uref.resize(nxnynz);
	urefu.resize(nxnynz);
	urefv.resize(nxnynz);
	urefw.resize(nxnynz);
	utotktp.resize(nxnynz);
	uktop.resize(nxnynz);
	vktop.resize(nxnynz);
	wktop.resize(nxnynz);
	deluc.resize(nxnynz);
	ustargz.resize(nxnynz); //for NLM -  non-local stress velocity scale in the vertical direction
	elzg.resize(nxnynz,(nxdx+1.)*dx);// initialized it with a value similar to QP
	utotcl1.resize(nxnynz);
	utotmax.resize(nxnynz);

	for(int i=0;i<numBuild;i++){
	  if(m_util_ptr->qpBuildoutData.buildings[i].type == 9)continue;
	  //! mdw 4-16-2004 added proper treatment of zfo
	  float temp=0.f;
	  int ktop=0;
	  for(int k=0;k<nzdz;k++){
            ktop=k;
            if(ht[i]+zfo[i]<z.at(k))break;
	  }
	  int kmid=0;
	  for(int k=0;k<nzdz;k++){        
            kmid=k;
            if(0.5*ht[i]+zfo[i]<z.at(k))break;
	  }
	  if(m_util_ptr->qpBuildoutData.buildings[i].type == 3){
            xcb.at(i)=xfo[i];
	  }
	  else{
            xcb.at(i)=xfo[i]+.5*lti[i];
	  }
	  ycb.at(i)=yfo[i];
	  temp=(xcb.at(i)-dx)/dx;//substracted dx to comply with GPU
	  icb.at(i)=nint(temp);
	  temp=(ycb.at(i)-dy)/dy;//substracted dy to comply with GPU
	  jcb.at(i)=nint(temp);
	  //!mdw 6-05-2005 put in procedure to calculate phi & phit
	  int kendv=0;
	  if(roofflag==2){
            float Bs=ht[i];
            float BL=wti[i];
            
            if(wti[i]<ht[i]){
	      Bs=wti[i];
	      BL=ht[i];
            }
	    // erp 5/20/2010
	    // The coefficients here are from Wilson 1979 ASHRAE paper
            float Rscale = ((pow(Bs,(2.f/3.f)))*(pow(BL,(1.f/3.f))));
            float temp=std::max(.22f*Rscale,.11f*wti[i]);
            float zclim  =std::max(temp,.11f*lti[i]);
            for(int k=0;k<nzdz;k++){        
	      kendv=k;
	      if(zclim<z.at(k))break;
            }
	  }
	  else{
            for(int k=0;k<nzdz;k++){        
	      kendv=k;
	      if(ht[i]+zfo[i]<z.at(k))break;
            }
	  }
	  kendv=std::min(kendv,nzdz);
        
	  int idvel=kendv*nxdx*nydy + jcb.at(i)*nxdx +icb.at(i);
	  double tempv=wind_vel[idvel].v;
	  double tempu=wind_vel[idvel].u;
	  phib.at(i)=atan2(tempv,tempu);
	  phi=phib.at(i);
	  cosphi=cos(phi);
	  int iupc=0;
	  if(cosphi>=0)
            iupc=0;//altered for GPU
	  else
            iupc=nxdx;
        
	  sinphi=sin(phi);
	  int jupc=0;
	  if(sinphi>=0)
            jupc=0;//altered for GPU
	  else
            jupc=nydy;
        
	  float phit=phi+0.5*pi;
	  cosphit=cos(phit);
	  sinphit=sin(phit);
        
	  //! ycbp3, and xcbp3 give points 1.5 units outside
	  //! of the bldg boundaries to compute reference utot
	  float ycbp3=0.f;
	  float xcbp3=0.f;
	  float ycbm3=0.f;
	  float xcbm3=0.f;
	  int icbp3=0;
	  int icbm3=0;
	  int jcbp3=0;
	  int jcbm3=0;
	  float dycbp3=0.f;
	  float dycbm3=0.f;
	  float dxcbp3=0.f;
	  float dxcbm3=0.f;
	  float ycbp=0.f;
	  float xcbp=0.f;
	  float ycbm=0.f;
	  float xcbm=0.f;
	  float xcd,ycd,xcu,ycu,xcul,ycul,cosfac;


	  if(fabs(sinphit)>=fabs(cosphit))
	    {
	      ycbp3=ycb.at(i)+(.5* m_util_ptr->qpBuildoutData.buildings[i].weff +.33* m_util_ptr->qpBuildoutData.buildings[i].weff )*sinphit;// ! Get reference values for x,y for non-local mixing
	      xcbp3=xcb.at(i)+(.5* m_util_ptr->qpBuildoutData.buildings[i].weff +.33* m_util_ptr->qpBuildoutData.buildings[i].weff )*cosphit;// ! 1/3 bldg width outside of building is the boundary for the non-local mixing
	      ycbm3=ycb.at(i)-(.5* m_util_ptr->qpBuildoutData.buildings[i].weff +.33* m_util_ptr->qpBuildoutData.buildings[i].weff )*sinphit;
	      xcbm3=xcb.at(i)-(.5* m_util_ptr->qpBuildoutData.buildings[i].weff +.33* m_util_ptr->qpBuildoutData.buildings[i].weff )*cosphit;
	      temp=(xcbp3-dx)/dx;
	      icbp3=nint(temp);//substracted dx to comply gpu
	      temp=(xcbm3-dx)/dx;
	      icbm3=nint(temp);//substracted dx to comply gpu
	      temp=(ycbp3-dy)/dy;
	      jcbp3=nint(temp);//substracted dx to comply gpu
	      temp=(ycbm3-dy)/dy;
	      jcbm3=nint(temp);//substracted dx to comply gpu
	      jcbp3=std::min(jcbp3,nydy-1);
	      jcbm3=std::min(jcbm3,nydy-1);
	      icbp3=std::min(icbp3,nxdx-1);
	      icbm3=std::min(icbm3,nxdx-1);
	      jcbp3=std::max(0,jcbp3);//changed from 1 to zeros to comply with gpu
	      jcbm3=std::max(0,jcbm3);
	      icbp3=std::max(0,icbp3);
	      icbm3=std::max(0,icbm3);
	      //! searching in the plus y direction for building free flow
	      int id=kmid*nxdx*nydy + jcbp3*nxdx +icbp3;
	      int jp1=0;
	      int jp2=0;
	      int isign=0;
	      if(retrieveCellTypeFromArray(id) == 0){
                if(sinphit>0.f){
		  jp1=jcbp3;
		  jp2=nydy-1;
		  isign=1;
                }
                else{
		  jp1=jcbp3;
		  jp2=0;//altered for GPU
		  isign=-1;
                }
            
                for(int ji=jp1;ji<=jp2;ji=ji+isign){
		  jcbp3=jcbp3+isign;
		  jcbp3=std::min(nydy-1,jcbp3);
		  dycbp3=dy*(jcbp3-1)-ycbp3;
		  ycbp3=dy*(jcbp3-1);
		  xcbp3=xcbp3+cosphit*dycbp3/sinphit;
		  icbp3=int(xcbp3/dx)+1-dx;
		  icbp3=std::min(nx-1,icbp3);
		  //!mdw 34/01/2004 forced indices to be within domain
		  int idMid=kmid*nxdx*nydy + jcbp3*nxdx +icbp3;
		  if(retrieveCellTypeFromArray(idMid)!= 0) break;
                }
	      }

	      //! searching in the minus y direction for building free flow
	      int id2=kmid*nxdx*nydy + jcbm3*nxdx +icbm3;
	      int jm2=0;
	      int jm1=0;
	      isign=0;
	      if(retrieveCellTypeFromArray(id2) == 0){
                if(sinphit>0.f){
		  jm2=0;//altered for GPU;
		  jm1=jcbm3;
		  isign=1;
                }
                else{
		  jm2=nydy-1;
		  jm1=jcbm3;
		  isign=-1;
                }
                for(int ji=jm1;ji>=jm2;ji=ji-isign){// do ji=jm1,jm2,-isign 
		  jcbm3=jcbm3-isign;
		  dycbm3=dy*(jcbm3-1)-ycbm3;
		  ycbm3=dy*(jcbm3-1);
		  xcbm3=xcbm3+cosphit*dycbm3/sinphit;
		  temp=(xcbm3-dx)/dx;
		  icbm3=nint(temp);
                                        
		  jcbp3=std::min(jcbp3,ny-1);
		  jcbm3=std::min(jcbm3,ny-1);
		  icbp3=std::min(icbp3,nx-1);
		  icbm3=std::min(icbm3,nx-1);
		  jcbp3=std::max(0,jcbp3);
		  jcbm3=std::max(0,jcbm3);
		  icbp3=std::max(0,icbp3);
		  icbm3=std::max(0,icbm3);
		  int idMid2=kmid*nxdx*nydy + jcbm3*nxdx +icbm3;
		  if(retrieveCellTypeFromArray(idMid2) != 0) break;
                }
	      }
	      ycbp=ycb.at(i)+(.5*m_util_ptr->qpBuildoutData.buildings[i].leff)*sinphi;
	      xcbp=xcb.at(i)+(.5*m_util_ptr->qpBuildoutData.buildings[i].leff)*cosphi;
	      ycbm=ycb.at(i)-(.5*m_util_ptr->qpBuildoutData.buildings[i].leff)*sinphi;
	      xcbm=xcb.at(i)-(.5*m_util_ptr->qpBuildoutData.buildings[i].leff)*cosphi;

            
	      if(cosphi>=0.f){
                //! Note the current upstream and downstream limits for the wake non-local mixing
                //! are 3*lr in the downstream direction and lfx upstream in the x direction
                //! and lfy upstream in the y direction
                xcd=xcb.at(i)+(.5*m_util_ptr->qpBuildoutData.buildings[i].leff+.1*dx)*cosphi; // ! get the first point on the center line outside of the building (downstream)
                ycd=ycb.at(i)+(.5*m_util_ptr->qpBuildoutData.buildings[i].leff+.1*dx)*sinphi;// !
                
                //!mdw 7-10-2006 made changes to xcd, ycd,xcu, & ycu - formerly used .5 dx
                if(m_util_ptr->qpBuildoutData.buildings[i].type == 3){
		  xcu=xcb.at(i)-(.4*m_util_ptr->qpBuildoutData.buildings[i].leff+dx)*cosphi;// ! (upstream)
		  ycu=ycb.at(i)-(.4*m_util_ptr->qpBuildoutData.buildings[i].leff+dx)*sinphi; //!
                }
                else{
		  xcu=xcb.at(i)-(.5*m_util_ptr->qpBuildoutData.buildings[i].leff+0.1*dx)*cosphi;// ! (upstream)
		  ycu=ycb.at(i)-(.5*m_util_ptr->qpBuildoutData.buildings[i].leff+0.1*dx)*sinphi;// !
                }
                 
		// Changes made to match the Windows 5.51 version of
		// gpuplume, in which turbulence seems to work better
		// at least for wind angles of 270 degrees.  It
		// appears that lr and lfr are swtiched in these
		// versions.  So, we're modifying to match the windows version.
		// -Pete

                //!mdw 7-05-2006 made changes to xcul & ycul - formerly used .5 dx

		// Was the following in Linux:
                // xcul=xcu-(m_util_ptr->qpBuildoutData.buildings[i].lr+dx)*cosphi;// ! get upper limit of the eddie
                // ycul=ycu-(m_util_ptr->qpBuildoutData.buildings[i].lr+dy)*sinphi;
		// Modified to match windows implementation below:
                xcul=xcu-(m_util_ptr->qpBuildoutData.buildings[i].lfr+dx)*cosphi;// ! get upper limit of the eddie
                ycul=ycu-(m_util_ptr->qpBuildoutData.buildings[i].lfr+dy)*sinphi;

                xcul=std::max(xcul,0.f);
                xcul=std::min(xcul,dx*(nxdx-1));
                ycul=std::max(ycul,0.f);
                ycul=std::min(ycul,dy*(nydy-1));
                cosfac=1.;
	      }
	      else{
                //!mdw 7-10-2006 made changes to xcd, ycd,xcu, & ycu - formerly used .5 dx
                xcd=xcb.at(i)+(.5*m_util_ptr->qpBuildoutData.buildings[i].leff+.1*dx)*cosphi;
                ycd=ycb.at(i)+(.5*m_util_ptr->qpBuildoutData.buildings[i].leff+.1*dx)*sinphi;
                if(m_util_ptr->qpBuildoutData.buildings[i].type == 3){
		  xcu=xcb.at(i)-(.4*m_util_ptr->qpBuildoutData.buildings[i].leff+dx)*cosphi;// ! (upstream)
		  ycu=ycb.at(i)-(.4*m_util_ptr->qpBuildoutData.buildings[i].leff+dx)*sinphi;// !
                }
                else{  
		  xcu=xcb.at(i)-(.5*m_util_ptr->qpBuildoutData.buildings[i].leff+0.1*dx)*cosphi;// ! (upstream)
		  ycu=ycb.at(i)-(.5*m_util_ptr->qpBuildoutData.buildings[i].leff+0.1*dx)*sinphi;// !
                }

		// Changes made to match the Windows 5.51 version of
		// gpuplume, in which turbulence seems to work better
		// at least for wind angles of 270 degrees.  It
		// appears that lr and lfr are swtiched in these
		// versions.  So, we're modifying to match the windows version.
		// -Pete
                //!mdw 7-05-2006 made changes to xcul & ycul - formerly used .5 dx

		// Was the following in Linux
                // xcul=xcu-(m_util_ptr->qpBuildoutData.buildings[i].lr+dx)*cosphi;// ! get upstream limit on the front cavity
                // ycul=ycu-(m_util_ptr->qpBuildoutData.buildings[i].lr+dy)*sinphi;// !
		// Modified to match windows implementation
		xcul=xcu-(m_util_ptr->qpBuildoutData.buildings[i].lfr+dx)*cosphi;// ! get upstream limit on the front cavity
                ycul=ycu-(m_util_ptr->qpBuildoutData.buildings[i].lfr+dy)*sinphi;// !		

                xcul=std::max(xcul,0.f);
                xcul=std::min(xcul,dx*(nxdx-1));
                ycul=std::max(ycul,0.f);
                ycul=std::min(ycul,dy*(nydy-1));
                cosfac=-1.;
	      }
	    }
	  else{// ! if you are more aligned with y than x
            //! MAN 9/15/2005 use weff and leff appropriately
            ycbp3=ycb.at(i)+(.5* m_util_ptr->qpBuildoutData.buildings[i].weff +.33* m_util_ptr->qpBuildoutData.buildings[i].weff )*sinphit;// ! get the effective length of the building
            xcbp3=xcb.at(i)+(.5* m_util_ptr->qpBuildoutData.buildings[i].weff +.33* m_util_ptr->qpBuildoutData.buildings[i].weff )*cosphit;
            ycbm3=ycb.at(i)-(.5* m_util_ptr->qpBuildoutData.buildings[i].weff +.33* m_util_ptr->qpBuildoutData.buildings[i].weff )*sinphit;
            xcbm3=xcb.at(i)-(.5* m_util_ptr->qpBuildoutData.buildings[i].weff +.33* m_util_ptr->qpBuildoutData.buildings[i].weff )*cosphit;
            //! end MAN 9/15/2005
            temp=(xcbp3-dx)/dx;
            icbp3=nint(temp);
            temp=(xcbm3-dx)/dx;
            icbm3=nint(temp);
            temp=(ycbp3-dy)/dy;
            jcbp3=nint(temp);
            temp=(ycbm3-dy)/dy;
            jcbm3=nint(temp);
            jcbp3=std::min(jcbp3,nydy-1);
            jcbm3=std::min(jcbm3,nydy-1);
            icbp3=std::min(icbp3,nxdx-1);
            icbm3=std::min(icbm3,nxdx-1);
            jcbp3=std::max(0,jcbp3);//altered from 1 to zero to comply GPU
            jcbm3=std::max(0,jcbm3);
            icbp3=std::max(0,icbp3);
            icbm3=std::max(0,icbm3);
            //! make sure you are outside of the building !
            int id=kmid*nxdx*nydy + jcbp3*nxdx + icbp3;
            int ip1=0;
            int ip2=0;
            int isign=0;
            
            if(retrieveCellTypeFromArray(id)== 0){
	      if(cosphit>0){
		ip1=icbp3;
		ip2=ny-1;
		isign=1;
	      }
	      else{
		ip1=icbp3;
		ip2=0;//altered for GPU
		isign=-1;
	      }
	      // ! decide which is closest building/floor
				
	      for(int ip=ip1;ip<=ip2;ip=ip+isign){//do ip=ip1,ip2,isign 
		icbp3=icbp3+isign;
		dxcbp3=dx*(icbp3-1)-xcbp3;
		xcbp3=dx*((icbp3-1));
		ycbp3=ycbp3+dxcbp3*sinphit/cosphit;
		temp=(ycbp3-dy)/dy;
		jcbp3=nint(temp);
		jcbp3=std::min(jcbp3,nydy-1);
		jcbm3=std::min(jcbm3,nydy-1);
		icbp3=std::min(icbp3,nxdx-1);
		icbm3=std::min(icbm3,nxdx-1);
		jcbp3=std::max(0,jcbp3);//altered for GPU
		jcbm3=std::max(0,jcbm3);//altered for GPU
		icbp3=std::max(0,icbp3);//altered for GPU
		icbm3=std::max(0,icbm3);//altered for GPU
		int idMid=kmid*nxdx*nydy + jcbp3*nxdx + icbp3;
		if(retrieveCellTypeFromArray(idMid)!= 0) break;
	      }
            }
            int id2=kmid*nxdx*nydy +jcbm3*nxdx + icbm3;
            
            if(retrieveCellTypeFromArray(id2) == 0){
	      int im1=0;
	      int im2=0;
	      isign=0;
	      if(cosphit>0.f){
		im1=icbm3;
		im2=0;//altered for GPU
		isign=1;
	      }
	      else{
		im1=icbm3;
		im2=nx-icbm3+1;
		isign=-1;
	      }
	      for(int im=im1;im<=im2;im=im+isign){//do im=im1,im2,-isign 
		icbm3=icbm3-isign;
		dxcbm3=dx*((icbm3-1))-xcbm3;
		xcbm3=dx*((icbm3-1));
		jcbm3=jcbm3+dxcbm3*sinphit/cosphit;
		jcbp3=std::min(jcbp3,ny-1);
		jcbm3=std::min(jcbm3,ny-1);
		icbp3=std::min(icbp3,nx-1);
		icbm3=std::min(icbm3,nx-1);
		jcbp3=std::max(0,jcbp3);
		jcbm3=std::max(0,jcbm3);
		icbp3=std::max(0,icbp3);
		icbm3=std::max(0,icbm3);
		int idMid2=kmid*nxdx*nydy + jcbm3*nxdx +icbm3;
		if(retrieveCellTypeFromArray(idMid2) != 0) break;
	      }
            }
            ycbp=ycb.at(i)+(.5*m_util_ptr->qpBuildoutData.buildings[i].leff)*sinphit;// !  get back of the building
            xcbp=xcb.at(i)+(.5*m_util_ptr->qpBuildoutData.buildings[i].leff)*cosphit;// !
            ycbm=ycb.at(i)-(.5*m_util_ptr->qpBuildoutData.buildings[i].leff)*sinphit;// !  get front of the building
            xcbm=xcb.at(i)-(.5*m_util_ptr->qpBuildoutData.buildings[i].leff)*cosphit;// !
            if(sinphi>=0.f){
	      //! Note the current upstream and downstream limits for the wake non-local mixing
	      //    ! are 3*lr in the downstream direction and lfx upstream in the x direction
	      //  ! and lfy upstream in the y direction
	      //! MAN 9/15/2005 use weff and leff appropriately
	      //!mdw 7-05-2006 made changes to xcu,ycu, xcd & ycd - formerly used .5 dy or .5 dx
	      xcd=xcb.at(i)+(.5*m_util_ptr->qpBuildoutData.buildings[i].leff+dy)*cosphi;// ! get the first point on the center line outside of the building (downstream)
	      ycd=ycb.at(i)+(.5*m_util_ptr->qpBuildoutData.buildings[i].leff+dy)*sinphi;// !
	      if(m_util_ptr->qpBuildoutData.buildings[i].type == 3){
		xcu=xcb.at(i)-(.4*m_util_ptr->qpBuildoutData.buildings[i].leff+dx)*cosphi;// ! (upstream)
		ycu=ycb.at(i)-(.4*m_util_ptr->qpBuildoutData.buildings[i].leff+dx)*sinphi; //!
	      }
	      else{
		xcu=xcb.at(i)-(.5*m_util_ptr->qpBuildoutData.buildings[i].leff+0.1*dx)*cosphi;// ! (upstream) 
		ycu=ycb.at(i)-(.5*m_util_ptr->qpBuildoutData.buildings[i].leff+0.1*dx)*sinphi;// !
	      }
	      //! end MAN 9/15/2005

	      // Changes made to match the Windows 5.51 version of
	      // gpuplume, in which turbulence seems to work better
	      // at least for wind angles of 270 degrees.  It
	      // appears that lr and lfr are swtiched in these
	      // versions.  So, we're modifying to match the windows version.
	      // -Pete

	      //! mdw 7-05-2006 eliminated .5 dx  or .5 dy in favor of dx & dy
	      // Was the following in Linux:
	      // xcul=xcu-(m_util_ptr->qpBuildoutData.buildings[i].lr+dx)*cosphi;// ! get upper limit of the eddie
	      // ycul=ycu-(m_util_ptr->qpBuildoutData.buildings[i].lr+dy)*sinphi;
	      // Modified to match windows implementation
	      xcul=xcu-(m_util_ptr->qpBuildoutData.buildings[i].lfr+dx)*cosphi;// ! get upper limit of the eddie
	      ycul=ycu-(m_util_ptr->qpBuildoutData.buildings[i].lfr+dy)*sinphi;	      

	      xcul=std::max(xcul,0.f);
	      xcul=std::min(xcul,dx*(nxdx-1));
	      ycul=std::max(ycul,0.f);
	      ycul=std::min(ycul,dy*(nydy-1));
	      cosfac=1.f;
            }
            else{
	      //! MAN 9/15/2005 use weff and leff appropriately
	      xcd=xcb.at(i)+(.5*m_util_ptr->qpBuildoutData.buildings[i].leff+dy)*cosphi;
	      ycd=ycb.at(i)+(.5*m_util_ptr->qpBuildoutData.buildings[i].leff+dy)*sinphi;
	      if(m_util_ptr->qpBuildoutData.buildings[i].type == 3){
		xcu=xcb.at(i)-(.4*m_util_ptr->qpBuildoutData.buildings[i].leff+dx)*cosphi;// ! (upstream)
		ycu=ycb.at(i)-(.4*m_util_ptr->qpBuildoutData.buildings[i].leff+dx)*sinphi;// !
	      }
	      else{
		xcu=xcb.at(i)-(.5*m_util_ptr->qpBuildoutData.buildings[i].leff+dx)*cosphi;// ! (upstream) 
		ycu=ycb.at(i)-(.5*m_util_ptr->qpBuildoutData.buildings[i].leff+dx)*sinphi;// !
	      }
	      //! end MAN 9/15/2005
                
	      // Changes made to match the Windows 5.51 version of
	      // gpuplume, in which turbulence seems to work better
	      // at least for wind angles of 270 degrees.  It
	      // appears that lr and lfr are swtiched in these
	      // versions.  So, we're modifying to match the windows version.
	      // -Pete

	      // Was the following in linux:
	      // xcul=xcu+(m_util_ptr->qpBuildoutData.buildings[i].lr+dx)*cosphi;// ! get upstream limit on the front cavity
	      // ycul=ycu+(m_util_ptr->qpBuildoutData.buildings[i].lr+dy)*sinphi;// !
	      // Modified to match the windows implementation:
	      xcul=xcu+(m_util_ptr->qpBuildoutData.buildings[i].lfr+dx)*cosphi;// ! get upstream limit on the front cavity
	      ycul=ycu+(m_util_ptr->qpBuildoutData.buildings[i].lfr+dy)*sinphi;// !


	      xcul=std::max(xcul,0.f);
	      xcul=std::min(xcul,dx*(nxdx-1));
	      ycul=std::max(ycul,0.f);
	      ycul=std::min(ycul,dy*(nydy-1));
	      cosfac=-1.f;
            }
	  }
	  //!mdw 7-05-2006 change form to ixxx or jxxx =nint()+1
	  temp=(xcd-dx)/dx;//altered to comply with GPU
	  int icd=nint(temp)+1;// ! get indicies for the downstream center line to back of the building
	  temp=(ycd-dy)/dy;//altered to comply with GPU
	  int jcd=nint(temp)+1;
	  //!mdw 4-16-2004 added correction for ktop+3 > nz-1
	  int ktp=std::min(ktop,nzdz-1);//didn't alter here as ktop is already aligned with GPU coordinates
	  float zk=0.f;
	  float zbrac=0.f;
	  float zkfac=0.f;
	  float xcdl=0.f;
	  float ycdl=0.f;
	  int icdl=0;
	  int jcdl=0;
	  int icu=0;
	  int jcu=0;
	  int icul=0;
	  int jcul=0;
	  float urefz=0.f;
	  float ds=0.f;
	  float sdown=0.f;
	  float sup=0.f;
	  float stin=0.f;
	  float istinf=0.f;
	  float st=0.f;
	  int istf=0;
	  int isf=0;
	  int isfu=0;
	  float utotp=0.f;
	  float utotm=0.f;
	  float cosu=0.f;
	  float sinv=0.f;
	  int isini=0;
	  float cosl=0.f;
	  float sinl=0.f;
	  float delutz=0.f;
	  float upvpg=0.f;
	  float upwpg=0.f;
	  float upsqg=0.f;
	  float vpsqg=0.f;
	  float vpwpg=0.f;
	  float wpsqg=0.f;
	  float duy=0.f;
        
	  for(int k=ktp;k>=0;k--){//do k=ktp,2,-1  ! Account for wake difference in the cavity
            
            zk=zm.at(k);
            if(zi.at(k)<.99*h){
	      zbrac=pow( (1.f-zi.at(k)/h) , 1.5f);
            }
            else{
	      zbrac=pow( (1.f-.99f),1.5f);
            }
            //zbrac=pow( (1.f-zi.at(k)/h) , 1.5f);
            //!mdw 4-16-2004 added correction for ktop+3 > nz-1
            int idupc=k*nxdx*nydy + jupc*nxdx +iupc;
            int idupcktop=(ktop+3)*nxdx*nydy + jupc*nxdx +iupc;
            int idupcnzm1=(nzdz-1)*nxdx*nydy + jupc*nxdx +iupc;
            if(ktop+3<=nzdz-1){
	      zcorf.at(k)=sqrt(wind_vel[idupc].u*wind_vel[idupc].u + wind_vel[idupc].v*wind_vel[idupc].v + wind_vel[idupc].w*wind_vel[idupc].w)/
		sqrt(wind_vel[idupcktop].u*wind_vel[idupcktop].u + wind_vel[idupcktop].v*wind_vel[idupcktop].v
		     + wind_vel[idupcktop].w*wind_vel[idupcktop].w);
            }
            else{
	      zcorf.at(k)=sqrt(wind_vel[idupc].u*wind_vel[idupc].u + wind_vel[idupc].v*wind_vel[idupc].v + wind_vel[idupc].w*wind_vel[idupc].w)/
		sqrt(wind_vel[idupcnzm1].u*wind_vel[idupcnzm1].u + wind_vel[idupcnzm1].v*wind_vel[idupcnzm1].v
		     + wind_vel[idupcnzm1].w*wind_vel[idupcnzm1].w);
            }
                        
            
            //! mdw 4-16-2004 added proper treatment of zfo
            if(zk<ht[i]+zfo[i]){
	      zkfac=sqrt(1.-pow((zk/(ht[i]+zfo[i])),2));
            }
            else{
	      if(k==ktp){
		zkfac=1.;
	      }
	      else{
		zkfac=0.f;
	      }
                
            }
            //! mdw 7-05-2006 changed from .5 dx or .5 dy to dx & dy to be consistent with nint
	    // Apparently, this is supposed to be lr as it matches the windows implementation... ??? -Pete
            xcdl=xcd+(3.*m_util_ptr->qpBuildoutData.buildings[i].lr+dx)*zkfac*cosphi;// ! calculate the x,y limit of the wake as a function of height
            ycdl=ycd+(3.*m_util_ptr->qpBuildoutData.buildings[i].lr+dy)*zkfac*sinphi;// !
            xcdl=std::min(xcdl,dx*(nxdx));
            ycdl=std::min(ycdl,dy*(nydy));
            xcdl=std::max(xcdl,0.f);
            ycdl=std::max(ycdl,0.f);
            
            temp=(xcdl-dx)/dx;//altered for GPU, substracted dx, same below
            icdl=nint(temp)+1;// ! Calculate the indicies for i,j according to xcdl,ycdl
            temp=(ycdl-dy)/dy;
            jcdl=nint(temp)+1;
            temp=(xcu-dx)/dx;
            icu=nint(temp)+1;//   ! indicies for the upstream cavity (building)
            temp=(ycu-dy)/dy;
            jcu=nint(temp)+1;//   !
            temp=(xcul-dx)/dx;
            icul=nint(temp)+1;// ! (furthest upstream)
            temp=(ycul-dy)/dy;
            jcul=nint(temp)+1;// !!!
            //!mdw 4-16-2004 added correction for ktop+3 > nz-1
            int idktop=(ktop+3)*nxdx*nydy + jcb.at(i)*nxdx +icb.at(i);
            if(ktop+3<=nzdz-1){
	      //! calculating the reference wind un-disturbed by the building
	      urefz=sqrt(wind_vel[idktop].u*wind_vel[idktop].u + wind_vel[idktop].v*wind_vel[idktop].v + wind_vel[idktop].w*wind_vel[idktop].w);
            }
            else{
	      urefz=sqrt(pow(ualoft,2.f)+pow(valoft,2.f));
            }
            ds=0.7*std::min(dx,dy);// ! pick a step that is small enough to not skip grid cells
            sdown=sqrt((xcdl-xcd)*(xcdl-xcd)+(ycdl-ycd)*(ycdl-ycd))+2.*ds;// ! calculate the limits for the distance measured along the centerline (rear)
            sup=sqrt((xcul-xcu)*(xcul-xcu)+(ycul-ycu)*(ycul-ycu))+2.*ds;//   ! same for the front eddy
            stin=.5*m_util_ptr->qpBuildoutData.buildings[i].leff;//
            temp=stin/ds;
            istinf=nint(temp)+1.f;
            //!mdw 7-11-2006 changed istinf to allow replacement to center of bldg
            //!mdw 5-14-2004 corrected expression for st; older versions gave errors for wide blds
            st=sqrt((xcbp3-xcb.at(i))*(xcbp3-xcb.at(i))+(ycbp3-ycb.at(i))*(ycbp3-ycb.at(i)))+1.*ds;// ! total distance to point
	    temp=(st+.333*m_util_ptr->qpBuildoutData.buildings[i].leff)/ds;
            istf=nint(temp)+1.f;//   ! (transverse direction) 
            //!mdw 6-9-2004 extended the transverse integration to st+.333*leff
            temp=sdown/ds;
            isf=nint(temp)+1;// ! setup limits of calculations (for do loops) (along cneterline down)  
            temp=sup/ds;
            isfu=nint(temp)+1;//  ! (along centerline up) 

	    
                 
	    // Changes made to match the Windows 5.51 version of
	    // gpuplume, in which turbulence seems to work better
	    // at least for wind angles of 270 degrees.  It
	    // appears that lr and lfr are swtiched in these
	    // versions.  So, we're modifying to match the windows version.
	    // -Pete
	    // Was the following under Linux:
            // if(m_util_ptr->qpBuildoutData.buildings[i].lr < 0.f)isfu=0;
	    // Modified to match windows implementation:
            if(m_util_ptr->qpBuildoutData.buildings[i].lfr < 0.f)isfu=0;
            
            //!mdw 4-16-2004 added correction for ktop+3 > nz-1
            
            //! Select the largest reference wind of the plus or minus side of the building
            int id1=k*nxdx*nydy + jcbp3*nxdx +icbp3;
            int id2=k*nxdx*nydy + jcbm3*nxdx +icbm3;

            utotp=sqrt(wind_vel[id1].u*wind_vel[id1].u + wind_vel[id1].v*wind_vel[id1].v + wind_vel[id1].w*wind_vel[id1].w);
            utotm=sqrt(wind_vel[id2].u*wind_vel[id2].u + wind_vel[id2].v*wind_vel[id2].v + wind_vel[id2].w*wind_vel[id2].w);
            int ik=k*nxdx*nydy + i;
            int idp=k*nxdx*nydy + jcbp3*nxdx +icbp3;
            int idm=k*nxdx*nydy + jcbm3*nxdx +icbm3;
            if(utotp>=utotm){
	      uref.at(ik)=utotp+.000001;
	      urefu.at(ik)=uref.at(ik)*cos(phib.at(i));
	      urefv.at(ik)=uref.at(ik)*sin(phib.at(i));
	      urefw.at(ik)=wind_vel[idp].w;
            }
            else{
	      uref.at(ik)=utotm+.000001;
	      urefu.at(ik)=uref.at(ik)*cos(phib.at(i));
	      urefv.at(ik)=uref.at(ik)*sin(phib.at(i));
	      urefw.at(ik)=wind_vel[idm].w;
            }
            //!!!!!!!
            cosu=(urefu.at(ik)+.000001)/uref.at(ik);
            sinv=urefv.at(ik)/uref.at(ik);
            //! downstream wake  along axis do loop for delta u
            isini=1;
            float xcell=0.f;
            float ycell=0.f;
            int icel=0;
            int jcel=0;
            float utot=0.f;
            for(int is=1;is<=isf;is++){//   do is=1,isf 
	      xcell=xcd+ds*(is-1)*cosphi;
	      ycell=ycd+ds*(is-1)*sinphi;
	      temp=(xcell-dx)/dx;//substracted dx for GPU
	      icel=nint(temp)+1;
	      temp=(ycell-dy)/dy;//substracted dy for GPU
	      jcel=nint(temp)+1;
	      icel=std::min(nxdx-1,icel);
	      icel=std::max(1,icel);//altered for GPU (2 to 1)
	      jcel=std::min(nydy-1,jcel);
	      jcel=std::max(1,jcel);//altered for GPU (2 to 1)
	      int id=k*nxdx*nydy + jcel*nxdx +icel;
	      if(retrieveCellTypeFromArray(id) == 0 && is==1){
		isini=2;
	      }
	      utot=sqrt(wind_vel[id].u*wind_vel[id].u + wind_vel[id].v*wind_vel[id].v + wind_vel[id].w*wind_vel[id].w);
	      //!mdw 4-16-2004 added correction for ktop+3 > nz-1
	      int iceljcel=jcel*nxdx +icel;
	      int idcel=ktop*nxdx*nydy + jcel*nxdx +icel;
                
	      if(k==ktp){
		if(ktop<=nzdz-1){
		  utotktp.at(iceljcel)=utot;
		  uktop.at(iceljcel)=wind_vel[idcel].u;
		  vktop.at(iceljcel)=wind_vel[idcel].v;
		  wktop.at(iceljcel)=wind_vel[idcel].w;
		}
		else{
		  utotktp.at(iceljcel)=sqrt(ualoft*ualoft+valoft*valoft); //check:: compare with QP, may be a bug in QP
		  uktop.at(iceljcel)=ualoft;
		  vktop.at(iceljcel)=valoft;
		  wktop.at(iceljcel)=0.f;
		}
	      }
	      //! this sets reference for vertical transfer
	      utot=utot+.000001;
	      int idcelk=k*nxdx*nydy +jcel*nxdx +icel;
	      int ik=k*nxdx*nydy+i;
	      cosl=wind_vel[idcelk].u/utot;
	      sinl=wind_vel[idcelk].v/utot;
	      if(retrieveCellTypeFromArray(idcelk) > 0){
		delutz=sqrt( pow( (wind_vel[idcelk].u-zcorf.at(k)*uktop.at(iceljcel)),2.f)
			     +pow( (wind_vel[idcelk].v -zcorf.at(k)*vktop.at(iceljcel)),2.f)
			     +pow( (wind_vel[idcelk].w -zcorf.at(k)*wktop.at(iceljcel)),2.f) );
		deluc.at(ik)=sqrt( pow( (urefu.at(ik)-wind_vel[idcelk].u),2.f)
				   +pow( (urefv.at(ik)-wind_vel[idcelk].v),2.f)
				   +pow( (urefw.at(ik)-wind_vel[idcelk].w),2.f));
		//!mdw 4-16-2004 added correction for ktop+3 > nz-1
		if(k!=ktp){
		  //! Selects the largest gradient (vert or horiz transfer)
		  //! mdw 4-16-2004 added proper treatment of zfo
                        
		  if((2.*deluc.at(ik)/ m_util_ptr->qpBuildoutData.buildings[i].weff )<(utotktp.at(iceljcel)/(ht[i]+zfo[i])) &&
		     delutz>.2*zcorf.at(k)*utotktp.at(iceljcel)){// ! vertical dominates
		    ustargz.at(idcelk)=std::max(knlc*utotktp.at(iceljcel),ustargz.at(idcelk)); 
		    if(fabs(ustargz.at(idcelk)-knlc*utotktp.at(iceljcel))<1.e-05*ustargz.at(idcelk)){//!This value dominates over prev. buildings.
		      elzg.at(idcelk)=ht[i]+zfo[i];
		      upvpg=0.f;
		      upwpg=-ctau13*zbrac*ustargz.at(idcelk)*ustargz.at(idcelk);
		      upsqg=cusq*zbrac*ustargz.at(idcelk)*ustargz.at(idcelk);
		      vpsqg=cvsq*zbrac*ustargz.at(idcelk)*ustargz.at(idcelk);
		      vpwpg=0.f;
		      wpsqg=cwsq*zbrac*ustargz.at(idcelk)*ustargz.at(idcelk);
		      ustarg.at(idcelk)=ustargz.at(idcelk);
		      rotate2d(idcelk,cosphi,sinphi,upsqg,upvpg,vpsqg,wpsqg,upwpg,vpwpg);
		    }
		  }
		  else{
		    //! We use the vertical gradient as dominant if it is sharper than the horizontal
		    duy=-deluc.at(ik)*sinl*cosu+deluc.at(ik)*sinv*cosl;
		    //! we now have the delta u between the outside of the bldg and the center of the wake
		    //! mdw 6-10-2004 removed
		    if(deluc.at(ik)>.2*uref.at(ik)){
		      ustarg.at(idcelk)=std::max(ustarg.at(idcelk),knlc*deluc.at(ik));
		      if(fabs(ustarg.at(idcelk)-knlc*deluc.at(ik))<1.e-05*ustarg.at(idcelk)){// ! if the horiz is dominant calculate sigmas
			upvpg=0.f;
			//! on axis u prime v prime is zero
			upwpg=0.f;
			//! for eddy transport in uv we dont consider uw
			upsqg=cusq*zbrac*ustarg.at(idcelk)*ustarg.at(idcelk);
			wpsqg=cvsq*zbrac*ustarg.at(idcelk)*ustarg.at(idcelk);
			vpwpg=0.f;
			elzg.at(idcelk)=0.5* m_util_ptr->qpBuildoutData.buildings[i].weff ;
			vpsqg=cwsq*zbrac*ustarg.at(idcelk)*ustarg.at(idcelk);
			rotate2d(idcelk,cosphi,sinphi,upsqg,upvpg,vpsqg,wpsqg,upwpg,vpwpg);
		      }
		    }
		  }
		}
	      }
	      else{
		deluc.at(ik)=0.;
		delutz=0.;
	      }
	      //! transverse do loop in downstream wake
                
	      for(int ist=2;ist<=istf;ist++){//do ist=2,istf 
		//! first direction in the transverse of the wake
		xcelt=xcell+ds*(ist-1.f)*cosphit;
		ycelt=ycell+ds*(ist-1.f)*sinphit;
		temp=(xcelt-dx)/dx;
		icelt=nint(temp)+1;
		temp=(ycelt-dy)/dy;
		jcelt=nint(temp)+1;
		if(fabs(xcelt-xcell)<.5f*ds)icelt=icel;
		if(fabs(ycelt-ycell)<.5f*ds)jcelt=jcel;
		icelt=std::min(nxdx-1,icelt);
		icelt=std::max(1,icelt);
		jcelt=std::min(nydy-1,jcelt);
		jcelt=std::max(1,jcelt);
		int iceltjcelt=jcelt*nxdx + icelt;
		int idceltk= k*nxdx*nydy + jcelt*nxdx +icelt;
		if(retrieveCellTypeFromArray(idceltk) > 0){
		  utott=sqrt(wind_vel[idceltk].u*wind_vel[idceltk].u + wind_vel[idceltk].v*wind_vel[idceltk].v
			     + wind_vel[idceltk].w*wind_vel[idceltk].w);
		  utott=utott+.000001f;
		  //!mdw 4-16-2004 added correction for ktop+3 > nz-1

		  int idceltktop=ktop*nxdx*nydy + jcelt*nxdx +icelt;
		  if(k==ktp){
		    if(ktop<nzdz-1){
		      utotktp.at(iceltjcelt)=utott;
		      uktop.at(iceltjcelt)=wind_vel[idceltktop].u;
		      vktop.at(iceltjcelt)=wind_vel[idceltktop].v;
		      wktop.at(iceltjcelt)=wind_vel[idceltktop].w;
		    }
		    else{
		      utotktp.at(iceltjcelt)=sqrt(ualoft*ualoft+valoft*valoft);
		      uktop.at(iceltjcelt)=ualoft;
		      vktop.at(iceltjcelt)=valoft;  
		      wktop.at(iceltjcelt)=0.;
		    }
		  }
		  int ik=k*nxdx*nydy +i;
		  delut=sqrt(pow( (urefu.at(ik)-wind_vel[idceltk].u),2.f)+
			     pow( (urefv.at(ik)-wind_vel[idceltk].v),2.f)+
			     pow( (urefw.at(ik)-wind_vel[idceltk].w),2.f));
		  delutz=sqrt(pow( (wind_vel[idceltk].u-zcorf.at(k)*uktop.at(iceltjcelt)),2.f)
			      +pow( (wind_vel[idceltk].v-zcorf.at(k)*vktop.at(iceltjcelt)),2.f)
			      +pow( (wind_vel[idceltk].w-zcorf.at(k)*wktop.at(iceltjcelt)),2.f));
		  //!mdw 4-16-2004 added correction for ktop+3 > nz-1
		  if(k!=ktp){
		    //! mdw 4-16-2004 added proper treatment of zfo
		    //! mdw 6-10-2004 changed to make check on centerline rather than local value
		    int ik=k*nxdx*nydy +i;
		    if((2.*deluc.at(ik)/ m_util_ptr->qpBuildoutData.buildings[i].weff )<(utotktp.at(iceltjcelt)/(ht[i]+zfo[i])) 
		       && delutz>.2*zcorf.at(k)*utotktp.at(iceltjcelt)){
		      if(ustargz.at(idceltk)<knlc*utotktp.at(iceltjcelt)){
			ustargz.at(idceltk)=knlc*utotktp.at(iceltjcelt);
			elzg.at(idceltk)=ht[i]+zfo[i];
			upvpg=0.;
			upwpg=-ctau13*zbrac*ustargz.at(idceltk)*ustargz.at(idceltk);
			upsqg=cusq*zbrac*ustargz.at(idceltk)*ustargz.at(idceltk);
			vpsqg=cvsq*zbrac*ustargz.at(idceltk)*ustargz.at(idceltk);
			vpwpg=0.;
			wpsqg=cwsq*zbrac*ustargz.at(idceltk)*ustargz.at(idceltk);
			ustarg.at(idceltk)=ustargz.at(idceltk);
			rotate2d(idceltk,cosphi,sinphi,upsqg,upvpg,vpsqg,wpsqg,upwpg,vpwpg);
		      }
		    }
		    else{
		      // We use the vertical gradient as dominant if it is sharper than the horizontal
		      cosl=wind_vel[idceltk].u/utott;
		      sinl=wind_vel[idceltk].v/utott;
		      duy=-deluc.at(ik)*sinl*cosu+deluc.at(ik)*sinv*cosl;
		      // mdw 6-10-2004 changed check from delut (local value) to deluc.at(ik); centerline
		      if(delut>.2*uref.at(ik)){
			if(ustarg.at(idceltk)<knlc*deluc.at(ik)){
			  ustarg.at(idceltk)=knlc*deluc.at(ik);
			  upvpg=-((ist-1.f)/(istf-1.f))*ustarg.at(idceltk)*ustarg.at(idceltk);
			  upwpg=0.;
			  // for eddy transport in uv we dont consider uw
			  upsqg=cusq*zbrac*ustarg.at(idceltk)*ustarg.at(idceltk);
			  wpsqg=cvsq*zbrac*ustarg.at(idceltk)*ustarg.at(idceltk);
			  vpwpg=0.;
			  vpsqg=cwsq*zbrac*ustarg.at(idceltk)*ustarg.at(idceltk);
			  elzg.at(idceltk)=.5* m_util_ptr->qpBuildoutData.buildings[i].weff ;
			  rotate2d(idceltk,cosphi,sinphi,upsqg,upvpg,vpsqg,wpsqg,upwpg,vpwpg);
			}
		      }
		    }
		    if(is==isini){
		      for(int isin=isini+1;isin<=istinf;isin++){//do isin=isini+1,istinf
			xceln=xcelt-ds*(isin-1)*cosphi;
			yceln=ycelt-ds*(isin-1)*sinphi;
			temp=(xceln-dx)/dx;
			iceln=nint(temp)+1;
			temp=(yceln-dy)/dy;
			jceln=nint(temp)+1;
			iceln=std::min(nxdx-1,iceln);
			iceln=std::max(1,iceln);
			jceln=std::min(nydy-1,jceln);
			jceln=std::max(1,jceln);
			// mdw 3/22/2004PM added if statement to avoid replacing non-zero ustarg stuff
			// with zero values
			int idcelnk= k*nxdx*nydy + jceln*nxdx +iceln;
			if(ustarg.at(idceltk)>ustarg.at(idcelnk)){
			  ustarg.at(idcelnk)=ustarg.at(idceltk);
			  elzg.at(idcelnk)=elzg.at(idceltk);
			  ufsqgi.at(idcelnk)=ufsqgi.at(idceltk);
			  vfsqgi.at(idcelnk)=vfsqgi.at(idceltk);
			  wfsqgi.at(idcelnk)=wfsqgi.at(idceltk);
			  ufvfgi.at(idcelnk)=ufvfgi.at(idceltk);
			  ufwfgi.at(idcelnk)=ufwfgi.at(idceltk);
			  vfwfgi.at(idcelnk)=vfwfgi.at(idceltk);
			}
			// mdw 3/22/2004PM new endif for new if then
		      }//enddo
		    }
		  }
		}
		// opposite direction in the transverse of the wake
		xcelt=xcell-ds*(ist-1.f)*cosphit;
		ycelt=ycell-ds*(ist-1.f)*sinphit;
		temp=(xcelt-dx)/dx;
		icelt=nint(temp)+1; 
		temp=(ycelt-dy)/dy;
		jcelt=nint(temp)+1; 
		if(fabs(xcelt-xcell)<.5*ds)icelt=icel;
		if(fabs(ycelt-ycell)<.5*ds)jcelt=jcel;
		icelt=std::min(nxdx-1,icelt); 
		icelt=std::max(1,icelt);
		jcelt=std::min(nydy-1,jcelt);
		jcelt=std::max(1,jcelt);
                    
		iceltjcelt=jcelt*nxdx + icelt;
		int idceltktop=ktop*nxdx*nydy + jcelt*nxdx +icelt;
		idceltk=k*nxdx*nydy + jcelt*nxdx +icelt;
		if(retrieveCellTypeFromArray(idceltk) > 0){
		  utott=sqrt(wind_vel[idceltk].u*wind_vel[idceltk].u+wind_vel[idceltk].v*wind_vel[idceltk].v
			     +wind_vel[idceltk].w*wind_vel[idceltk].w);
		  utott=utott+.000001;
		  delut=sqrt(pow( (urefu.at(ik)-wind_vel[idceltk].u),2)
			     + pow( (urefv.at(ik)-wind_vel[idceltk].v),2)
			     + pow( (urefw.at(ik)-wind_vel[idceltk].w),2));
		  // mdw 4-16-2004 added correction for ktop+3 > nz-1
                        
		  if(k==ktp){
		    if(ktop<=nzdz-1){
		      utotktp.at(iceltjcelt)=utott;
		      uktop.at(iceltjcelt)=wind_vel[idceltktop].u;
		      vktop.at(iceltjcelt)=wind_vel[idceltktop].v;
		      wktop.at(iceltjcelt)=wind_vel[idceltktop].w;
		    }
		    else{
		      utotktp.at(iceltjcelt)=sqrt(ualoft*ualoft+valoft*valoft);
		      uktop.at(iceltjcelt)=ualoft;
		      vktop.at(iceltjcelt)=valoft;
		      wktop.at(iceltjcelt)=0.;
		    }
		  }
		  // mdw 4-16-2004 added correction for ktop+3 > nz-1
                        
		  if(k!=ktp){
		    delutz=sqrt(pow( (wind_vel[idceltk].u-zcorf.at(k)*uktop.at(iceltjcelt)),2)
				+pow( (wind_vel[idceltk].v-zcorf.at(k)*vktop.at(iceltjcelt)),2)
				+pow( (wind_vel[idceltk].w-zcorf.at(k)*wktop.at(iceltjcelt)),2));
		    // mdw 4-16-2004 added proper treatment of zfo
		    // mdw 6-10-2004 made check on centerline rather than local value
		    if((2.*deluc.at(ik)/ m_util_ptr->qpBuildoutData.buildings[i].weff )<(utotktp.at(iceltjcelt)/(ht[i]+zfo[i])) 
		       && delutz>.2*zcorf.at(k)*utotktp.at(iceltjcelt)){
		      if(ustargz.at(idceltk)<knlc*utotktp.at(iceltjcelt)){
			ustargz.at(idceltk)=knlc*utotktp.at(iceltjcelt);
			elzg.at(idceltk)=ht[i]+zfo[i];
			upvpg=0.;
			upwpg=-ctau13*zbrac*ustargz.at(idceltk)*ustargz.at(idceltk);
			upsqg=cusq*zbrac*ustargz.at(idceltk)*ustargz.at(idceltk);
			vpsqg=cvsq*zbrac*ustargz.at(idceltk)*ustargz.at(idceltk);
			vpwpg=0.;
			wpsqg=cwsq*zbrac*ustargz.at(idceltk)*ustargz.at(idceltk);
			ustarg.at(idceltk)=ustargz.at(idceltk);
			rotate2d(idceltk,cosphi,sinphi,upsqg,upvpg,vpsqg,wpsqg,upwpg,vpwpg);
		      }
		    }
		    else{
		      // We use the vertical gradient as dominant if it is sharper than the horizontal
		      cosl=wind_vel[idceltk].u/utott;
		      sinl=wind_vel[idceltk].v/utott;
		      duy=-deluc.at(ik)*sinl*cosu+deluc.at(ik)*sinv*cosl; 
		      // mdw 6-10-2004 made check on centerline value rather than local value
                                
		      if(delut>.2f*uref.at(ik)){
			if(ustarg.at(idceltk)<knlc*deluc.at(ik)){
			  ustarg.at(idceltk)=knlc*deluc.at(ik);
			  upvpg=((ist-1.f)/(istf-1.f))*ustarg.at(idceltk)*ustarg.at(idceltk);
			  upwpg=0.f;
			  // for eddy transport in uv we dont consider uw
			  upvpg=ctau13*zbrac*((ist-1.f)/(istf-1.f))*ustarg.at(idceltk)*ustarg.at(idceltk);
			  upwpg=0.f;
			  // for eddy transport in uv we dont consider uw
			  upsqg=cusq*zbrac*ustarg.at(idceltk)*ustarg.at(idceltk);
			  wpsqg=cvsq*zbrac*ustarg.at(idceltk)*ustarg.at(idceltk);
			  elzg.at(idceltk)=0.5f* m_util_ptr->qpBuildoutData.buildings[i].weff ;
			  vpwpg=0.f;
			  vpsqg=cwsq*zbrac*ustarg.at(idceltk)*ustarg.at(idceltk);
			  rotate2d(idceltk,cosphi,sinphi,upsqg,upvpg,vpsqg,wpsqg,upwpg,vpwpg);
			}
		      }
		    }
		    if(is==isini){
		      for(int isin=isini+1;isin<=istinf;isin++){
			xceln=xcelt-ds*(isin-1)*cosphi;
			yceln=ycelt-ds*(isin-1)*sinphi;
			temp=(xceln-dx)/dx;
			iceln=nint(temp)+1; 
			temp=(yceln-dy)/dy;
			jceln=nint(temp)+1; 
			iceln=std::min(nxdx-1,iceln);
			iceln=std::max(1,iceln);
			jceln=std::min(nydy-1,jceln);
			jceln=std::max(1,jceln);
			int idcelnk=k*nxdx*nydy +jceln*nxdx +iceln;
			// mdw 3/22/2004pm adding new if then structure to avoid replacing non-zero
			// ustarg with zero ones
			if(ustarg.at(idceltk)>ustarg.at(idcelnk)){
			  ustarg.at(idcelnk)=ustarg.at(idceltk);
			  elzg.at(idcelnk)=elzg.at(idceltk); 
			  ufsqgi.at(idcelnk)=ufsqgi.at(idceltk);
			  vfsqgi.at(idcelnk)=vfsqgi.at(idceltk);
			  wfsqgi.at(idcelnk)=wfsqgi.at(idceltk);
			  ufvfgi.at(idcelnk)=ufvfgi.at(idceltk);
			  ufwfgi.at(idcelnk)=ufwfgi.at(idceltk);
			  vfwfgi.at(idcelnk)=vfwfgi.at(idceltk);
			}
		      }//enddo
		      // mdw 3/22/2004 end of new if then structure
		    }//endif
		  }//endif
		}
	      }   //lp021
            }  //lp022

            isini=1;
            for(int is=1; is<=isfu;is++){//do is=1,isfu
	      // upstream front eddy along the centerline
	      xcell=xcu-ds*(is-1.f)*cosphi;
	      ycell=ycu-ds*(is-1.f)*sinphi;
	      //mdw 7-05-2006 changed form form =nint( / ) to nint( / )+1
	      temp=(xcell-dx)/dx;
	      icel=nint(temp)+1; 
	      temp=(ycell-dy)/dy;
	      jcel=nint(temp)+1; 
	      icel=std::min(nxdx-1,icel);
	      icel=std::max(1,icel);
	      jcel=std::min(nydy-1,jcel);
	      jcel=std::max(1,jcel);
	      int idcelk=k*nxdx*nydy +jcel*nxdx +icel;
	      int iceljcel=jcel*nxdx +icel;
	      if(retrieveCellTypeFromArray(idcelk) == 0 && is == 1){
		isini=2;
	      }
	      int idcelktop=ktop*nxdx*nydy + jcel*nxdx +icel;
	      idcelk=k*nxdx*nydy + jcel*nxdx +icel;
	      utot=sqrt(wind_vel[idcelk].u*wind_vel[idcelk].u+wind_vel[idcelk].v*wind_vel[idcelk].v+wind_vel[idcelk].w*wind_vel[idcelk].w);
	      // mdw 1-22-2004 new lines in support of bldg infiltration
	      if((k==kmid)&&(is==1))utotcl1.at(i)=utot;
	      if((k==kmid)&&(utot>utotmax.at(i)))utotmax.at(i)=utot; 
	      utot=utot+.000001;
	      //mdw 4-16-2004 added correction for ktop+3 > nz-1
	      if(k==ktp){
		if(ktop<=nzdz-1){
		  utotktp.at(iceljcel)=utot;
		  uktop.at(iceljcel)=wind_vel[idcelktop].u;
		  vktop.at(iceljcel)=wind_vel[idcelktop].v;
		  wktop.at(iceljcel)=wind_vel[idcelktop].w;
		}
		else{
		  utotktp.at(iceljcel)=sqrt(ualoft*ualoft+valoft*valoft);
		  uktop.at(iceljcel)=ualoft;
		  vktop.at(iceljcel)=valoft;
		  wktop.at(iceljcel)=0.;
		}
	      }
	      deluc.at(ik)=sqrt(pow( (urefu.at(ik)-wind_vel[idcelk].u),2)+
				pow( (urefv.at(ik)-wind_vel[idcelk].v),2)+
				pow( (urefw.at(ik)-wind_vel[idcelk].w),2));
	      //mdw 4-16-2004 added correction for ktop+3 > nz-1
	      if(k!=ktp){
		delutz=sqrt(pow( (wind_vel[idcelk].u-zcorf.at(k)*uktop.at(iceljcel)),2)+
			    pow( (wind_vel[idcelk].v-zcorf.at(k)*vktop.at(iceljcel)),2)+
			    pow( (wind_vel[idcelk].w-zcorf.at(k)*wktop.at(iceljcel)),2));
		deluc.at(ik)=sqrt(pow( (urefu.at(ik)-wind_vel[idcelk].u),2)+
				  pow( (urefv.at(ik)-wind_vel[idcelk].v),2)+
				  pow( (urefw.at(ik)-wind_vel[idcelk].w),2));
		// Selects the largest gradient (vert or horiz transfer)
		// mdw 4-16-2004 added proper treatment of zfo
		if((2.*deluc.at(ik)/ m_util_ptr->qpBuildoutData.buildings[i].weff )<(utotktp.at(iceljcel)/(ht[i]+zfo[i])) 
		   && delutz>.2*zcorf.at(k)*utotktp.at(iceljcel)){ // vertical dominates
		  if(ustargz.at(idcelk)<knlc*utotktp.at(iceljcel)){ // This value dominates over prev. buildings.
		    ustargz.at(idcelk)=knlc*utotktp.at(iceljcel);
		    elzg.at(idcelk)=ht[i]+zfo[i];
		    upvpg=0.;
		    upwpg=-ctau13*zbrac*ustargz.at(idcelk)*ustargz.at(idcelk);
		    upsqg=cusq*zbrac*ustargz.at(idcelk)*ustargz.at(idcelk);
		    vpsqg=cvsq*zbrac*ustargz.at(idcelk)*ustargz.at(idcelk);
		    vpwpg=0.;
		    wpsqg=cwsq*zbrac*ustargz.at(idcelk)*ustargz.at(idcelk); // align sigmas with the overall mean wind
		    ustarg.at(idcelk)=ustargz.at(idcelk);
		    rotate2d(idcelk,cosphi,sinphi,upsqg,upvpg,vpsqg,wpsqg,upwpg,vpwpg);
		  }
		}
		else{
		  // We use the vertical gradient as dominant if it is sharper than the horizontal
		  cosl=wind_vel[idcelk].u/utot;
		  sinl=wind_vel[idcelk].v/utot;
		  duy=-deluc.at(ik)*sinl*cosu+deluc.at(ik)*sinv*cosl;
		  // we now have the delta u between the outside of the bldg and the center of the wake
		  if(deluc.at(ik)>.2*uref.at(ik)){
		    if(ustarg.at(idcelk)<knlc*deluc.at(ik)){
		      ustarg.at(idcelk)=knlc*deluc.at(ik);
		      upvpg=0.;
		      // on axis u prime v prime is zero
		      upwpg=0.;
		      // for eddy transport in uv we dont consider uw
		      upsqg=cusq*zbrac*ustarg.at(idcelk)*ustarg.at(idcelk);
		      wpsqg=cvsq*zbrac*ustarg.at(idcelk)*ustarg.at(idcelk);
		      vpwpg=0.;
		      elzg.at(idcelk)=0.5* m_util_ptr->qpBuildoutData.buildings[i].weff ;
		      vpsqg=cwsq*zbrac*ustarg.at(idcelk)*ustarg.at(idcelk);
		      rotate2d(idcelk,cosphi,sinphi,upsqg,upvpg,vpsqg,wpsqg,upwpg,vpwpg);
		    }
		  }
		}
	      }
	      for(int ist=2;ist<=istf;ist++){//do ist=2,istf
		// first direction in the transverse of the front eddy
		xcelt=xcell+ds*(ist-1.f)*cosphit;
		ycelt=ycell+ds*(ist-1.f)*sinphit;
		//mdw 7-05-2006 changed form from nint( / ) to nint( / )+1
		temp=(xcelt-dx)/dx;
		icelt=nint(temp)+1;
		temp=(ycelt-dy)/dy;
		jcelt=nint(temp)+1;
		if(fabs(xcelt-xcell)<.5*ds)icelt=icel;
		if(fabs(ycelt-ycell)<.5*ds)jcelt=jcel;
		//mdw 7-11-2006 check added to use closest axis cell
		icelt=std::min(nxdx-1,icelt);
		icelt=std::max(1,icelt);
		jcelt=std::min(nydy-1,jcelt);
		jcelt=std::max(1,jcelt);
		int idceltk=k*nxdx*nydy + jcelt*nxdx +icelt;
		int idceltktop=ktop*nxdx*nydy + jcelt*nxdx +icelt;
		int iceltjcelt=jcelt*nxdx + icelt;
		utott=sqrt(wind_vel[idceltk].u*wind_vel[idceltk].u+wind_vel[idceltk].v*wind_vel[idceltk].v
			   +wind_vel[idceltk].w*wind_vel[idceltk].w);
		utott=utott+.000001;
		delut=sqrt(pow( (urefu.at(ik)-wind_vel[idceltk].u),2)+
			   pow( (urefv.at(ik)-wind_vel[idceltk].v),2)+
			   pow( (urefw.at(ik)-wind_vel[idceltk].w),2));
		//mdw 4-16-2004 added correction for ktop+3 > nz-1
		if(k==ktp){
		  if(ktop<=nzdz-1){
		    utotktp.at(iceltjcelt)=utott;
		    uktop.at(iceltjcelt)=wind_vel[idceltktop].u;
		    vktop.at(iceltjcelt)=wind_vel[idceltktop].v;
		    wktop.at(iceltjcelt)=wind_vel[idceltktop].w;
		  }
		  else{
		    utotktp.at(iceltjcelt)=sqrt(ualoft*ualoft+valoft*valoft);
		    uktop.at(iceltjcelt)=ualoft;
		    vktop.at(iceltjcelt)=valoft;
		    wktop.at(iceltjcelt)=0.;
		  }
		}
		//mdw 4-16-2004 added correction for ktop+3 > nz-1
		if(k!=ktp){
		  delutz=sqrt(pow( (wind_vel[idceltk].u-zcorf.at(k)*uktop.at(iceltjcelt)),2)+
			      pow( (wind_vel[idceltk].v-zcorf.at(k)*vktop.at(iceltjcelt)),2)+
			      pow( (wind_vel[idceltk].w-zcorf.at(k)*wktop.at(iceltjcelt)),2));
		  // mdw 4-16-2004 added proper treatment of zfo
		  // mdw 6-10-2004 made check on centerline deluc rather than local delut
		  if((2.*deluc.at(ik)/ m_util_ptr->qpBuildoutData.buildings[i].weff )<(utotktp.at(iceltjcelt)/(ht[i]+zfo[i])) 
		     && delutz>.2*zcorf.at(k)*utotktp.at(iceltjcelt)){
		    if(ustargz.at(idceltk)<knlc*utotktp.at(iceltjcelt)){
		      ustargz.at(idceltk)=knlc*utotktp.at(iceltjcelt);
		      elzg.at(idceltk)=ht[i]+zfo[i];
		      upvpg=0.;
		      upwpg=-ctau13*zbrac*ustargz.at(idceltk)*ustargz.at(idceltk);
		      upsqg=cusq*zbrac*ustargz.at(idceltk)*ustargz.at(idceltk);
		      vpsqg=cvsq*zbrac*ustargz.at(idceltk)*ustargz.at(idceltk);
		      vpwpg=0.;
		      wpsqg=cwsq*zbrac*ustargz.at(idceltk)*ustargz.at(idceltk);
		      ustarg.at(idceltk)=ustargz.at(idceltk);
		      rotate2d(idceltk,cosphi,sinphi,upsqg,upvpg,vpsqg,wpsqg,upwpg,vpwpg);
		    }
		  }
		  else{
		    // We use the vertical gradient as dominant if it is sharper than the horizontal
		    cosl=wind_vel[idceltk].u/utott;
		    sinl=wind_vel[idceltk].v/utott;
		    duy=-deluc.at(ik)*sinl*cosu+deluc.at(ik)*sinv*cosl;
		    // mdw 6-10-2004 made check on centerline rather than local value
		    if(delut>.2*uref.at(ik)){
		      if(ustarg.at(idceltk)<knlc*deluc.at(ik)){
			ustarg.at(idceltk)=knlc*deluc.at(ik);
			// for eddy transport in uv we dont consider uw
			upvpg=-ctau13*zbrac*((ist-1.f)/(istf-1.f))*ustarg.at(idceltk)*ustarg.at(idceltk);
			upwpg=0.;
			// for eddy transport in uv we dont consider uw
			upsqg=cusq*zbrac*ustarg.at(idceltk)*ustarg.at(idceltk);
			wpsqg=cvsq*zbrac*ustarg.at(idceltk)*ustarg.at(idceltk);
			vpwpg=0.;
			vpsqg=cwsq*zbrac*ustarg.at(idceltk)*ustarg.at(idceltk);
			elzg.at(idceltk)=0.5* m_util_ptr->qpBuildoutData.buildings[i].weff ;
			rotate2d(idceltk,cosphi,sinphi,upsqg,upvpg,vpsqg,wpsqg,upwpg,vpwpg);
		      }
		    }
		  }
		  if(is==isini){
		    for(int isin=isini+1;isin<=istinf;isin++){//do isin=isini+1,istinf
		      xceln=xcelt+ds*(isin-1)*cosphi;
		      yceln=ycelt+ds*(isin-1)*sinphi;
		      temp=(xceln-dx)/dx;
		      iceln=nint(temp)+1; 
		      temp=(yceln-dy)/dy;
		      jceln=nint(temp)+1; 
		      iceln=std::min(nxdx-1,iceln);
		      iceln=std::max(1,iceln);
		      jceln=std::min(nydy-1,jceln);
		      jceln=std::max(1,jceln);
                                
		      int idcelnk=k*nxdx*nydy + jceln*nxdx +iceln;
		      int idcelnktop=ktop*nxdx*nydy + jceln*nxdx +iceln;
		      int icelnjceln=jceln*nxdx + iceln;
		      // mdw 3/22/2004pm added new if then structure to prevent replacing non-zero
		      // ustarg s with zero ones
		      if(ustarg.at(idceltk)>ustarg.at(idcelnk)){
			ustarg.at(idcelnk)=ustarg.at(idceltk);
			elzg.at(idcelnk)=elzg.at(idceltk);
			ufsqgi.at(idcelnk)=ufsqgi.at(idceltk);
			vfsqgi.at(idcelnk)=vfsqgi.at(idceltk);
			wfsqgi.at(idcelnk)=wfsqgi.at(idceltk);
			ufvfgi.at(idcelnk)=ufvfgi.at(idceltk);
			ufwfgi.at(idcelnk)=ufwfgi.at(idceltk);
			vfwfgi.at(idcelnk)=vfwfgi.at(idceltk);
		      }
		      // mdw 3/22/2004pm end  of new if then structure
		    }//enddo
		  }
		}
		// opposite direction in the transverse of the front eddy
		xcelt=xcell-ds*(ist-1.f)*cosphit;
		ycelt=ycell-ds*(ist-1.f)*sinphit;
		//mdw 7-05-2006 changed form from nint( / ) to nint( / )+1
		temp=(xcelt-dx)/dx;
		icelt=nint(temp)+1; 
		temp=(ycelt-dy)/dy;
		jcelt=nint(temp)+1; 
                    
		if(fabs(xcelt-xcell)<.5*ds)icelt=icel;
		if(fabs(ycelt-ycell)<.5*ds)jcelt=jcel;
		//mdw 7-11-2006 check added to use closest axis cell
		icelt=std::min(nxdx-1,icelt);
		icelt=std::max(1,icelt);
		jcelt=std::min(nydy-1,jcelt);
		jcelt=std::max(1,jcelt);
		iceltjcelt=jcelt*nxdx + icelt;
		idceltktop=ktop*nxdx*nydy+jcelt*nxdx +icelt;
		idceltk=k*nxdx*nydy +jcelt*nxdx +icelt;
		utott=sqrt(wind_vel[idceltk].u*wind_vel[idceltk].u+wind_vel[idceltk].v*wind_vel[idceltk].v
			   +wind_vel[idceltk].w*wind_vel[idceltk].w);
                    
		utott=utott+.000001;
		//mdw 4-16-2004 added correction for ktop+3 > nz-1
		if(k==ktp){
		  if(ktop<=nzdz-1){
		    utotktp.at(iceltjcelt)=utott;
		    uktop.at(iceltjcelt)=wind_vel[idceltktop].u;
		    vktop.at(iceltjcelt)=wind_vel[idceltktop].v;
		    wktop.at(iceltjcelt)=wind_vel[idceltktop].w;
		  }
		  else{
		    utotktp.at(iceltjcelt)=sqrt(ualoft*ualoft+valoft*valoft);
		    uktop.at(iceltjcelt)=ualoft;
		    vktop.at(iceltjcelt)=valoft;
		    wktop.at(iceltjcelt)=0.;
		  }
		}
		delut=sqrt(pow( (urefu.at(ik)-wind_vel[idceltk].u),2)
			   +pow( (urefv.at(ik)-wind_vel[idceltk].v),2)
			   +pow( (urefw.at(ik)-wind_vel[idceltk].w),2));
		//mdw 4-16-2004 added correction for ktop+3 > nz-1
		if(k!=ktp){
		  delutz=sqrt(pow( (wind_vel[idceltk].u-zcorf.at(k)*uktop.at(iceltjcelt)),2)+
			      pow( (wind_vel[idceltk].v-zcorf.at(k)*vktop.at(iceltjcelt)),2)+
			      pow( (wind_vel[idceltk].w-zcorf.at(k)*wktop.at(iceltjcelt)),2));
		  // mdw 4-16-2004 added proper treatment of zfo
		  // mdw 6-10-2004 made check on centerline rather than local value
		  if((2.*deluc.at(ik)/ m_util_ptr->qpBuildoutData.buildings[i].weff )<(utotktp.at(iceltjcelt)/(ht[i]+zfo[i])) 
		     &&delutz>.2*zcorf.at(k)*utotktp.at(iceltjcelt)){
                            
		    if(ustargz.at(idceltk)<knlc*utotktp.at(iceltjcelt)){
		      ustargz.at(idceltk)=knlc*utotktp.at(iceltjcelt);
		      elzg.at(idceltk)=ht[i]+zfo[i];
		      upvpg=0.;
		      upwpg=-ctau13*zbrac*ustargz.at(idceltk)*ustargz.at(idceltk);
		      upsqg=cusq*zbrac*ustargz.at(idceltk)*ustargz.at(idceltk);
		      vpsqg=cvsq*zbrac*ustargz.at(idceltk)*ustargz.at(idceltk);
		      vpwpg=0.;
		      wpsqg=cwsq*zbrac*ustargz.at(idceltk)*ustargz.at(idceltk);
		      ustarg.at(idceltk)=ustargz.at(idceltk);
		      rotate2d(idceltk,cosphi,sinphi,upsqg,upvpg,vpsqg,wpsqg,upwpg,vpwpg);
		    }
		  }
		  else{
		    // We use the vertical gradient as dominant if it is sharper than the horizontal
		    cosl=wind_vel[idceltk].u/utott;
		    sinl=wind_vel[idceltk].v/utott;
		    duy=-deluc.at(ik)*sinl*cosu+deluc.at(ik)*sinv*cosl;
		    // mdw 6-10-2004 made check on centerline rather than local (delut) value
		    if(delut>.2*uref.at(k)){
		      if(ustarg.at(idceltk)<knlc*deluc.at(ik)&&ustargz.at(idceltk)<knlc*deluc.at(ik)){
			ustarg.at(idceltk)=knlc*deluc.at(ik);
                                    
			// for eddy transport in uv we dont consider uw
			float tau13=0.;
			upvpg=-tau13*zbrac*((ist-1.f)/(istf-1.f))*ustarg.at(idceltk)*ustarg.at(idceltk);//check:: might be bug in QP
			upwpg=0.;
			// for eddy transport in uv we dont consider uw
			upsqg=cusq*zbrac*ustarg.at(idceltk)*ustarg.at(idceltk);
			wpsqg=cvsq*zbrac*ustarg.at(idceltk)*ustarg.at(idceltk);
			vpwpg=0.;
			elzg.at(idceltk)=0.5* m_util_ptr->qpBuildoutData.buildings[i].weff ;
			vpsqg=cwsq*zbrac*ustarg.at(idceltk)*ustarg.at(idceltk);
			rotate2d(idceltk,cosphi,sinphi,upsqg,upvpg,vpsqg,wpsqg,upwpg,vpwpg);
		      }
		    }
		  }
		  if(is==isini){
		    for(int isin=isini+1;isin<=istinf;isin++){//do isin=isini+1,istinf{
		      xceln=xcelt+ds*(isin-1)*cosphi;
		      yceln=ycelt+ds*(isin-1)*sinphi;
		      temp=(xceln-dx)/dx;
		      iceln=nint(temp)+1; 
		      temp=(yceln-dy)/dy;
		      jceln=nint(temp)+1; 
		      iceln=std::min(nxdx-1,iceln);
		      iceln=std::max(1,iceln);
		      jceln=std::min(nydy-1,jceln);
		      jceln=std::max(1,jceln);
		      int idcelnk=k*nxdx*nydy + jceln*nxdx +iceln;
		      int idcelnktop=ktop*nxdx*nydy + jceln*nxdx +iceln;
		      int icelnjceln=jceln*nxdx + iceln;
		      // mdw 3/22/2004pm added new if then structure to prevent replacing non-zero
		      // ustargs with zero ones
		      if(ustarg.at(idceltk)>ustarg.at(idcelnk)){
			ustarg.at(idcelnk)=ustarg.at(idceltk);
                                    
			elzg.at(idcelnk)=elzg.at(idceltk);
			ufsqgi.at(idcelnk)=ufsqgi.at(idceltk);
			vfsqgi.at(idcelnk)=vfsqgi.at(idceltk);
			wfsqgi.at(idcelnk)=wfsqgi.at(idceltk);
			ufvfgi.at(idcelnk)=ufvfgi.at(idceltk);
			ufwfgi.at(idcelnk)=ufwfgi.at(idceltk);
			vfwfgi.at(idcelnk)=vfwfgi.at(idceltk);
		      }
		      // mdw 3/22/2004pm end of new if then structure
		    }
		  }
		}

	      }//   lp023
            }//   lp024
	  }//   lp025
	  float xpent1, ypent1;
	  int npentx,npenty,ipent1,ipent2,jpent1,jpent2,ibuild;
        
	  std::cout << "pre building switch =====================================>" << std::endl;

	  ibuild=1;
	  switch(m_util_ptr->qpBuildoutData.buildings[i].type) {
	    case(3):

	      std::cout << "case 3 =====================================>" << std::endl;
	      xpent1=xfo[i]-wti[i]*.2f-dx;
	      temp = ((.4f*wti[i])/dx); 
	      npentx=nint(temp)+1; 
	      ypent1=yfo[i]-wti[i]*.2-dy;
	      temp=((.4f*wti[i])/dy);
	      npenty=nint(temp)+1; 
	      temp = (xpent1-dx)/dx;
	      ipent1=nint(temp)+1; 
	      ipent2=ipent1+npentx;
	      temp= ((ypent1-dy)/dy);
	      jpent1=nint(temp)+1; 
	      jpent2=jpent1+npenty;
            
	      for(int icel=ipent1;icel<=ipent2;icel++){
                for(int jcel=jpent1;jcel<=jpent2;jcel++){
		  for(int k=ktp;k>=0;k--){
		    int idcelk=k*nxdx*nydy + jcel*nxdx +icel;    
		    int iceljcel=jcel*nxdx +icel;
                        
		    utot=sqrt(wind_vel[idcelk].u*wind_vel[idcelk].u+wind_vel[idcelk].v*wind_vel[idcelk].v+wind_vel[idcelk].w*wind_vel[idcelk].w);
		    utot=utot+.000001;
                        
		    //mdw 4-16-2004 added correction for ktop+3 > nz-1
		    if(k==ktp){
		      if(ktop<=nzdz){
			int idcelktop=ktop*nxdx*nydy + jcel*nxdx +icel;
			int iceljcel=jcel*nxdx +icel; 
			utotktp.at(iceljcel)=utot;
			uktop.at(iceljcel)=wind_vel[idcelktop].u;
			vktop.at(iceljcel)=wind_vel[idcelktop].v;
			wktop.at(iceljcel)=wind_vel[idcelktop].w;
		      }
		      else{
			utotktp.at(iceljcel)=sqrt(ualoft*ualoft+valoft*valoft);
			uktop.at(iceljcel)=ualoft;
			vktop.at(iceljcel)=valoft;
			wktop.at(iceljcel)=0.;
		      }
		    }
		    if(k!=ktp && retrieveCellTypeFromArray(idcelk) != 0){
		      // MAN 9/14/2005 pentagon courtyard nonlocal mixing fix
		      delutz=sqrt((wind_vel[idcelk].u-uktop.at(iceljcel))*(wind_vel[idcelk].u-uktop.at(iceljcel))
				  +(wind_vel[idcelk].v-vktop.at(iceljcel))*(wind_vel[idcelk].v-vktop.at(iceljcel))+ 
				  (wind_vel[idcelk].w-wktop.at(iceljcel))*(wind_vel[idcelk].w-wktop.at(iceljcel)));
		      if(delutz>.2*utotktp.at(iceljcel)){ // vertical dominates
			// end MAN 9/14/2005           
			if(ustargz.at(idcelk)<knlc*utotktp.at(iceljcel)){ // This value dominates over prev. buildings.
			  ustargz.at(idcelk)=knlc*utotktp.at(iceljcel);
			  elzg.at(idcelk)=ht[i]+zfo[i];
			  upvpg=0.;
			  upwpg=-ustargz.at(idcelk)*ustargz.at(idcelk);
			  upsqg=6.25*ustargz.at(idcelk)*ustargz.at(idcelk);
			  vpsqg=(4./6.25)*upsqg;
			  vpwpg=0.;
			  wpsqg=1.69*ustargz.at(idcelk)*ustargz.at(idcelk); // align sigmas with the overall mean wind
			  upvpg=0.;
			  upwpg=-ctau13*zbrac*ustargz.at(idcelk)*ustargz.at(idcelk);
			  upsqg=cusq*zbrac*ustargz.at(idcelk)*ustargz.at(idcelk);
			  vpsqg=cvsq*zbrac*ustargz.at(idcelk)*ustargz.at(idcelk);
			  vpwpg=0.;
			  wpsqg=cwsq*zbrac*ustargz.at(idcelk)*ustargz.at(idcelk); // align sigmas with the overall mean wind
			  ustarg.at(idcelk)=ustargz.at(idcelk);
			  rotate2d(idcelk,cosphi,sinphi,upsqg,upvpg,vpsqg,wpsqg,upwpg,vpwpg);
			}
		      }
		    }
		  }
		  check1=check1+1;
                }
                check2=check2+1;
	      }
	      check3=check3+1;
	      break;
	    case(4):
	    case(5):
            
	      std::cout << "case 5 =====================================>" << std::endl;

	    float cgamma = cos( m_util_ptr->qpBuildoutData.buildings[i].gamma * pi/180.0 );
	    float sgamma = sin( m_util_ptr->qpBuildoutData.buildings[i].gamma * pi/180.0 );

	    float x0=xfo[i]+0.5*lti[i]*cgamma;
	    float y0=yfo[i]+0.5*lti[i]*sgamma;
            
	    float x1=xfo[i]+0.5*wti[ibuild]*sgamma;
	    float y1=yfo[i]-0.5*wti[ibuild]*cgamma;

	    float x2=x1+lti[i]*cgamma;
	    float y2=y1+lti[i]*sgamma;
	    float x4=xfo[i]-0.5*wti[i]*sgamma;
	    float y4=yfo[i]+0.5*wti[i]*cgamma;
	    float x3=x4+lti[i]*cgamma;
	    float y3=y4+lti[i]*sgamma;

            float temp1 = std::min(x1,x2);
            float temp2 = std::min(temp1,x3);
            float temp3 = std::min(temp2,x4);
            int icelmin = int(temp3/dx);
            temp1 = std::max(x1,x2);
            temp2 = std::max(temp1,x3);
            temp3 = std::max(temp2,x4);
            int icelmax = int(temp3/dx);

            temp1 = std::min(y1,y2);
            temp2 = std::min(temp1,y3);
            temp3 = std::min(temp2,y4);
            int jcelmin = int(temp3/dy);
            temp1 = std::max(y1,y2);
            temp2 = std::max(temp1,y3);
            temp3 = std::max(temp2,y4);
            int jcelmax = int(temp3/dy);
            
            for(int icel=icelmin;icel<=icelmax+1;icel++){
	      for(int jcel=jcelmin; jcel<=jcelmax+1;jcel++){
		float xc=(((icel)-0.5)*dx-x0)*cgamma+
		  (((jcel)-0.5)*dy-y0)*sgamma;
		float yc=-(((icel)-0.5)*dx-x0)*sgamma+
		  (((jcel)-0.5)*dy-y0)*cgamma;
		int kk=0;
		// *********************************************
		std::cout << "initial k: " << k << std::endl;
		for(int k=1;k<=ktp;k++){
		  kk=k;
		  if(zfo[i]<z.at(k))break;
		}
		std::cout << "ending k: " << k << std::endl;
		// *********************************************
		int kzfo=kk;
                    
		// *********************************************
		std::cout << ">>>> initial k: " << k << std::endl;
		for(k=ktp;k>=kzfo;k--){ 
		  dz=dz_array.at(k);
		  int incourt=0;
		  int idcelk=k*nxdx*nydy + jcel*nxdx +icel;
		  int idcelktop=ktop*nxdx*nydy + jcel*nxdx +icel;
		  int iceljcel=jcel*nxdx +icel;
		  if(retrieveCellTypeFromArray(idcelk) != 0){
		    utot=sqrt(wind_vel[idcelk].u*wind_vel[idcelk].u+wind_vel[idcelk].v*wind_vel[idcelk].v+wind_vel[idcelk].w*wind_vel[idcelk].w)+.000001f;
		    if(m_util_ptr->qpBuildoutData.buildings[i].type == 4){
		      if(xc > -0.5*lti[i] && xc < 0.5*lti[i] && 
			 yc > -0.5*wti[i] && yc < 0.5*wti[i]){
			incourt=1;
		      }
		    }
		    else{
		      float rc=sqrt((xc*xc)+(yc*yc));
		      float tc=atan2(yc,xc);
		      if(rc < 0.25*lti[i]*wti[i]/
			 sqrt((pow( (0.5f*lti[i]*sin(tc)),2.f))+(pow( (0.5f*wti[i]*cos(tc)),2.f)))){
			incourt=1;
		      }
		    }
		  }
		  else{
		    continue;
		  }
		  //mdw 4-16-2004 added correction for ktop+3 > nz-1
		  if(incourt == 1){
		    if(k==ktp){
		      if(ktop<=nz-1){
			utotktp.at(iceljcel)=utot;
			uktop.at(iceljcel)=wind_vel[idcelktop].u;
			vktop.at(iceljcel)=wind_vel[idcelktop].v;
			wktop.at(iceljcel)=wind_vel[idcelktop].w;
		      }
		      else{
			utotktp.at(iceljcel)=sqrt(ualoft*ualoft+valoft*valoft);
			uktop.at(iceljcel)=ualoft;
			vktop.at(iceljcel)=valoft;
			wktop.at(iceljcel)=0.;
		      }
		    }
		    if(k!=ktp && retrieveCellTypeFromArray(idcelk) != 0){
		      // MAN 9/14/2005 pentagon courtyard nonlocal mixing fix
		      delutz=sqrt((wind_vel[idcelk].u-uktop.at(iceljcel))*(wind_vel[idcelk].u-uktop.at(iceljcel))
				  +(wind_vel[idcelk].v-vktop.at(iceljcel))*(wind_vel[idcelk].v-vktop.at(iceljcel))+ 
				  (wind_vel[idcelk].w-wktop.at(iceljcel))*(wind_vel[idcelk].w-wktop.at(iceljcel)));
		      if(delutz>.2*utotktp.at(iceljcel)){ // vertical dominates
			// end MAN 9/14/2005              
			if(ustargz.at(idcelk)<knlc*utotktp.at(iceljcel)){ // This value dominates over prev. buildings.
			  ustargz.at(idcelk)=knlc*utotktp.at(iceljcel);
			  elzg.at(idcelk)=ht[i]+zfo[i];
			  upvpg=0.;
			  upwpg=-ustargz.at(idcelk)*ustargz.at(idcelk);
			  upsqg=6.25*ustargz.at(idcelk)*ustargz.at(idcelk);
			  vpsqg=(4./6.25)*upsqg;
			  vpwpg=0.;
			  wpsqg=1.69*ustargz.at(idcelk)*ustargz.at(idcelk); // align sigmas with the overall mean wind
			  upvpg=0.;
			  upwpg=-ctau13*zbrac*ustargz.at(idcelk)*ustargz.at(idcelk);
			  upsqg=cusq*zbrac*ustargz.at(idcelk)*ustargz.at(idcelk);
			  vpsqg=cvsq*zbrac*ustargz.at(idcelk)*ustargz.at(idcelk);
			  vpwpg=0.;
			  wpsqg=cwsq*zbrac*ustargz.at(idcelk)*ustargz.at(idcelk); // align sigmas with the overall mean wind
			  ustarg.at(idcelk)=ustargz.at(idcelk);
			  rotate2d(idcelk,cosphi,sinphi,upsqg,upvpg,vpsqg,wpsqg,upwpg,vpwpg);
			}
		      }
		    }
		  }
		}
		std::cout << ">>>> ending k: " << k << std::endl;
		// *********************************************
		check1=check1+1;
	      }
	      check2=check2+1;
            }
            check3=check3+1;
	  }
	  check=check+1;
    
         
	}//for loop for buildings for non-local mixing ends
      }
    
    // Pete

    // Following code is for local mixing-Balli -06/14/09
    // calculate distance to ground and walls if within 2 cells
    float zbrac=0.;
    float m_roof=6.;
    float eps=0.;
    float sigu=0.;
    float sigv=0.;
    float sigw=0.;
    float upwp=0.;
    float delym=0.;
    float delxm=0.;
    float u3psq=0.;
    float v3psq=0.;
    float w3psq=0.;
    float upvp=0.;
    float vpwp=0.;
    float ufwf=0.;
    float ufvf=0.;
    float vfwf=0.;
    float utotm=0.;
    float utotp=0.;
    float dutotdxp=0.;
    float dutotdxm=0.;
    float dutotdxa=0.;
    float dutotdyp=0.;
    float dutotdym=0.;
    float dutotdyc=0.;
    float dutotdya=0.;
    float x_b=0.;
    float y_b=0.;
    float dwallg=0.;
    float elzv=0.;
    float xloc=0.;
    float ufsq=0.;
    float vfsq=0.;
    float wfsq=0.;
    float dwall=0.;
    float ufsqb=0.;
    float wfsqb=0.;
    float vfsqb=0.;
    
    
    
    for(int j=0;j<nydy;j++){//altered for GPU, in QP it was ->do j=1,ny-1 -Balli(06/14/09)
      for(int i=0;i<nxdx;i++){//altered for GPU, in QP it was ->do i=1,nx-1 -Balli(06/14/09)
	int ij=j*nxdx+i;
	if(hgtveg.at(ij) > 0.){
	  for(int  kk=1;kk<=nzdz;kk++){
	    kcantop=kk;
	    if(hgtveg.at(ij) <= z.at(kk))break;
	  }
	  int idcan=kcantop*nxdx*nydy+j*nxdx+i;
	  int idcan1=(kcantop+1)*nxdx*nydy+j*nxdx+i;
	  ucantop=.5*sqrt(wind_vel[idcan].u*wind_vel[idcan].u+wind_vel[idcan].v*wind_vel[idcan].v+wind_vel[idcan].w*wind_vel[idcan].w)+
	    .5*sqrt(wind_vel[idcan1].u*wind_vel[idcan1].u+wind_vel[idcan1].v*wind_vel[idcan1].v+wind_vel[idcan1].w*wind_vel[idcan1].w);
	}
	for(int k=0;k<nzdz;k++){ //altered for GPU, in QP it was ->do k=2,nz-1 -Balli(06/14/09)
	  sigu=0.;
	  sigv=0.;
	  sigw=0.;
                
	  int id=k*nxdx*nydy +j*nxdx +i;
	  int idkm1=(k-1)*nxdx*nydy +j*nxdx +i;
	  int kp1=(k+1)*nxdx*nydy +j*nxdx +i;
                
	  int row = k / (numInRow);
	  int texidx = row * width * nydy * 4 +
	    j * width * 4 +
	    k % (numInRow) * nxdx * 4 +
	    i * 4;
                
	  sigui.at(id)=0.;
	  sigvi.at(id)=0.;
	  sigwi.at(id)=0.;


	  //Balli:New stuff- vertical varying grid
	  dz=dz_array.at(k);
	  elcanopy=0.;
	  int klim=1;
                
	  if(retrieveCellTypeFromArray(id)==8){
	    zbrac=1.;
	  }
	  else{
	    if(zi.at(k)<.99*h){
	      zbrac=pow( (1.f-zi.at(k)/h),1.5f);
	    }
	    else{
	      zbrac=pow( (1.f-.99f),1.5f);
	    }
	  }
	  dzm.at(id)=std::max(zm.at(k)-hgt.at(ij),0.f);
	  if(dzm.at(id)>zi.at(k)-std::max((hgtveg.at(ij)-elcanopy/kkar),0.f)&&(zi.at(k)-std::max((hgtveg.at(ij)-elcanopy/kkar),0.f)>0.f))
	    dzm.at(id)=zi.at(k)-std::max((hgtveg.at(ij)-elcanopy/kkar),0.f);
	  // MAN 9/21/2005 roof top mixing length fix
	  eleff.at(id)=dzm.at(id);
	  if(retrieveCellTypeFromArray(id)==8)eleff.at(id)=elcanopy*std::min(1.,(dzm.at(id)-z0)/(.3*hgtveg.at(ij)))/kkar;
	  int kdif=k-1;
	  eleff.at(id)=std::max(zm.at(k)-hgt.at(ij)*pow( (hgt.at(ij)/zm.at(k)),m_roof),0.0);
	  if(zi.at(k)-std::max((hgtveg.at(ij)-elcanopy/kkar),0.f)>0.f)
	    eleff.at(id)=std::min(eleff.at(id),zi.at(k)-std::max((hgtveg.at(ij)-elcanopy/kkar),0.f));
	  if(retrieveCellTypeFromArray(id)==8)eleff.at(id)=elcanopy*std::min(1.,(zi.at(k)-z0)/(.3*hgtveg.at(ij)))/kkar;
	  klim=nzdz;
	  // calculation of ustar in the vertical
	  if(retrieveCellTypeFromArray(idkm1) == 0 && retrieveCellTypeFromArray(id) != 0){
	    utot=sqrt(wind_vel[id].u*wind_vel[id].u+wind_vel[id].v*wind_vel[id].v*+wind_vel[id].w*wind_vel[id].w);
	    if(retrieveCellTypeFromArray(id)!=8){
	      if(rcl>0){
		phim=1.+4.7*rcl*0.5*dz;
		psim=-4.7*rcl*0.5*dz;
	      }
	      else{
		phim=pow( (1.f-15.f*rcl*0.5f*dz),(-.25f));
		psim=2.*log((1.+1./phim)/2.)+log((1.+1./pow(phim,2.f))/2.)-2.*atan(1./phim)+pi/2.;
	      }
	    }
	    else{
	      phim=1.;
	      psim=0.;
	      elz.at(id)=elcanopy*std::min(1.,(.5*dz-z0)*kkar/(.3*hgtveg.at(ij)));
	      ustar=elz.at(id)*utot/(.5*dz-z0);
	      dutotdzi.at(id)=ustar/elz.at(id);
	      ustarz.at(id)=ustar;
	    }
	  }
	  else{
	    if(retrieveCellTypeFromArray(id) != 0){
	      // potential problem if kp1 is greater than index bounds...
	      // std::cout << "wind_vel[ " << kp1 << " ] = (" << wind_vel[kp1].u << ", " << wind_vel[kp1].v << ", " << wind_vel[kp1].w << ")" << std::endl;
	      utotu=sqrt(wind_vel[kp1].u*wind_vel[kp1].u+wind_vel[kp1].v*wind_vel[kp1].v+wind_vel[kp1].w*wind_vel[kp1].w);
	      utot=sqrt(wind_vel[id].u*wind_vel[id].u+wind_vel[id].v*wind_vel[id].v*+wind_vel[id].w*wind_vel[id].w);
	      if(fabs(dutotdzi.at(id))>1.e-06 && retrieveCellTypeFromArray(id)!=8 && dzm.at(id)>2.*dz){
		elz.at(id)=kkar*utot/fabs(dutotdzi.at(id));
		// MAN 9/21/2005 roof top mixing length fix
		if((kkar*eleff.at(id))<elz.at(id))
		  elz.at(id)=kkar*eleff.at(id);
		else
		  elz.at(id)=kkar*eleff.at(id);
		if(retrieveCellTypeFromArray(id)==8){
		  elz.at(id)=elcanopy*std::min(1.,(zi.at(k)-z0)/(.3*hgtveg.at(ij)));
		  eleff.at(id)=elz.at(id)/kkar;
		}
	      }
	      if(k < nz-1){
		if((retrieveCellTypeFromArray(kp1)!=8)&&(retrieveCellTypeFromArray(id)==8)){
		  if(fabs(dutotdzi.at(id))>1.e-06){
		    elz.at(id)=kkar*utot/fabs(dutotdzi.at(id));
		    if((kkar*eleff.at(id))<elz.at(id)) elz.at(id)=kkar*eleff.at(id);
		  }
		}
	      }
	      // We have just put in the vortex mixing length for the last cell in the canopy
	      if(retrieveCellTypeFromArray(idkm1)!=8){
		if(rcl>0){
		  phim=1.+4.7*rcl*eleff.at(id);
		  psim=-4.7*rcl*eleff.at(id);
		}
		else{
		  phim=pow( (1.f-15.f*rcl*eleff.at(id)),(-.25f));
		  psim=2.*log((1.+1./phim)/2.)+log((1.+1./pow(phim,2.f))/2.)-2.*atan(1./phim)+pi/2.;
		}
		ustar=kkar*eleff.at(id)*fabs(dutotdzi.at(id))/phim;
		ustarz.at(id)=ustar;
	      }
	      else{
		utotl=sqrt(wind_vel[idkm1].u*wind_vel[idkm1].u+wind_vel[idkm1].v*wind_vel[idkm1].v+wind_vel[idkm1].w*wind_vel[idkm1].w);
		dutotdzi.at(id)=(utotu-utotl)/(2.*dz);
		//corrected wrong gradient at the vegetative canopy top 12/22/2008
		if(retrieveCellTypeFromArray(id)!=8){
		  elz.at(id)=kkar*.5*dz;
		}
		else{
		  elz.at(id)=elcanopy*std::min(1.,(zi.at(k)-z0)/(.3*hgtveg.at(ij)));
		}
		ustar=elz.at(id)*fabs(dutotdzi.at(id));
		ustarz.at(id)=ustar;
	      }
	      if(retrieveCellTypeFromArray(idkm1)!=8){
		if(rcl>0){
		  phim=1.+4.7*rcl*eleff.at(id);
		  psim=-4.7*rcl*eleff.at(id);
		}
		else{
		  phim=pow( (1.f-15.f*rcl*eleff.at(id)),(-.25f));
		  psim=2.*log((1.+1./phim)/2.)+log((1.+1./pow(phim,2.f))/2.)-2.*atan(1./phim)+pi/2.;
		}
		ustar=kkar*eleff.at(id)*fabs(dutotdzi.at(id))/phim;
		ustarz.at(id)=ustar;
	      }
	    }
	  }
	  // for neutral conditions sigw is only dependent on ustar
                
	  // for vertical downward distance (dzm)
	  klim=nzdz-1;
                
	  float dutotdn=0.f;
	  float dutotds=0.f;
                
	  sig[id].u = 0.0f;
	  sig[id].v = 0.0f;   
	  sig[id].w = 0.0f;   
                
	  tau[id].t11   = 0.0f;        
	  tau[id].t22 = 0.0f;        
	  tau[id].t33 = 0.0f;
	  tau[id].t13 = 0.0f;

	  // ????
	  // tau[id].t11   = 0.0f;        
	  // tau[id+1].t22 = 0.0f;        
	  // tau[id+2].t33 = 0.0f;
	  // tau[id+3].t13 = 0.0f;

	  //Make tau's a texture so that they can be visualized as horizontal layers in the domain
	  dataTau[texidx] = 0.0f;   
	  dataTau[texidx+1] =0.0f;  
	  dataTau[texidx+2] =0.0f;  
	  dataTau[texidx+3] = 0.0f; 
                
	  dataTwo[texidx]   = 0.0f;
	  dataTwo[texidx+1] =  0.0f;
	  dataTwo[texidx+2] =  0.0f;
	  dataTwo[texidx+3] =  0.0f;
                
	  dataWind[texidx]   = wind_vel[id].u;
	  dataWind[texidx+1] = wind_vel[id].v;
	  dataWind[texidx+2] = wind_vel[id].w;
	  dataWind[texidx+3] = 0.;
	  if(k>0){
	    dzp.at(id)=10.f*dz+z.at(klim)-z.at(k-1)+.5f*dz;
	  }
	  else{
	    dzp.at(id)=10.f*dz+z.at(klim)+.5f*dz;
	  }
	  for(int kk=k;kk<=klim;kk++){//do kk=k,klim
	    int idkk=kk*nxdx*nydy +j*nxdx +i;
	    int celltypeidkk=0;
	    if(idkk>=0){
	      celltypeidkk=retrieveCellTypeFromArray(idkk);
	    }
	    if(celltypeidkk == 0){
	      dzp.at(id)=.5*dz+(kk-k-1)*dz;
	      break;
	    }
	  }
	  // for distance to the left (dxm)
	  int ilim=0;
	  dxm.at(id)=.5*dx+i*dx+(nxdx+1)*dx;//altered for GPU [added 1 to nxdx as in QP nx is nx+1 actually] -Balli(06/14/09)
	  for(int ii=i;ii>ilim;ii--){//do ii=i,ilim,-1
	    // calculation of the distance to the wall in the negative x direction
	    int idii=k*nxdx*nydy +j*nxdx +ii;
	    int celltypeidii=1;//as in x or y direction we assume fluid in all directions out of domain
	    if(idii>=0){
	      celltypeidii=retrieveCellTypeFromArray(idii);
	    }                        
	    if(celltypeidii == 0){
	      dxm.at(id)=.5*dx+(i-ii-1)*dx;
	      break;
	    }
	  }
	  // for distance to the right (dxp)
	  ilim=nxdx-1;
	  dxp.at(id)=.5*dx+(ilim-i)*dx+(nxdx+1)*dx;
	  for(int ii=i;ii<=ilim;ii++){// ii=i,ilim
	    // calculation of the distance to the wall in the positive x direction
	    int idii=k*nxdx*nydy +j*nxdx +ii;
	    int celltypeidii=1;//as in x or y direction we assume fluid in all directions out of domain
	    if(idii>=0){
	      celltypeidii=retrieveCellTypeFromArray(idii);
	    }
                    
	    if(celltypeidii == 0){
	      dxp.at(id)=.5*dx+(ii-i-1)*dx;
	      break;
	    }
	  }
	  // for distance  from the back (dym)
	  int jlim=0;
	  dym.at(id)=.5*dy+j*dy+(nydy+1)*dy; //added 1 to nydy ,same reason as in x-directiom, see above-Balli(06/14/09)
	  for(int jj=j;jj>jlim;jj--){//do jj=j,jlim,-1
	    // calculation of the distance to the wall in the negative y direction
	    int idjj=k*nxdx*nydy +jj*nxdx +i;
	    int celltypeidjj=1;//as in x or y direction we assume fluid in all directions out of domain
	    if(idjj>=0){
	      celltypeidjj=retrieveCellTypeFromArray(idjj);
	    }
	    if(celltypeidjj == 0){
	      dym.at(id)=.5*dy+(j-jj-1)*dy;
	      break;
	    }
	  }
	  // for distance to the front  (dyp)
	  jlim=nydy-1;
	  dyp.at(id)=.5*dy+(jlim-j)*dy+(nydy+1)*dy; //added 1 to nydy ,same reason as in x-directiom, see above-Balli(06/14/09)
	  for(int jj=j;jj<=jlim;jj++){//do jj=j,jlim
	    // calculation of the distance to the wall in the positive x direction
	    int idjj=k*nxdx*nydy +jj*nxdx +i;
	    int celltypeidjj=1;//as in x or y direction we assume fluid in all directions out of domain
	    if(idjj>=0){
	      celltypeidjj=retrieveCellTypeFromArray(idjj);
	    }
	    if(celltypeidjj == 0){
	      dyp.at(id)=.5*dy+(jj-j-1)*dy;
	      break;
	    }
	  }
	  // we need to calculate the largest change in utot
	  if(retrieveCellTypeFromArray(id) == 0){
	    eps=0.;
	    sigu=0.;
	    sigv=0.;
	    sigw=0.;
	    upwp=0.;
	    elz.at(id)=0.;
	    eleff.at(id)=0.;
	    sig[id].u = 0.0f;
	    sig[id].v = 0.0f;   
	    sig[id].w = 0.0f;
	  }
	  if(retrieveCellTypeFromArray(id) != 0){//for all fuid cells
	    // first we set up parameters for cells near boundary
	    if(j<1||j>=ny-1||i<1||i>=nx-1){//boundary cells
	      // calculation of near-boundary values of u*y, ly, dely, and the
	      // gradients of speed in the x and y directions
	      delym=(nydy+1)*dy;
	      utot=sqrt(wind_vel[id].u*wind_vel[id].u+wind_vel[id].v*wind_vel[id].v+wind_vel[id].w*wind_vel[id].w);
	      sigvi.at(id)=0.;
	      delxm=dx*(nxdx-1);
	      dutotdxi.at(id)=0.;
	      dutotdyi.at(id)=0.;
	      elz.at(id)=kkar*eleff.at(id);
	      dutotdni.at(id)=dutotdzi.at(id);

	      detang(0,id,dutotds,dutotdn,i,j,k);
	      if(rcl>0){
		phim=1.+4.7*rcl*eleff.at(id);
		psim=-4.7*rcl*eleff.at(id);
	      }
	      else{
		phim=pow( (1.-15.*rcl*eleff.at(id)),(-.25) );
		psim=2.*log((1.+1./phim)/2.)+log((1.+1./pow(phim,2.f))/2.)-2.*atan(1./phim)+pi/2.;
	      }
	      ustarz.at(id)=elz.at(id)*dutotdni.at(id)/phim; // calculate local ustar
	      ustarz.at(id)=std::max(ustarz.at(id),3.e-02f);
	      u3psq=cusq*zbrac*ustarz.at(id)*ustarz.at(id);   //// (u''')^2
	      v3psq=cvsq*zbrac*ustarz.at(id)*ustarz.at(id);   //...
	      w3psq=cwsq*zbrac*ustarz.at(id)*ustarz.at(id); //...
	      upwp=-ctau13*zbrac*ustarz.at(id)*ustarz.at(id); // -tau13
	      upvp=0.;
	      vpwp=0.;
	      if(rcl<0 && zi.at(k)<.99*h){
		u3psq=u3psq+.6*(ustarz.at(id)*ustarz.at(id))*pow( (-h*rcl),(2.f/3.f));
		v3psq=v3psq+.6*(ustarz.at(id)*ustarz.at(id))*pow( (-h*rcl),(2.f/3.f));
		w3psq=w3psq+3.3*(ustarz.at(id)*ustarz.at(id))*pow( (-zi.at(k)*rcl),(2.3f))*pow( (1.f-.8f*zi.at(k)/h),2.f);
		upwp=upwp*pow( (1.f-zi.at(k)/h),(.5f*rcl*h/(1.0f-rcl*h)) );
	      }
	      else{
		if(rcl<0){
		  u3psq=u3psq+.6*(ustarz.at(id)*ustarz.at(id))*pow( (-h*rcl),(2.f/3.f));
		  v3psq=v3psq+.6*(ustarz.at(id)*ustarz.at(id))*pow( (-h*rcl),(2.f/3.f));
		  w3psq=w3psq+3.3*(ustarz.at(id)*ustarz.at(id))*pow( (-.99f*h*rcl),(2.3f))*pow( (1.f-.8f*.99f),2.f);
		  upwp=upwp*pow( (1.f-.99f),(.5f*rcl*h/(1.0f-rcl*h)) );  
		}
	      }
	      rotu3psq(id,u3psq,utot,upvp,upwp,vpwp,v3psq,w3psq,ufsqb,wfsqb,vfsqb,ufvf,ufwf,vfwf); // rotate values back into the orig. grid
	      ufwfi.at(id)=ufwf;
	      ufvfi.at(id)=ufvf;
	      vfwfi.at(id)=vfwf;
	      ustarij.at(id)=ustarz.at(id);
	      sigui.at(id)=sqrt(u3psq);
	      sigvi.at(id)=sqrt(v3psq);
	      sigwi.at(id)=sqrt(w3psq);
	      upwpi.at(id)=upwp;
	      // along the boundaries we make y effects negligible
	    }
	    else{
	      utot=sqrt(wind_vel[id].u*wind_vel[id].u+wind_vel[id].v*wind_vel[id].v+wind_vel[id].w*wind_vel[id].w);
	      // away from boundaries u*y, ly, dely, and gradients
	      int idim1=k*nxdx*nydy +j*nxdx +(i-1);
	      int idip1=k*nxdx*nydy +j*nxdx +(i+1);
	      if(retrieveCellTypeFromArray(idim1) != 0 && retrieveCellTypeFromArray(id) != 0  && retrieveCellTypeFromArray(idip1) != 0){
		//mdw 3-08-2004 start changes for highest gradient
		utot=sqrt(wind_vel[id].u*wind_vel[id].u+wind_vel[id].v*wind_vel[id].v+wind_vel[id].w*wind_vel[id].w);
		utotm=sqrt(wind_vel[idim1].u*wind_vel[idim1].u+wind_vel[idim1].v*wind_vel[idim1].v+wind_vel[idim1].w*wind_vel[idim1].w);
		utotp=sqrt(wind_vel[idip1].u*wind_vel[idip1].u+wind_vel[idip1].v*wind_vel[idip1].v+wind_vel[idip1].w*wind_vel[idip1].w);

		dutotdxp=(utotp-utot)/dx;
		dutotdxm=(utot-utotm)/dx;
		dutotdxa=std::max(fabs(dutotdxp),fabs(dutotdxm));
		if(dutotdxa==fabs(dutotdxm)){
		  dutotdxi.at(id)=dutotdxm;
		}
		else{
		  dutotdxi.at(id)=dutotdxp;
		}
		// mdw 3-08-2004end changes
	      }
	      else{
		if(retrieveCellTypeFromArray(id) == 0){ ////BALLI
		  dutotdxi.at(id)=0.;
		}
		else{
		  if(retrieveCellTypeFromArray(idim1) == 0){ ////BALLI
		    dutotdxi.at(id)=2.*sqrt(wind_vel[id].u*wind_vel[id].u+wind_vel[id].v*wind_vel[id].v+wind_vel[id].w*wind_vel[id].w)/dx;
		    dutotdxi.at(id)=sqrt(wind_vel[id].u*wind_vel[id].u+wind_vel[id].v*wind_vel[id].v+wind_vel[id].w*wind_vel[id].w)
		      /(log((.5*dx)/z0)*(.5*dx));
		  }
		  else{
		    dutotdxi.at(id)=-sqrt(wind_vel[id].u*wind_vel[id].u+wind_vel[id].v*wind_vel[id].v+wind_vel[id].w*wind_vel[id].w)
		      /(log((.5*dx)/z0)*(.5*dx));
		  }
		}
	      }
                        
	      int idjm1=k*nxdx*nydy +(j-1)*nxdx +i;
	      int idjp1=k*nxdx*nydy +(j+1)*nxdx +i;
	      if(retrieveCellTypeFromArray(id) != 0 && retrieveCellTypeFromArray(idjm1) != 0 && retrieveCellTypeFromArray(idjp1) != 0){
		//mdw 3-08-2008 start gradient changes
		utot=sqrt(wind_vel[id].u*wind_vel[id].u+wind_vel[id].v*wind_vel[id].v+wind_vel[id].w*wind_vel[id].w);
		utotm=sqrt(wind_vel[idjm1].u*wind_vel[idjm1].u+wind_vel[idjm1].v*wind_vel[idjm1].v+wind_vel[idjm1].w*wind_vel[idjm1].w);
		utotp=sqrt(wind_vel[idjp1].u*wind_vel[idjp1].u+wind_vel[idjp1].v*wind_vel[idjp1].v+wind_vel[idjp1].w*wind_vel[idjp1].w);
		dutotdyc=0.5*(utotp-utotm)/dy;
		dutotdyp=(utotp-utot)/dy;
		dutotdym=(utot-utotm)/dy;
		dutotdya=std::max(fabs(dutotdyp),fabs(dutotdym));
		if(dutotdya==fabs(dutotdym)){
		  dutotdyi.at(id)=dutotdym;
		}
		else{
		  dutotdyi.at(id)=dutotdyp;
		}
		// mdw 3-08-2004end changes
	      }
	      else{
		if(retrieveCellTypeFromArray(id) == 0){
		  dutotdyi.at(id)=0.;
		}
		else{
		  if(retrieveCellTypeFromArray(idjm1) == 0){
		    dutotdyi.at(id)=sqrt(wind_vel[id].u*wind_vel[id].u+wind_vel[id].v*wind_vel[id].v+wind_vel[id].w*wind_vel[id].w)
		      /(log((.5*dy)/z0)*(.5*dy));
		  }
		  else{
		    dutotdyi.at(id)=-sqrt(wind_vel[id].u*wind_vel[id].u+wind_vel[id].v*wind_vel[id].v+wind_vel[id].w*wind_vel[id].w)
		      /(log((.5*dy)/z0)*(.5*dy));
		  }
		}
	      }
	    }
	    detang(0,id,dutotds,dutotdn,i,j,k); // Calculates the parameters fot the triple rotatation of coord sys.
	    dwall=std::min(std::min(eleff.at(id),std::min(dxm.at(id),dxp.at(id))),std::min(dym.at(id),dyp.at(id)));
	    dwall=std::min(dwall,dzm.at(id));
	    elz.at(id)=kkar*dwall; // length scale based on distance to wall

	    if(retrieveCellTypeFromArray(id) !=8)elz.at(id)=kkar*dwall; // length scale based on distance to wall
	    if(fabs(dutotdni.at(id))>1.e-6){
	      x_b=std::min(dxm.at(id),dxp.at(id));
	      if(x_b>std::max(del_b,dx)) x_b=0;
	      y_b=std::min(dym.at(id),dyp.at(id));
	      if(y_b>std::max(del_b,dy)) y_b=0;
	      dwallg=fabs(dutotdyi.at(id))*y_b+fabs(dutotdxi.at(id))*x_b;
	      dwallg=dwallg+fabs(dutotdzi.at(id))*eleff.at(id);
	      dwallg=dwallg/dutotdni.at(id);
	      elzv=kkar*utot/dutotdni.at(id); // length scale based on distance to null wind
	      if(dwallg*kkar<elzv && (x_b+y_b)>0.) {
		// mdw 6-29-2006 changed test so that must be near vertical wall
		elz.at(id)=kkar*dwallg; // pick the smallest length scale
	      }
	      else{
		// mdw 6-30-2006 changed test so that vortex test does not override normal stuff
		if(elzv<=elz.at(id)){
		  elz.at(id)=elzv;
		}
	      }
	    }
	    if(retrieveCellTypeFromArray(id)!=8){
	      if(rcl>0){
		phim=1.+4.7*rcl*eleff.at(id);
		psim=-4.7*rcl*eleff.at(id);
	      }
	      else{
		phim=pow( (1.f-15.f*rcl*eleff.at(id)),(-.25f) );
		psim=2.*log((1.+1./phim)/2.)+log((1.+1./pow(phim,2.f))/2.f)-2.*atan(1./phim)+pi/2.;
	      }
	    }
	    else{
	      phim=1.;
	      psim=0.;
	      elz.at(id)=elcanopy*std::min(1.f,(dzm.at(id)-z0)*kkar/(.3f*hgtveg.at(ij)));
	    }
	    ustarz.at(id)=elz.at(id)*dutotdni.at(id)/phim; // calculate local ustar
	    ustarz.at(id)=std::max(ustarz.at(id),3.e-02f);
	    //mdw 6-23-2004 adjust for vertical structure
	    u3psq=cusq*zbrac*ustarz.at(id)*ustarz.at(id);   // (u''')^2
	    v3psq=cvsq*zbrac*ustarz.at(id)*ustarz.at(id);   //...
	    w3psq=cwsq*zbrac*ustarz.at(id)*ustarz.at(id); //...
	    upwp=-ctau13*zbrac*ustarz.at(id)*ustarz.at(id); // -tau13
	    upvp=0.;
	    vpwp=0.;

	    // Turn off non-local mixing by setting xloc to 0
	    xloc=1.;
	    // 
	    // Changed per suggestion from Bugs's Evernote slides: 4/25/12
	    if(ustarz.at(id)>xloc*ustarg.at(id)){
	      if(rcl<0. && zi.at(k)<.99*h){
		u3psq=u3psq+.6*(ustarz.at(id)*ustarz.at(id))*pow( (-h*rcl),(2.f/3.f));
		v3psq=v3psq+.6*(ustarz.at(id)*ustarz.at(id))*pow( (-h*rcl),(2.f/3.f));
		w3psq=w3psq+3.3*(ustarz.at(id)*ustarz.at(id))*pow( (-zi.at(k)*rcl),(2.3f))*pow( (1.f-.8f*zi.at(k)/h),2.f);
		upwp=upwp*pow( (1.f-zi.at(k)/h),(.5f*rcl*h/(1.f-rcl*h)) );
	      }
	      else{
		if(rcl<0.){
		  u3psq=u3psq+.6*(ustarz.at(id)*ustarz.at(id))*pow( (-h*rcl),(2.f/3.f));
		  v3psq=v3psq+.6*(ustarz.at(id)*ustarz.at(id))*pow( (-h*rcl),(2.f/3.f));
		  w3psq=w3psq+3.3*(ustarz.at(id)*ustarz.at(id))*pow( (-.99f*h*rcl),(2.3f))*pow( (1.f-.8f*.99f),2.f);
		  upwp=upwp*pow( (1.f-.99f),(.5f*rcl*h/(1.f-rcl*h)) );
		}
	      }
	      rotu3psq(id,u3psq,utot,upvp,upwp,vpwp,v3psq,w3psq,ufsqb,wfsqb,vfsqb,ufvf,ufwf,vfwf); // rotate values back into the orig. grid
	      ufwfi.at(id)=ufwf;
	      ufvfi.at(id)=ufvf;
	      vfwfi.at(id)=vfwf;
	      ustarij.at(id)=ustarz.at(id);
	      sigui.at(id)=sqrt(u3psq);
	      sigvi.at(id)=sqrt(v3psq);
	      sigwi.at(id)=sqrt(w3psq);
	      upwpi.at(id)=upwp;
	    }
	    else{ // non-local dominates (possible place for into of TPT)
	      ufsq=ufsqgi.at(id);
	      vfsq=vfsqgi.at(id);
	      wfsq=wfsqgi.at(id);
	      ufvf=ufvfgi.at(id);
	      ufwf=ufwfgi.at(id);
	      vfwf=vfwfgi.at(id);
	      sigui.at(id)=sqrt(ufsq);
	      sigvi.at(id)=sqrt(vfsq);
	      sigwi.at(id)=sqrt(wfsq);
	      ufwfi.at(id)=ufwf;
	      ufvf=0.;
	      ufvfi.at(id)=ufvf;
	      vfwfi.at(id)=vfwf;
	      //mdw 7-25-2005 corrections for axis rotation with non-local mixing
	      ustarij.at(id)=ustarg.at(id);
	      rotufsq(id,u3psq,upwp,v3psq,w3psq,ufsq,ufvf,ufwf,vfsq,vfwf,wfsq);
	      sigui.at(id)=sqrt(u3psq);
	      sigvi.at(id)=sqrt(v3psq);
	      sigwi.at(id)=sqrt(w3psq);
	      upwpi.at(id)=upwp;
	    }
	    sigu=sigui.at(id);
	    sigv=sigvi.at(id);
	    sigw=sigwi.at(id);
	    if(zi.at(k)<=.99*h){
	      eps=pow(ustarij.at(id),3.f)*(1.f-.75f*zi.at(k)*rcl)*pow((1.f-.85f*zi.at(k)/h),(1.5f))/eleff.at(id); // calculate epsilon for grid cell centers
	    }
	    else{
	      eps=pow(ustarij.at(id),3.f)*(1.f-.75f*.99f*h*rcl)*pow((1.f-.85f*.99f),(1.5f))/eleff.at(id); 
	    }
                    
	  }
	  epsi.at(id)=eps;
                
	  // erp June 18, 2009  shader needs sigu sigv
	  // convert them to taus and load them onto the shaders
	  //tau13= ufwf; tau12 = ufvf; tau23 = vfwf need to be written to shaders
	  turbfield<<std::setw(7) << std::setprecision(3)<< std::scientific<<xi.at(i)<<"   "
		   <<yi.at(j)<<"   "<<zi.at(k)<<"   "<< sigu<<"  "<<sigv<<"  "<<sigw<<"  "<<elz.at(id)<<"  "<<
	    eleff.at(id)<<"  "<<eps<<std::endl;
                
	  sig[id].u = sigu;   //sigU
	  sig[id].v = sigv;   //sigV
	  sig[id].w = sigw;   //sigW
                
	  float tau11=sigu*sigu;
	  float tau22=sigv*sigv;
	  float tau33=sigw*sigw;
	  float tau13=ustarij.at(id)*ustarij.at(id);
                
	  tau[id].t11   = tau11;             //Tau11
	  tau[id].t22 = tau22;             //Tau22
	  tau[id].t33 = tau33;             //Tau33
	  tau[id].t13 = tau13;             //Tau13

	  // Is this correct - Pete (01/30/12) why are we
	  // indexing ahead of the structure rather than setting
	  // the various tau components... fixed above in what I
	  // think is correct as of this writing.
	  // tau[id].t11   = tau11;             //Tau11
	  // tau[id+1].t22 = tau22;             //Tau22
	  // tau[id+2].t33 = tau33;             //Tau33
	  // tau[id+3].t13 = tau13;             //Tau13

	  //Make tau's a texture so that they can be visualized as horizontal layers in the domain

                
	  dataTwo[texidx]   =  sigu;
	  dataTwo[texidx+1] =  sigv;
	  dataTwo[texidx+2] =  sigw;
	  dataTwo[texidx+3] =  ustarij.at(id);
                
	  data3[texidx]   = dutotdxi.at(id);
	  data3[texidx+1] = dutotdyi.at(id);
	  data3[texidx+2] = dutotdzi.at(id);
	  data3[texidx+3] = 0.;

	  data4[texidx]   = dzm.at(id);
	  data4[texidx+1] = dzp.at(id);
	  data4[texidx+2] = dym.at(id);
	  data4[texidx+3] = dyp.at(id);
                
	  data[texidx]   = dxm.at(id);
	  data[texidx+1] = dxp.at(id);
	  data[texidx+2] = .00987;
	  data[texidx+3] = .00986;
                
	  ustar=std::max(ustarij.at(id),0.003f);
	  float ustar3=ustar*ustar*ustar;
	  dataWind[texidx+3] = 0.;
	}//   lp027
      }//      lp028
    }//         lp029
    

    // This shouldn't be here!!!! We need to have a different step (or
    // this one) that puts the data into CUDA memory on the device!
    
    createTexture(lambda, GL_RGBA32F_ARB, width,height, dataTwo);
    createTexture(windField, GL_RGBA32F_ARB, width, height, dataWind);
    createTexture(tauTex, GL_RGBA32F_ARB, width, height, dataTau);
    createTexture(tau_dz, GL_RGBA32F_ARB, width,height, data);
    
    delete[] dataTwo;
    delete[] dataWind;
    delete[] dataTau;
    delete[] data;

    // Note from Pete: I wonder if this is a logical place to break
    // this function into two parts... especially if we don't need all
    // the data above????

    std::vector<float>dsigwdni,dsigvdni,dsigudni,dupwpdni;
    dsigwdni.resize(nzdz*nydy*nxdx);
    dsigvdni.resize(nzdz*nydy*nxdx);
    dsigudni.resize(nzdz*nydy*nxdx);
    dupwpdni.resize(nzdz*nydy*nxdx);
    ani.resize(nzdz*nydy*nxdx);
    bni.resize(nzdz*nydy*nxdx);
    cni.resize(nzdz*nydy*nxdx);

    std::ofstream dsigwdnOut;
    dsigwdnOut.open("dsigwdn.dat");
    float dsigwdx=0.;
    float dsigwdy=0.;
    float dsigudx=0.;
    float dsigudy=0.;
    float dsigvdx=0.;
    float dsigvdy=0.;
    float dupwpdx=0.;
    float dupwpdy=0.;
    float dsigwdz=0.;
    float dsigvdz=0.;
    float dsigudz=0.;
    float dupwpdz=0.;
    float dsigwdn=0.;
    float dsigvdn=0.;
    float dsigudn=0.;
    float dupwpdn=0.;

    for(int k=0; k<nzdz;k++){//do k=2,nz-1
      for(int j=0;j<nydy;j++){//do j=1,ny-1
	for(int i=0;i<nxdx;i++){//do i=1,nx-1
	  int id=k*nxdx*nydy + j*nxdx + i;
                
	  int row = k / (numInRow);
	  int texidx = row * width * nydy * 4 +
	    j * width * 4 +
	    k % (numInRow) * nxdx * 4 +
	    i * 4;
                
                
	  if(retrieveCellTypeFromArray(id) != 0){
	    int idim1=k*nxdx*nydy +j*nxdx +(i-1);
	    int idip1=k*nxdx*nydy +j*nxdx +(i+1);
                    
	    int idjm1=k*nxdx*nydy +(j-1)*nxdx +i;
	    int idjp1=k*nxdx*nydy +(j+1)*nxdx +i;
                    
	    int idkm1=(k-1)*nxdx*nydy +j*nxdx +i;
	    int idkp1=(k+1)*nxdx*nydy +j*nxdx +i;
                    
	    if(j<1||j>=nydy-1||i<1||i>=nxdx-1){
	      dsigwdx=0.;
	      dsigwdy=0.;
	      dsigudx=0.;
	      dsigudy=0.;
	      dsigvdx=0.;
	      dsigvdy=0.;
	      dupwpdx=0.;
	      dupwpdy=0.;
	    }
	    //
	    // calculate the gradient of sigma w normal to the flow using a CDD
	    //
	    utot=sqrt(wind_vel[id].u*wind_vel[id].u+wind_vel[id].v*wind_vel[id].v+wind_vel[id].w*wind_vel[id].w);
	    if(dxm.at(id)>=dx && dxp.at(id)>=dx && i!=0 && i!=nxdx-1){
	      if(idim1<0){
		dsigwdx=.5*(sigwi.at(idip1)-sigwi.at(id))/dx;
		dsigvdx=.5*(sigvi.at(idip1)-sigvi.at(id))/dx;
		dsigudx=.5*(sigui.at(idip1)-sigui.at(id))/dx;
		dupwpdx=.5*(upwpi.at(idip1)-upwpi.at(id))/dx;
	      }
	      else{
		dsigwdx=.5*(sigwi.at(idip1)-sigwi.at(idim1))/dx;
		dsigvdx=.5*(sigvi.at(idip1)-sigvi.at(idim1))/dx;
		dsigudx=.5*(sigui.at(idip1)-sigui.at(idim1))/dx;
		dupwpdx=.5*(upwpi.at(idip1)-upwpi.at(idim1))/dx;							  
	      }
	    }
	    else{
	      if(i==1||i==nxdx-1){
		dsigwdx=0.;
		dsigvdx=0.;
		dsigudx=0.;
		dupwpdx=0.;
	      }
	      else{
		if(dxm.at(id)<dx && dxp.at(id)>dx){
		  //mdw 11-21-2005 modified if statements to address particle in 3-walled cells
		  dsigwdni.at(id)=(sigwi.at(idip1)-sigwi.at(id))/dx;
		  dsigvdni.at(id)=(sigvi.at(idip1)-sigvi.at(id))/dx;
		  dsigudni.at(id)=(sigui.at(idip1)-sigui.at(id))/dx;
		  dupwpdni.at(id)=(upwpi.at(idip1)-upwpi.at(id))/dx;
		  dsigwdni.at(id)=0.;
		  dsigvdni.at(id)=0.;
		  dsigudni.at(id)=0.;
		  dupwpdni.at(id)=0.;
		  sigwi.at(id)=std::max(sigwi.at(idip1),sigwi.at(id));
		  sigvi.at(id)=std::max(sigvi.at(idip1),sigvi.at(id));
		  sigui.at(id)=std::max(sigui.at(idip1),sigui.at(id));
		  ustarij.at(id)=std::max(ustarij.at(idip1),ustarij.at(id));
		  if(fabs(upwpi.at(id))<fabs(upwpi.at(idip1))){
		    upwpi.at(id)=upwpi.at(idip1);
		  }
		  else{
		    upwpi.at(idip1)=upwpi.at(id);
		  }
		}
		if(dxp.at(id)<dx && dxm.at(id)>dx){
		  //mdw 11-21-2005 modified if statements to address particle in 3-walled cells
		  if(idim1<0){
		    dsigwdni.at(id)=0.;//(sigwi.at(id)-sigwi.at(id))/dx;
		    dsigvdni.at(id)=0.;//(sigvi.at(id)-sigvi.at(id))/dx;
		    dsigudni.at(id)=0.;//(sigui.at(id)-sigui.at(id))/dx;
		    dupwpdni.at(id)=0.;//(upwpi.at(id)-upwpi.at(id))/dx;
		    sigwi.at(id)=std::max(0.f,sigwi.at(id));
		    sigvi.at(id)=std::max(0.f,sigvi.at(idim1));
		    sigui.at(id)=std::max(0.f,sigui.at(id));
		    ustarij.at(id)=std::max(0.f,ustarij.at(id));
		  }
		  else{
		    dsigwdni.at(id)=(sigwi.at(idim1)-sigwi.at(id))/dx;
		    dsigvdni.at(id)=(sigvi.at(idim1)-sigvi.at(id))/dx;
		    dsigudni.at(id)=(sigui.at(idim1)-sigui.at(id))/dx;
		    dupwpdni.at(id)=(upwpi.at(idim1)-upwpi.at(id))/dx;
		    sigwi.at(id)=std::max(sigwi.at(idim1),sigwi.at(id));
		    sigvi.at(id)=std::max(sigvi.at(idim1),sigvi.at(idim1));
		    sigui.at(id)=std::max(sigui.at(idim1),sigui.at(id));
		    ustarij.at(id)=std::max(ustarij.at(idim1),ustarij.at(id));
		  }
		  dsigwdni.at(id)=0.f;
		  dsigvdni.at(id)=0.f;
		  dsigudni.at(id)=0.f;
		  dupwpdni.at(id)=0.f;
                                
		  if(idim1<0){
		    if(fabs(upwpi.at(id))<fabs(0.f)){
		      upwpi.at(id)=upwpi.at(id);
		    }
		  }
		  else{
		    if(fabs(upwpi.at(id))<fabs(upwpi.at(idim1))){
		      upwpi.at(id)=upwpi.at(idim1);
		    }
		    else{
		      upwpi.at(idim1)=upwpi.at(id);
		    }
		  }
		}  
	      }
	    }
                    
	    if(dym.at(id)>=dy && dyp.at(id)>=dy && j!=0 && j!=ny-1){
	      if(idjm1<0){
		dsigwdy=.5*(sigwi.at(idjp1)-sigwi.at(id))/dy;
		dsigvdy=.5*(sigvi.at(idjp1)-sigvi.at(id))/dy;
		dsigudy=.5*(sigui.at(idjp1)-sigui.at(id))/dy;
		dupwpdy=.5*(upwpi.at(idjp1)-upwpi.at(id))/dy;
	      }
	      else{
		dsigwdy=.5*(sigwi.at(idjp1)-sigwi.at(idjm1))/dy;
		dsigvdy=.5*(sigvi.at(idjp1)-sigvi.at(idjm1))/dy;
		dsigudy=.5*(sigui.at(idjp1)-sigui.at(idjm1))/dy;
		dupwpdy=.5*(upwpi.at(idjp1)-upwpi.at(idjm1))/dy;
	      }
	    }
	    else{
	      if(j==1||j==ny-1){
		dsigwdy=0.;
	      }
	      else{
		if(dym.at(id)<dy && dyp.at(id)>dy){
		  //mdw 11-21-2006 modified if statements to address particle in 3-walled cells
		  dsigwdni.at(id)=(sigwi.at(idjp1)-sigwi.at(id))/dy;
		  dsigvdni.at(id)=(sigvi.at(idjp1)-sigvi.at(id))/dy;
		  dsigudni.at(id)=(sigui.at(idjp1)-sigui.at(id))/dy;
		  dupwpdni.at(id)=(upwpi.at(idjp1)-upwpi.at(id))/dy;
		  dsigwdni.at(id)=0.;
		  dsigvdni.at(id)=0.;
		  dsigudni.at(id)=0.;
		  dupwpdni.at(id)=0.;
		  sigwi.at(id)=std::max(sigwi.at(idjp1),sigwi.at(id));
		  sigvi.at(id)=std::max(sigvi.at(idjp1),sigvi.at(id));
		  sigui.at(id)=std::max(sigui.at(idjp1),sigui.at(id));
		  ustarij.at(id)=std::max(ustarij.at(idjp1),ustarij.at(id));
		  if(fabs(upwpi.at(id))<fabs(upwpi.at(idjp1))){
		    upwpi.at(id)=upwpi.at(idjp1);
		  }
		  else{
		    upwpi.at(idjp1)=upwpi.at(id);
		  }
		}
		if(dyp.at(id)<dy && dym.at(id)>dy){
		  //mdw 11-21-2005 modified if statements to address particle in 3-walled cells
		  if(idjm1<0){
		    dsigwdni.at(id)=0.;//(sigwi.at(id)-sigwi.at(id))/dy;
		    dsigvdni.at(id)=0.;//(sigvi.at(id)-sigvi.at(id))/dy;
		    dsigudni.at(id)=0.;//(sigui.at(id)-sigui.at(id))/dy;
		    dupwpdni.at(id)=0.;//(upwpi.at(id)-upwpi.at(id))/dy;
                                    
		    sigwi.at(id)=std::max(0.f,sigwi.at(id));
		    sigvi.at(id)=std::max(0.f,sigvi.at(id));
		    sigui.at(id)=std::max(0.f,sigui.at(id));
		    ustarij.at(id)=std::max(0.f,ustarij.at(id));
		  }
		  else{
		    dsigwdni.at(id)=(sigwi.at(idjm1)-sigwi.at(id))/dy;
		    dsigvdni.at(id)=(sigvi.at(idjm1)-sigvi.at(id))/dy;
		    dsigudni.at(id)=(sigui.at(idjm1)-sigui.at(id))/dy;
		    dupwpdni.at(id)=(upwpi.at(idjm1)-upwpi.at(id))/dy;
                                    
		    sigwi.at(id)=std::max(sigwi.at(idjm1),sigwi.at(id));
		    sigvi.at(id)=std::max(sigvi.at(idjm1),sigvi.at(id));
		    sigui.at(id)=std::max(sigui.at(idjm1),sigui.at(id));
		    ustarij.at(id)=std::max(ustarij.at(idjm1),ustarij.at(id));
		  }
                                
		  dsigwdni.at(id)=0.f;
		  dsigvdni.at(id)=0.f;
		  dsigudni.at(id)=0.f;
		  dupwpdni.at(id)=0.f;
		  if(idjm1<0){
		    if(fabs(upwpi.at(id))<fabs(0.f)){
		      upwpi.at(id)=upwpi.at(idjm1);
		    }
		  }
		  else{
		    if(fabs(upwpi.at(id))<fabs(upwpi.at(idjm1))){
		      upwpi.at(id)=upwpi.at(idjm1);
		    }
		    else{
		      upwpi.at(idjm1)=upwpi.at(id);
		    }
		  }
				
		}
	      }
	    }
	    if(dzm.at(id)>dz && k!=nzdz-1 && dzp.at(id)>dz){
	      if(idkm1<0){
		dsigwdz=(sigwi.at(idkp1))/(dz_array.at(k)+0.5+dz_array.at(k+1));
		dsigvdz=(sigvi.at(idkp1))/(dz_array.at(k)+0.5+dz_array.at(k+1));
		dsigudz=(sigui.at(idkp1))/(dz_array.at(k)+0.5+dz_array.at(k+1));
		dupwpdz=(upwpi.at(idkp1))/(dz_array.at(k)+0.5+dz_array.at(k+1));
	      }
	      else{
		dsigwdz=(sigwi.at(idkp1)-sigwi.at(idkm1))/(dz_array.at(k)+.5*dz_array.at(k-1)+dz_array.at(k+1));
		dsigvdz=(sigvi.at(idkp1)-sigvi.at(idkm1))/(dz_array.at(k)+.5*dz_array.at(k-1)+dz_array.at(k+1));
		dsigudz=(sigui.at(idkp1)-sigui.at(idkm1))/(dz_array.at(k)+.5*dz_array.at(k-1)+dz_array.at(k+1));
		dupwpdz=(upwpi.at(idkp1)-upwpi.at(idkm1))/(dz_array.at(k)+.5*dz_array.at(k-1)+dz_array.at(k+1));
	      }
	    }
	    if(dzm.at(id)<=dz  && k!=nzdz-1 && dzp.at(id)>dz){
	      dsigwdn=(sigwi.at(idkp1)-sigwi.at(id))/(dz_array.at(k)+.5+dz_array.at(k+1));
	      dsigvdn=(sigvi.at(idkp1)-sigvi.at(id))/(dz_array.at(k)+.5+dz_array.at(k+1));
	      dsigudn=(sigui.at(idkp1)-sigui.at(id))/(dz_array.at(k)+.5+dz_array.at(k+1));
	      dupwpdn=(upwpi.at(idkp1)-upwpi.at(id))/(dz_array.at(k)+.5+dz_array.at(k+1));
	      //mdw 9-26-2005 force dsigwdn to be zero near surface
	      dsigwdn=0.;
	      dsigudn=0.;
	      dsigvdn=0.;
	      dupwpdn=0.;
	      sigui.at(id)=std::max(sigui.at(idkp1),sigui.at(id));
	      sigvi.at(id)=std::max(sigvi.at(idkp1),sigvi.at(id));
	      sigwi.at(id)=std::max(sigwi.at(idkp1),sigwi.at(id));
	      ustarij.at(id)=std::max(ustarij.at(idkp1),ustarij.at(id));
	      if(fabs(upwpi.at(id))<fabs(upwpi.at(idkp1))){
		upwpi.at(id)=upwpi.at(idkp1);
	      }
	      else{
		upwpi.at(idkp1)=upwpi.at(id);
	      }
	    }
	    if(dzp.at(id)<=dz && k!=nzdz-1 && dzm.at(id)>dz){
	      //mdw 9-26-2005 force dsigwdn to be zero near surface
	      dsigwdn=0.;
	      dsigudn=0.;
	      dsigvdn=0.;
	      dupwpdn=0.;
	      if(idkm1<0){
		sigui.at(id)=std::max(0.f,sigui.at(id));
		sigvi.at(id)=std::max(0.f,sigvi.at(id));
		sigwi.at(id)=std::max(0.f,sigwi.at(id));
		ustarij.at(id)=std::max(0.f,ustarij.at(id));
	      }
	      else{
		sigui.at(id)=std::max(sigui.at(idkm1),sigui.at(id));
		sigvi.at(id)=std::max(sigvi.at(idkm1),sigvi.at(id));
		sigwi.at(id)=std::max(sigwi.at(idkm1),sigwi.at(id));
		ustarij.at(id)=std::max(ustarij.at(idkm1),ustarij.at(id));
	      }
	      if(idkm1<0){
		if(fabs(upwpi.at(id))<0.f){
		  upwpi.at(id)=0.;
		}
	      }
	      else{
		if(fabs(upwpi.at(id))<fabs(upwpi.at(idkm1))){
		  upwpi.at(id)=upwpi.at(idkm1);
		}
		else{
		  upwpi.at(idkm1)=upwpi.at(id);
		}
	      }
                        
	    }
	    // erp June 18, 2009: only need diswdni used to calculate tau gradients
	    // export this value to be used, send to shaders
	    if((dxm.at(id)>=dx)&&(dxp.at(id)>=dx)&&(dym.at(id)>=dy)&& 
	       (dyp.at(id)>=dy)&&(dzm.at(id)>=dz)&&(dzp.at(id)>=dz)){
	      dsigwdn=ani.at(id)*dsigwdx+bni.at(id)*dsigwdy+cni.at(id)*dsigwdz;
	      dsigvdn=ani.at(id)*dsigvdx+bni.at(id)*dsigvdy+cni.at(id)*dsigvdz;
	      dsigudn=ani.at(id)*dsigudx+bni.at(id)*dsigudy+cni.at(id)*dsigudz;
	      dupwpdn=ani.at(id)*dupwpdx+bni.at(id)*dupwpdy+cni.at(id)*dupwpdz;
	    }
	    dsigwdni.at(id)=dsigwdn;
	    dsigvdni.at(id)=dsigvdn;
	    dsigudni.at(id)=dsigudn;
	    dupwpdni.at(id)=dupwpdn;
                    
	    // limiting form for near wall circumstances
	    data3[texidx+3] = dsigwdni.at(id);
	  }
	  dsigwdnOut<<xi.at(i)<<" "<<yi.at(j)<<" "<<zi.at(k)<<" "<<dsigwdni.at(id)<<std::endl;
	}//   lp030
      }//   lp031
    } //  lp032
    //500 line code ends
    createTexture(duvw_dz, GL_RGBA32F_ARB, width,height, data3);
    createTexture(dxyz_wall, GL_RGBA32F_ARB, width,height, data4);
    delete[] data3;
    delete[] data4;





#if 0 // REMOVE BELOW HERE

    unsigned data_sz = width*height*4;
    GLfloat *data     = new GLfloat[data_sz];
    GLfloat *dataWind = new GLfloat[data_sz];
    GLfloat *dataTwo  = new GLfloat[data_sz];
    GLfloat *dataTau  = new GLfloat[data_sz];
    GLfloat *data3    = new GLfloat[data_sz];
    GLfloat *data4    = new GLfloat[data_sz];

    //Balli's new additions this is essentially a direct copy of the FORTRAN
    initCellType();
    
    //Balli: Substracting 1 from nzdz as it is increased by 1  after reading from QU_simparams.inp in Util.cpp
    nzdz=nzdz-1;
    float nxnynz=nxdx*nydy*nzdz;
    std::string s;

    float dx = m_util_ptr->dx;
    float dy = m_util_ptr->dy;
    float dz = m_util_ptr->dz;
    std::vector<float> dz_array,z,zm;
    dz_array.resize(nzdz,dz);
    z.resize(nzdz,0.0f);
    zm.resize(nzdz,0.0f);
    //Balli: Initialized first element of z and zm array before begining the loop as in GPU Plume
    //we do not store values below the ground, which are zero anyways
    z.at(0)  = dz_array.at(0);
    zm.at(0) = z.at(0)-0.5*dz_array.at(0);
    
    for(int k=1;k<nzdz;k++){
        z.at(k)  = z.at(k-1)+dz_array.at(k);
        zm.at(k) = z.at(k)-0.5*dz_array.at(k);
    }
    //**************************ATTENTION********************************************************
    //Balli: Following should be grabbed from the input files: Hardwired here !!!!     IMPORTANT!!!
    float rcl  = 0.0f;  //Monin-obo length, should be in the input file-Balli-06/10/09
    float z0   = 0.1f;  //Should be in the input file-Balli-06/10/09
    int roofflag = 2;   //should be in the input file-Balli-06/10/09
    float h      = 2000.f; //Boundary Layer Height-should be in the input file-Balli-06/10/09
    //**************************ATTENTION********************************************************

    
    //Balli: declaring few constants
    const float kkar = 0.4f;           //von karman constant
    const float pi   = 4.f*atan(1.0f);
    const float knlc   = 0.113f;         
    const float ctau13 = 1.f;           
    const float cusq   = 2.5f*2.5f;     
    const float cvsq   = 2.f*2.f;         
    const float cwsq   = 1.3f*1.3f;    


    //Balli: "theta" is never read into or initilized in the FORTRAN code, but used for calculating ualoft and valoft.
    float theta  = 0.f;          // This varibale is not used anymore in QP but the legacy code still uses it-Balli-06/10/09
    float ualoft = 0.f;          
    float valoft = 0.f;             

    int time_idx=1;
    int check1=0;
    int check2=0;
    int check3=0;
    int check=0;

    //Balli: For writing turbulence data- Can be removed later
    std::ofstream turbfield;
    turbfield.open("GPU_turbfield.dat");
    
    //Balli : Declaring local vectors
    std::vector<float> elz,ustarz,sigwi,sigvi,ustarij,xi,yi,zi,hgt,hgtveg,eleff,xcb,ycb,icb,jcb,phib,weff,leff,lfr,zcorf,lr;
    std::vector<float>uref,urefu,urefv,urefw, utotktp,uktop,vktop,wktop,deluc,ustargz,elzg,ustarg;
    std::vector<float>utotcl1,utotmax,gamma,atten,Sx,Sy;
    std::vector<int>bldtype;
    
    //Balli : Vectors needs to be resized before they are  used otherwise they sometime give runtime errors
    eleff.resize(nxnynz,0.0);// efective length scale-initialized with zero values.
    ustarg.resize(nxnynz,0.0);


    
    //Balli : Reading "QU_buildout.inp"; THis should be handled in Util.cpp
    //This file is required for getiign effective lenths of downstream and upstream(along with other parameters) cavities from QUICURB
    std::ifstream QUbldout;
    char char18[1024];
    QUbldout.open("QP_buildout.inp");
    int inumveg,inumgarage;
    //Balli: IMPORTANT!!!!
    
    //OVERWRITING!!! the building parameters here and reading them from QP_bldout file. The values of the key parameters(xfo,yfo,zfo,ht,lti,wti)
    //will remain same as bldout file has all the information contained in QU_buildings with some additional information needed for
    //non-local mixing
    QUbldout>>numBuild;
    bldtype.resize(numBuild);
    gamma.resize(numBuild);
    atten.resize(numBuild);
    Sx.resize(numBuild);
    Sy.resize(numBuild);
    weff.resize(numBuild);
    leff.resize(numBuild);
    lfr.resize(numBuild);
    lr.resize(numBuild);
    std::getline(QUbldout,s);
    QUbldout>>inumveg;
    std::getline(QUbldout,s);
    
    for(int i=0;i<numBuild-inumveg;i++){
        QUbldout>>char18>>char18>>char18;
        int buildnum;
        QUbldout>>buildnum;
        QUbldout>>char18>>char18;
        QUbldout>>bldtype.at(i);
        QUbldout>>char18>>char18;
        QUbldout>>gamma.at(i);
        QUbldout>>char18>>char18;
        QUbldout>>ht[i];
        QUbldout>>char18>>char18;
        QUbldout>>wti[i];
        QUbldout>>char18>>char18;
        QUbldout>>lti[i];
        QUbldout>>char18>>char18;
        QUbldout>>xfo[i];
        QUbldout>>char18>>char18;
        QUbldout>>yfo[i];
        QUbldout>>char18>>char18;
        QUbldout>>zfo[i];
        QUbldout>>char18>>char18;
        QUbldout>>weff.at(i);
        QUbldout>>char18>>char18;
        QUbldout>>leff.at(i);
        QUbldout>>char18>>char18;
        QUbldout>>lfr.at(i);
        QUbldout>>char18>>char18;
        QUbldout>>lr.at(i);
        QUbldout>>char18>>char18;
        QUbldout>>atten.at(i);
        QUbldout>>char18>>char18;
        QUbldout>>Sx.at(i);
        QUbldout>>char18>>char18;
        QUbldout>>Sy.at(i);
        QUbldout>>char18>>char18>>char18>>char18;
    }
    QUbldout.close();
    
    //Balli: IMPORTANT!!
    //QP differs from GPU in the indices i,j,k of all the arrays (u,v,w etc.)
    //Following calculations are for Boundary layer case only and it provides an insight into the coordinate sytem differences
    //QP velocity vectors, for example u(i,j,k), i goes from 1(0.5) to nx(9.5), j goes from 1(0.5) to ny(9.5)
    //and k goes from 1(-0.5) to nz+1(29.5)
    //[Note: nx,ny,nz above are what QP reads from input file, QP adds 1 to nx and ny, and 2 to nz after reading them from input file]
    
    //QP's k goes from 1 to nz+2, therefore for dz=1, zi goes from -0.5(k=1) to 30.5(k=32)
    //GPU k goes from 0 to nz-1, therefore for dz=1, zi goes from 0.5(k=0) to 29.5 (k=29)

    zi.resize(nzdz);
    for(int k=0;k<nzdz;k++){ 
        zi.at(k)=.5*dz+dz*k; //this expression is different from that used in the QP, but values are same
    }
    //QP's j goes from 1 to ny+1, therefore for dy=1, yi goes from 0.5(j=1) to 10.5(j=11)
    //GPU j goes from 0 to ny-1, therefore for dy=1, yi goes from 0.5(j=0) to 9.5 (j=9)
    
    yi.resize(nydy);
    for(int j=0;j<nydy;j++){
        yi.at(j)=.5*dy+dy*j;
    }
    //QP's i goes from 1 to nx+1, therefore for dx=1, xi goes from 0.5(i=1) to 10.5(i=11)
    //GPU i goes from 0 to nx-1, therefore for dx=1, xi goes from 0.5(i=0) to 9.5 (i=9)
    
    xi.resize(nxdx);
    for(int i=0;i<nxdx;i++){
        xi.at(i)=.5*dx+dx*i;
    }
    
    float ht_avg = 0.0;
    int k=0;
    if(numBuild > 0 && numBuild != inumveg){
        for(int  i_b=0;i_b<numBuild;i_b++){
            if(bldtype.at(i_b)==9){
                continue;
            }
            ht_avg=ht[i_b]+zfo[i_b]+ht_avg;
        }
        ht_avg=ht_avg/float(numBuild-inumveg);
        float temp=ht_avg/dz;
        for(int kk=0;kk<nzdz;kk++){
            k=kk;
            if(ht_avg<z.at(kk))break;
        }
    }
    else{
        //BL Flow case: Control comes here as we have no buildings for this test case
        k=0; //altered to comply with GPU
    }
    dz=dz_array.at(k);
    
    //Obtain avg. velocity from the boundary of the domain at above cal avg ht of the buildings
    int i=0;//altered to comply with GPU
    int j=0;
    float u_left=0;
    for(j=0;j<nydy;j++){
        int p2idx = k*nxdx*nydy + j*nxdx + i;
        u_left=sqrt(wind_vel[p2idx].u*wind_vel[p2idx].u + wind_vel[p2idx].v*wind_vel[p2idx].v + wind_vel[p2idx].w*wind_vel[p2idx].w) +u_left;
    }
    u_left=u_left/(nydy);//altered to comply GPU
    // in QP, total number of cells in y is read from input file as ny and then QP adds 1to ny, therefore, 1 is substracted from ny above in QP.
    
    j=nydy-1;// substracted 1 as edge of the domain in y is nydy-1
    float u_top=0;
    for(i=0;i<nxdx;i++){
        int p2idx = k*nxdx*nydy + j*nxdx + i;
        u_top=sqrt(wind_vel[p2idx].u*wind_vel[p2idx].u + wind_vel[p2idx].v*wind_vel[p2idx].v + wind_vel[p2idx].w*wind_vel[p2idx].w) +u_top;
    }
    u_top=u_top/nxdx;
    i=nxdx-1;//same explanation as in case of j above
    float u_right=0;
    for(j=0;j<nydy;j++){
        int p2idx = k*nxdx*nydy + j*nxdx + i;
        u_right=sqrt(wind_vel[p2idx].u*wind_vel[p2idx].u + wind_vel[p2idx].v*wind_vel[p2idx].v + wind_vel[p2idx].w*wind_vel[p2idx].w) +u_right;
    }
    u_right=u_right/nydy;
    j=0;//alterted for GPU
    
    float u_bottom=0;
    for(i=0;i<nxdx;i++){
        int p2idx = k*nxdx*nydy + j*nxdx + i;
        u_bottom=sqrt(wind_vel[p2idx].u*wind_vel[p2idx].u + wind_vel[p2idx].v*wind_vel[p2idx].v + wind_vel[p2idx].w*wind_vel[p2idx].w) +u_bottom;
    }
    u_bottom=u_bottom/(nxdx);
    
    float u_b=0.25*(u_left+ u_top+ u_right+ u_bottom);//average velocity
    float nu_b=1.5e-5; //nu for air
    float del_b=(0.328* pow(nu_b/u_b,.2f) ) * pow(ht_avg,.8f);//expression for BL layer thickness (growth) (i think)
    // above expression is used for obtaining turbulence close to the walls of the buildings
    
    hgt.resize(nxnynz,0.);
    hgtveg.resize(nxnynz,0.);
    if(time_idx == 1){
        for(int k=0; k<nzdz;k++){
            for(int j=0; j<nydy;j++){
                for(int i=0; i<nxdx;i++){
                    int p2idx = k*nxdx*nydy + j*nxdx + i;
                    int ij = j*nxdx + i;
                    if(cellQuic[p2idx].c == 0)hgt.at(ij)=std::max(hgt.at(ij),z.at(k));
                    if(cellQuic[p2idx].c == 8)hgtveg.at(ij)=std::max(hgtveg.at(ij),z.at(k));
                }
            }
        }
    }    
    
    elz.resize(nxnynz);
    ustarz.resize(nxnynz);
    sigwi.resize(nxnynz);
    sigvi.resize(nxnynz);
    ustarij.resize(nxnynz);
    ustarz.resize(nxnynz);

    
    //Balli:Allocating global arrays, declared in header file
    dutotdxi.resize(nxnynz);
    dutotdyi.resize(nxnynz);
    dutotdzi.resize(nxnynz,0.0);
    dutotdni.resize(nxnynz);
    dutotdsi.resize(nxnynz);
    alph1ij.resize(nxnynz);
    alph2ij.resize(nxnynz);
    alph3ij.resize(nxnynz);
    bet1ij.resize(nxnynz);
    bet2ij.resize(nxnynz);
    bet3ij.resize(nxnynz);
    gam1ij.resize(nxnynz);
    gam2ij.resize(nxnynz);
    gam3ij.resize(nxnynz);
    alphn1ij.resize(nxnynz);
    alphn2ij.resize(nxnynz);
    alphn3ij.resize(nxnynz);
    betn1ij.resize(nxnynz);
    betn2ij.resize(nxnynz);
    betn3ij.resize(nxnynz);
    gamn1ij.resize(nxnynz);
    gamn2ij.resize(nxnynz);
    gamn3ij.resize(nxnynz);
    ani.resize(nxnynz);
    bni.resize(nxnynz);
    cni.resize(nxnynz);
    ufsqgi.resize(nxnynz);
    vfsqgi.resize(nxnynz);
    wfsqgi.resize(nxnynz);
    ufvfgi.resize(nxnynz);
    ufwfgi.resize(nxnynz);
    vfwfgi.resize(nxnynz);

    
    std::vector<float> dzm,dzp,dym,dyp,dxm,dxp,ufwfi,ufvfi,vfwfi,sigui,upwpi,epsi;
    
    dzm.resize(nzdz*nydy*nxdx);
    dzp.resize(nzdz*nydy*nxdx);
    dxm.resize(nzdz*nydy*nxdx);
    dxp.resize(nzdz*nydy*nxdx);
    dym.resize(nzdz*nydy*nxdx);
    dyp.resize(nzdz*nydy*nxdx);
    ufwfi.resize(nzdz*nydy*nxdx);
    ufvfi.resize(nzdz*nydy*nxdx);
    vfwfi.resize(nzdz*nydy*nxdx);
    sigui.resize(nzdz*nydy*nxdx);
    upwpi.resize(nzdz*nydy*nxdx);
    epsi.resize(nzdz*nydy*nxdx);
    
    int kcantop=0;
    float ucantop=0.;
    float elcanopy=0.;
    float utotl=0.;
    float utotu=0.;
    float phim=0.;
    float psim=0.;
    float dutotl=0.;
    float dutotu=0.f;
    float dutot=0.f;
    float utot=0.f;
    float dutotdzc=0.;
    float dutotdzp=0.;
    float dutotdzm=0.;
    float dutotdza=0.;
   
    for(int j=0;j<nydy;j++){
        for(int i=0;i<nxdx;i++){
            int ij = j*nxdx + i;
            if(hgtveg.at(ij) > 0.){
                for(int kk=0;kk<nzdz;k++){
                    kcantop=kk;
                    if(hgtveg.at(ij) <= z.at(kk))break;
                }
                int idcan  = kcantop*nxdx*nydy + j*nxdx + i;
                int id1can = (kcantop+1)*nxdx*nydy + j*nxdx + i;
                
                ucantop=.5*sqrt(wind_vel[idcan].u*wind_vel[idcan].u+wind_vel[idcan].v*wind_vel[idcan].v+wind_vel[idcan].w*wind_vel[idcan].w)
                    + .5*sqrt(wind_vel[id1can].u*wind_vel[id1can].u+wind_vel[id1can].v*wind_vel[id1can].v+wind_vel[id1can].w*wind_vel[id1can].w); 
            }
            for(int k=0;k<nzdz;k++){

                dz=dz_array.at(k);
                int km1   = (k-1)*nxdx*nydy + j*nxdx + i;
                int kp1   = (k+1)*nxdx*nydy + j*nxdx + i;
                int knz1 = (nzdz-1)*nxdx*nydy + j*nxdx + i;
                int p2idx = k*nxdx*nydy + j*nxdx + i;
                int ij = j*nxdx + i;
                int idklow=0;
                //new changes from QUIC
                if(cellQuic[p2idx].c != 0){
                    dzm.at(p2idx)=zm.at(k)-hgt.at(ij);
                    eleff.at(p2idx)=dzm.at(p2idx);
                }
                else{
                    dzm.at(p2idx)=0.f;
                    eleff.at(p2idx)=0.f;
                }
                elcanopy=0.f;
                if(((cellQuic[km1].c == 0) || (cellQuic[km1].c==8)) && 
                   (cellQuic[p2idx].c != 0 && cellQuic[p2idx].c != 8) || k == 0){//altered k
                    utotl=0.f;
                    utotu=sqrt(wind_vel[p2idx].u*wind_vel[p2idx].u+wind_vel[p2idx].v*wind_vel[p2idx].v+wind_vel[p2idx].w*wind_vel[p2idx].w);
                    //MDW 7-01-2005 changed the way vertical gradients are calculated to avoid inaccuracies
                    // in the representation of the gradients of a log-law term
                    
                    if(rcl>0){
                        phim=1.f+4.7f*rcl*.5f*dz;
                        psim=-4.7f*rcl*.5f*dz;
                    }
                    else{
                        phim=pow( (1.f-15.f*rcl*.5f*dz),(-.25f));
                        psim=2.f*log((1.f+1.f/phim)/2.f)+log((1.f+1.f/pow(phim,2.f))/2.f)-2.f*atan(1.f/phim)+pi/2.f;
                    }
                    if(hgtveg.at(ij)>zi.at(k)){                           
                        phim=1.f;
                        psim=0.f;
                        elz.at(p2idx)=elcanopy*std::min(1.f,(zi.at(k)-z0)/(.3f*hgtveg.at(ij)));
                        ustar=elz.at(p2idx)*utotu/(.5f*dz);
                        dutotdzi.at(p2idx)=utotu/(.5f*dz);
                        ustarz.at(p2idx)=ustar;
                    }
                    else{
                        if(cellQuic[km1].c!=8){
                            ustar=kkar*utotu/(log(.5f*dz/z0)-psim);
                            elz.at(p2idx)=kkar*.5f*dz;
                            ustarz.at(p2idx)=ustar;
                            dutotdzi.at(p2idx)=ustar*phim/elz.at(p2idx);
                        }
                        else{
                            utotl=sqrt(wind_vel[km1].u*wind_vel[km1].u+wind_vel[km1].v*wind_vel[km1].v+wind_vel[km1].w*wind_vel[km1].w);
                            dutotdzi.at(p2idx)=2.f*(utotu-utotl)/(dz_array.at(k-1)+dz_array.at(k));
                            elz.at(p2idx)=kkar*.5f*dz;
                            ustar=elz.at(p2idx)*dutotdzi.at(p2idx);
                            ustarz.at(p2idx)=ustar;
                        }
                    }
                    if(cellQuic[km1].c!=8 && k!=0){
                        sigwi.at(km1)=0.f;
                        sigvi.at(km1)=0.f;
                        ustarij.at(km1)=0.f;
                        ustarz.at(km1)=0.f;
                    }
                }
                else{
                    if(k==nzdz-1){ // find gradient using a non-CDD approach
                        utotl=sqrt(wind_vel[km1].u*wind_vel[km1].u+wind_vel[km1].v*wind_vel[km1].v+wind_vel[km1].w*wind_vel[km1].w);
                        utotu=sqrt(wind_vel[p2idx].u*wind_vel[p2idx].u+wind_vel[p2idx].v*wind_vel[p2idx].v+wind_vel[p2idx].w*wind_vel[p2idx].w);
                        dutotdzi.at(knz1)=dutotdzi.at(km1)*zm.at(k-1)/zm.at(k);
                        elz.at(p2idx)=kkar*(eleff.at(p2idx)-hgtveg.at(ij));
                    }
                    else{ // find gradient using a CDD approach
                        utotl=sqrt(wind_vel[km1].u*wind_vel[km1].u+wind_vel[km1].v*wind_vel[km1].v+wind_vel[km1].w*wind_vel[km1].w);
                        utotu=sqrt(wind_vel[kp1].u*wind_vel[kp1].u+wind_vel[kp1].v*wind_vel[kp1].v+wind_vel[kp1].w*wind_vel[kp1].w);
                        // mdw 7-08-2005 changed the way vertical gradients are calculated to better represent
                        // log-law behavior
                        if(cellQuic[p2idx].c==8){
                            dutotdzi.at(p2idx)=(utotu-utotl)/(dz_array.at(k)+.5*dz_array.at(k-1)+.5*dz_array.at(k+1));
                            ustarz.at(p2idx)=elcanopy*dutotdzi.at(p2idx);
                            elz.at(p2idx)=elcanopy*std::min(1.f,(zi.at(k)-z0)/(0.3f*hgtveg.at(ij)));
                        }
                        else{
                            int klow=0;
                            for (int kk=0;kk<nzdz;kk++){
                                klow=kk;
                                if(std::max(hgt.at(ij),hgtveg.at(ij))<z.at(kk))break;
                            }

                            idklow = klow*nxdx*nydy + j*nxdx + i;
                            if(rcl>0){
                                phim=1.f+4.7f*rcl*(eleff.at(km1)-std::max((hgtveg.at(ij)-elcanopy/kkar),0.f));
                                psim=-4.7f*rcl*(eleff.at(km1)-std::max((hgtveg.at(ij)-elcanopy/kkar),0.f));
                            }
                            else{
                                phim=pow( (1.f-15.f*rcl*(eleff.at(km1)-std::max((hgtveg.at(ij)-elcanopy/kkar),0.f))),(-.25f));
                                psim=2.f*log((1.f+1.f/phim)/2.f)+log((1.f+1.f/pow(phim,2.f))/2.f)-2.f*atan(1.f/phim)+pi/2.f;
                            }
                            dutotl=utotl-ustarz.at(idklow)*(log((zi.at(k-1)-std::max((hgtveg.at(ij)-elcanopy/kkar),0.f))/z0)-psim)/kkar;
                            if(rcl>0){
                                phim=1.f+4.7f*rcl*(eleff.at(kp1)-std::max((hgtveg.at(ij)-elcanopy/kkar),0.f));
                                psim=-4.7f*rcl*(eleff.at(kp1)-std::max((hgtveg.at(ij)-elcanopy/kkar),0.f));
                            }
                            else{
                                phim=pow( (1.f-15.f*rcl*(eleff.at(kp1)-std::max((hgtveg.at(ij)-elcanopy/kkar),0.f))),(-.25f));
                                psim=2.f*log((1.f+1.f/phim)/2.f)+log((1.f+1.f/pow(phim,2.f))/2.f)-2.f*atan(1.f/phim)+pi/2.f;
                            }
                            dutotu=utotu-ustarz.at(idklow)*(log((zi.at(k+1)-std::max((hgtveg.at(ij)-elcanopy/kkar),0.f))/z0)-psim)/kkar;
                            dutotdzi.at(p2idx)=(dutotu-dutotl)/(dz_array.at(k)+.5*dz_array.at(k-1)+.5*dz_array.at(k+1))
                                +ustarz.at(idklow)*psim/(kkar*zi.at(k));
                            elz.at(p2idx)=kkar*(eleff.at(p2idx)-std::max((hgtveg.at(ij)-elcanopy/kkar),0.f));
                              if(cellQuic[kp1].c != 0 && cellQuic[p2idx].c != 0  && cellQuic[km1].c != 0){
                                  // mdw 7-01-2005 centered around k instead of k-1 and ajusted for log-law behavior
                                  utot=sqrt(wind_vel[p2idx].u*wind_vel[p2idx].u+wind_vel[p2idx].v*wind_vel[p2idx].v+wind_vel[p2idx].w*wind_vel[p2idx].w);
                                  utotl=sqrt(wind_vel[km1].u*wind_vel[km1].u+wind_vel[km1].v*wind_vel[km1].v+wind_vel[km1].w*wind_vel[km1].w);
                                  utotu=sqrt(wind_vel[kp1].u*wind_vel[kp1].u+wind_vel[kp1].v*wind_vel[kp1].v+wind_vel[kp1].w*wind_vel[kp1].w);
                                  if(rcl>0){
                                      phim=1.f+4.7f*rcl*(eleff.at(km1)-std::max((hgtveg.at(ij)-elcanopy/kkar),0.f));
                                      psim=-4.7f*rcl*(eleff.at(km1)-std::max((hgtveg.at(ij)-elcanopy/kkar),0.f));
                                  }
                                  else{
                                      phim=pow( (1.-15.*rcl*(eleff.at(km1)-std::max((hgtveg.at(ij)-elcanopy/kkar),0.f))),(-.25));
                                      psim=2.*log((1.+1./phim)/2.)+log((1.+1./pow(phim,2.f))/2.)-2.*atan(1./phim)+pi/2.;
                                  }
                                  dutotl=utotl-ustarz.at(idklow)*(log((zi.at(k-1)-std::max((hgtveg.at(ij)-elcanopy/kkar),0.f))/z0)-psim)/kkar;
                                  if(rcl>0){
                                      phim=1.+4.7*rcl*(eleff.at(kp1)-std::max((hgtveg.at(ij)-elcanopy/kkar),0.f));
                                      psim=-4.7*rcl*(eleff.at(kp1)-std::max((hgtveg.at(ij)-elcanopy/kkar),0.f));
                                  }
                                  else{
                                      phim=pow( (1.-15.*rcl*(eleff.at(kp1)-std::max((hgtveg.at(ij)-elcanopy/kkar),0.f))),(-.25));
                                      psim=2.*log((1.+1./phim)/2.)+log((1.+1./pow(phim,2.f))/2.)-2.*atan(1./phim)+pi/2.;
                                  }
                                  dutotu=utotu-ustarz.at(idklow)*(log((zi.at(k+1)-std::max((hgtveg.at(ij)-elcanopy/kkar),0.f))/z0)-psim)/kkar;
                                  // mdw 3-08-2004 begin changes for highest gradient rather than centered diff gradient
                                  if(rcl>0){
                                      phim=1.+4.7*rcl*(eleff.at(p2idx)-std::max((hgtveg.at(ij)-elcanopy/kkar),0.f));
                                      psim=-4.7*rcl*(eleff.at(p2idx)-std::max((hgtveg.at(ij)-elcanopy/kkar),0.f));
                                  }
                                  else{
                                      phim=pow( (1.-15.*rcl*(eleff.at(p2idx)-std::max((hgtveg.at(ij)-elcanopy/kkar),0.f))),(-.25) );
                                      psim=2.*log((1.+1./phim)/2.)+log((1.+1./pow(phim,2.f))/2.)-2.*atan(1./phim)+pi/2.;
                                  }
                                  dutot=utot-ustarz.at(idklow)*(log((zi.at(k)-std::max((hgtveg.at(ij)-elcanopy/kkar),0.f))/z0)-psim)/kkar;
                                  dutotdzc=(dutotu-dutotl)/(dz_array.at(k)+.5*dz_array.at(k+1)+.5*dz_array.at(k-1));
                                  dutotdzp=(dutotu-dutot)/(.5*dz_array.at(k+1)+.5*dz_array.at(k));
                                  dutotdzm=(dutot-dutotl)/(.5*dz_array.at(k)+.5*dz_array.at(k-1));
                                  dutotdza=0.5*(fabs(dutotdzp+ustarz.at(idklow)*phim/(kkar*(zi.at(k) 
                                                                                          -std::max((hgtveg.at(ij)-elcanopy/kkar),0.f)))) 
                                                +fabs(dutotdzm+ustarz.at(idklow)*phim/(kkar*(zi.at(k) 
                                                                                           -std::max((hgtveg.at(ij)-elcanopy/kkar),0.f)))));
                                  if(abs(dutotdzp+ustarz.at(idklow)*phim/(kkar*(zi.at(k)-std::max((hgtveg.at(ij)-elcanopy/kkar),0.f))))> 
                                     fabs(dutotdzm+ustarz.at(idklow)*phim/(kkar*(zi.at(k)-std::max((hgtveg.at(ij)-elcanopy/kkar),0.f))))){
                                      dutotdzi.at(p2idx)=dutotdzp+ustarz.at(idklow)*phim/(kkar*(zi.at(k) 
                                                                                               -std::max((hgtveg.at(ij)-elcanopy/kkar),0.f)));
                                  }
                                  else{
                                      dutotdzi.at(p2idx)=dutotdzm+ustarz.at(idklow)*phim/(kkar*(zi.at(k) 
                                                                                                -std::max((hgtveg.at(ij)-elcanopy/kkar),0.f)));
                                      
                                      
                                  }
                              }
                              // use centered differences away from the boundaries
                        }
                    }
                }
            }
        }
    }//end for loops
    //Balli: Above loop is working as expected. I have matched every variable value with QP (Balli-06/14/09)
    // IMPORTANT!!! Theta is never read from input file. so its values will be zero always.
    //Therefore following few lines do not effect the final solution at all
    // phi will be calculated again by taking into account the actual wind angle at each building.

    float phi=270.-theta;
    phi=phi*pi/180.;
    float cosphi=cos(phi);
    int iupc=0;
    if(cosphi>=0)
        iupc=0;//altered for GPU
    else
        iupc=nxdx;
    
    float sinphi=sin(phi);
    int jupc=0;
    if(sinphi>=0.f)
        jupc=0;//altered for GPU
    else
        jupc=nydy;
    
    float phit=phi+0.5*pi;
    float cosphit=cos(phit);
    float sinphit=sin(phit);
    
    //Following variables are required for non-local mixing.
    float xcelt=0.f;
    float ycelt=0.f;
    int icelt=0;
    int jcelt=0;
    
    float xceln=0.f;
    float yceln=0.f;
    int iceln=0;
    int jceln=0;
    float utott=0.f;
    float delut=0.f;
    float delutz=0.f;
    xcb.resize(numBuild);
    ycb.resize(numBuild);
    icb.resize(numBuild);
    jcb.resize(numBuild);
    phib.resize(numBuild);
    zcorf.resize(nxnynz);
    uref.resize(nxnynz);
    urefu.resize(nxnynz);
    urefv.resize(nxnynz);
    urefw.resize(nxnynz);
    utotktp.resize(nxnynz);
    uktop.resize(nxnynz);
    vktop.resize(nxnynz);
    wktop.resize(nxnynz);
    deluc.resize(nxnynz);
    ustargz.resize(nxnynz);
    elzg.resize(nxnynz,(nxdx+1.)*dx);// initialized it with a value similar to QP
    utotcl1.resize(nxnynz);
    utotmax.resize(nxnynz);

    for(int i=0;i<numBuild;i++){
        if(bldtype.at(i)==9)continue;
        //! mdw 4-16-2004 added proper treatment of zfo
        float temp=0.f;
        int ktop=0;
        for(int k=0;k<nzdz;k++){
            ktop=k;
            if(ht[i]+zfo[i]<z.at(k))break;
        }
        int kmid=0;
        for(int k=0;k<nzdz;k++){        
            kmid=k;
            if(0.5*ht[i]+zfo[i]<z.at(k))break;
        }
        if(bldtype.at(i)==3){
            xcb.at(i)=xfo[i];
        }
        else{
            xcb.at(i)=xfo[i]+.5*lti[i];
        }
        ycb.at(i)=yfo[i];
        temp=(xcb.at(i)-dx)/dx;//substracted dx to comply with GPU
        icb.at(i)=nint(temp);
        temp=(ycb.at(i)-dy)/dy;//substracted dy to comply with GPU
        jcb.at(i)=nint(temp);
        //!mdw 6-05-2005 put in procedure to calculate phi & phit
        int kendv=0;
        if(roofflag==2){
            float Bs=ht[i];
            float BL=wti[i];
            
            if(wti[i]<ht[i]){
                Bs=wti[i];
                BL=ht[i];
            }
            float Rscale = ((pow(Bs,(2.f/3.f)))*(pow(BL,(1.f/3.f))));
            float temp=std::max(.22f*Rscale,.11f*wti[i]);
            float zclim  =std::max(temp,.11f*lti[i]);
            for(int k=0;k<nzdz;k++){        
                kendv=k;
                if(zclim<z.at(k))break;
            }
        }
        else{
            for(int k=0;k<nzdz;k++){        
                kendv=k;
                if(ht[i]+zfo[i]<z.at(k))break;
            }
        }
        kendv=std::min(kendv,nzdz);
        
        int idvel=kendv*nxdx*nydy + jcb.at(i)*nxdx +icb.at(i);
        double tempv=wind_vel[idvel].v;
        double tempu=wind_vel[idvel].u;
        phib.at(i)=atan2(tempv,tempu);
        phi=phib.at(i);
        cosphi=cos(phi);
        int iupc=0;
        if(cosphi>=0)
            iupc=0;//altered for GPU
        else
            iupc=nxdx;
        
        sinphi=sin(phi);
        int jupc=0;
        if(sinphi>=0)
            jupc=0;//altered for GPU
        else
            jupc=nydy;
        
        float phit=phi+0.5*pi;
        cosphit=cos(phit);
        sinphit=sin(phit);
        
        //! ycbp3, and xcbp3 give points 1.5 units outside
        //! of the bldg boundaries to compute reference utot
        float ycbp3=0.f;
        float xcbp3=0.f;
        float ycbm3=0.f;
        float xcbm3=0.f;
        int icbp3=0;
        int icbm3=0;
        int jcbp3=0;
        int jcbm3=0;
        float dycbp3=0.f;
        float dycbm3=0.f;
        float dxcbp3=0.f;
        float dxcbm3=0.f;
        float ycbp=0.f;
        float xcbp=0.f;
        float ycbm=0.f;
        float xcbm=0.f;
        float xcd,ycd,xcu,ycu,xcul,ycul,cosfac;


        if(fabs(sinphit)>=fabs(cosphit)){
            ycbp3=ycb.at(i)+(.5*weff.at(i)+.33*weff.at(i))*sinphit;// ! Get reference values for x,y for non-local mixing
            xcbp3=xcb.at(i)+(.5*weff.at(i)+.33*weff.at(i))*cosphit;// ! 1/3 bldg width outside of building is the boundary for the non-local mixing
            ycbm3=ycb.at(i)-(.5*weff.at(i)+.33*weff.at(i))*sinphit;
            xcbm3=xcb.at(i)-(.5*weff.at(i)+.33*weff.at(i))*cosphit;
            temp=(xcbp3-dx)/dx;
            icbp3=nint(temp);//substracted dx to comply gpu
            temp=(xcbm3-dx)/dx;
            icbm3=nint(temp);//substracted dx to comply gpu
            temp=(ycbp3-dy)/dy;
            jcbp3=nint(temp);//substracted dx to comply gpu
            temp=(ycbm3-dy)/dy;
            jcbm3=nint(temp);//substracted dx to comply gpu
            jcbp3=std::min(jcbp3,nydy-1);
            jcbm3=std::min(jcbm3,nydy-1);
            icbp3=std::min(icbp3,nxdx-1);
            icbm3=std::min(icbm3,nxdx-1);
            jcbp3=std::max(0,jcbp3);//changed from 1 to zeros to comply with gpu
            jcbm3=std::max(0,jcbm3);
            icbp3=std::max(0,icbp3);
            icbm3=std::max(0,icbm3);
            //! searching in the plus y direction for building free flow
            int id=kmid*nxdx*nydy + jcbp3*nxdx +icbp3;
            int jp1=0;
            int jp2=0;
            int isign=0;
            if(cellQuic[id].c == 0){
                if(sinphit>0.f){
                    jp1=jcbp3;
                    jp2=nydy-1;
                    isign=1;
                }
                else{
                    jp1=jcbp3;
                    jp2=0;//altered for GPU
                    isign=-1;
                }
            
                for(int ji=jp1;ji<=jp2;ji=ji+isign){
                    jcbp3=jcbp3+isign;
                    jcbp3=std::min(nydy-1,jcbp3);
                    dycbp3=dy*(jcbp3-1)-ycbp3;
                    ycbp3=dy*(jcbp3-1);
                    xcbp3=xcbp3+cosphit*dycbp3/sinphit;
                    icbp3=int(xcbp3/dx)+1-dx;
                    icbp3=std::min(nx-1,icbp3);
                    //!mdw 34/01/2004 forced indices to be within domain
                    int idMid=kmid*nxdx*nydy + jcbp3*nxdx +icbp3;
                    if(cellQuic[idMid].c!= 0) break;
                }
            }

            //! searching in the minus y direction for building free flow
            int id2=kmid*nxdx*nydy + jcbm3*nxdx +icbm3;
            int jm2=0;
            int jm1=0;
            isign=0;
            if(cellQuic[id2].c == 0){
                if(sinphit>0.f){
                    jm2=0;//altered for GPU;
                    jm1=jcbm3;
                    isign=1;
                }
                else{
                    jm2=nydy-1;
                    jm1=jcbm3;
                    isign=-1;
                }
                for(int ji=jm1;ji>=jm2;ji=ji-isign){// do ji=jm1,jm2,-isign 
                    jcbm3=jcbm3-isign;
                    dycbm3=dy*(jcbm3-1)-ycbm3;
                    ycbm3=dy*(jcbm3-1);
                    xcbm3=xcbm3+cosphit*dycbm3/sinphit;
                    temp=(xcbm3-dx)/dx;
                    icbm3=nint(temp);
                                        
                    jcbp3=std::min(jcbp3,ny-1);
                    jcbm3=std::min(jcbm3,ny-1);
                    icbp3=std::min(icbp3,nx-1);
                    icbm3=std::min(icbm3,nx-1);
                    jcbp3=std::max(0,jcbp3);
                    jcbm3=std::max(0,jcbm3);
                    icbp3=std::max(0,icbp3);
                    icbm3=std::max(0,icbm3);
                    int idMid2=kmid*nxdx*nydy + jcbm3*nxdx +icbm3;
                    if(cellQuic[idMid2].c != 0) break;
                }
            }
            ycbp=ycb.at(i)+(.5*leff.at(i))*sinphi;
            xcbp=xcb.at(i)+(.5*leff.at(i))*cosphi;
            ycbm=ycb.at(i)-(.5*leff.at(i))*sinphi;
            xcbm=xcb.at(i)-(.5*leff.at(i))*cosphi;

            
            if(cosphi>=0.f){
                //! Note the current upstream and downstream limits for the wake non-local mixing
                //! are 3*lr in the downstream direction and lfx upstream in the x direction
                //! and lfy upstream in the y direction
                xcd=xcb.at(i)+(.5*leff.at(i)+.1*dx)*cosphi; // ! get the first point on the center line outside of the building (downstream)
                ycd=ycb.at(i)+(.5*leff.at(i)+.1*dx)*sinphi;// !
                
                //!mdw 7-10-2006 made changes to xcd, ycd,xcu, & ycu - formerly used .5 dx
                if(bldtype.at(i)==3){
                    xcu=xcb.at(i)-(.4*leff.at(i)+dx)*cosphi;// ! (upstream)
                    ycu=ycb.at(i)-(.4*leff.at(i)+dx)*sinphi; //!
                }
                else{
                    xcu=xcb.at(i)-(.5*leff.at(i)+0.1*dx)*cosphi;// ! (upstream)
                    ycu=ycb.at(i)-(.5*leff.at(i)+0.1*dx)*sinphi;// !
                }
                 
                //!mdw 7-05-2006 made changes to xcul & ycul - formerly used .5 dx
                xcul=xcu-(lfr.at(i)+dx)*cosphi;// ! get upper limit of the eddie
                ycul=ycu-(lfr.at(i)+dy)*sinphi;
                xcul=std::max(xcul,0.f);
                xcul=std::min(xcul,dx*(nxdx-1));
                ycul=std::max(ycul,0.f);
                ycul=std::min(ycul,dy*(nydy-1));
                cosfac=1.;
            }
            else{
                //!mdw 7-10-2006 made changes to xcd, ycd,xcu, & ycu - formerly used .5 dx
                xcd=xcb.at(i)+(.5*leff.at(i)+.1*dx)*cosphi;
                ycd=ycb.at(i)+(.5*leff.at(i)+.1*dx)*sinphi;
                if(bldtype.at(i)==3){
                    xcu=xcb.at(i)-(.4*leff.at(i)+dx)*cosphi;// ! (upstream)
                    ycu=ycb.at(i)-(.4*leff.at(i)+dx)*sinphi;// !
                }
                else{  
                    xcu=xcb.at(i)-(.5*leff.at(i)+0.1*dx)*cosphi;// ! (upstream)
                    ycu=ycb.at(i)-(.5*leff.at(i)+0.1*dx)*sinphi;// !
                }
                //!mdw 7-05-2006 made changes to xcul & ycul - formerly used .5 dx
                xcul=xcu-(lfr.at(i)+dx)*cosphi;// ! get upstream limit on the front cavity
                ycul=ycu-(lfr.at(i)+dy)*sinphi;// !
                xcul=std::max(xcul,0.f);
                xcul=std::min(xcul,dx*(nxdx-1));
                ycul=std::max(ycul,0.f);
                ycul=std::min(ycul,dy*(nydy-1));
                cosfac=-1.;
            }
        }
        else{// ! if you are more aligned with y than x
            //! MAN 9/15/2005 use weff and leff appropriately
            ycbp3=ycb.at(i)+(.5*weff.at(i)+.33*weff.at(i))*sinphit;// ! get the effective length of the building
            xcbp3=xcb.at(i)+(.5*weff.at(i)+.33*weff.at(i))*cosphit;
            ycbm3=ycb.at(i)-(.5*weff.at(i)+.33*weff.at(i))*sinphit;
            xcbm3=xcb.at(i)-(.5*weff.at(i)+.33*weff.at(i))*cosphit;
            //! end MAN 9/15/2005
            temp=(xcbp3-dx)/dx;
            icbp3=nint(temp);
            temp=(xcbm3-dx)/dx;
            icbm3=nint(temp);
            temp=(ycbp3-dy)/dy;
            jcbp3=nint(temp);
            temp=(ycbm3-dy)/dy;
            jcbm3=nint(temp);
            jcbp3=std::min(jcbp3,nydy-1);
            jcbm3=std::min(jcbm3,nydy-1);
            icbp3=std::min(icbp3,nxdx-1);
            icbm3=std::min(icbm3,nxdx-1);
            jcbp3=std::max(0,jcbp3);//altered from 1 to zero to comply GPU
            jcbm3=std::max(0,jcbm3);
            icbp3=std::max(0,icbp3);
            icbm3=std::max(0,icbm3);
            //! make sure you are outside of the building !
            int id=kmid*nxdx*nydy + jcbp3*nxdx + icbp3;
            int ip1=0;
            int ip2=0;
            int isign=0;
            
            if(cellQuic[id].c== 0){
                if(cosphit>0){
                    ip1=icbp3;
                    ip2=ny-1;
                    isign=1;
                }
                else{
                    ip1=icbp3;
                    ip2=0;//altered for GPU
                    isign=-1;
                }
                // ! decide which is closest building/floor
				
                for(int ip=ip1;ip<=ip2;ip=ip+isign){//do ip=ip1,ip2,isign 
                    icbp3=icbp3+isign;
                    dxcbp3=dx*(icbp3-1)-xcbp3;
                    xcbp3=dx*((icbp3-1));
                    ycbp3=ycbp3+dxcbp3*sinphit/cosphit;
                    temp=(ycbp3-dy)/dy;
                    jcbp3=nint(temp);
                    jcbp3=std::min(jcbp3,nydy-1);
                    jcbm3=std::min(jcbm3,nydy-1);
                    icbp3=std::min(icbp3,nxdx-1);
                    icbm3=std::min(icbm3,nxdx-1);
                    jcbp3=std::max(0,jcbp3);//altered for GPU
                    jcbm3=std::max(0,jcbm3);//altered for GPU
                    icbp3=std::max(0,icbp3);//altered for GPU
                    icbm3=std::max(0,icbm3);//altered for GPU
                    int idMid=kmid*nxdx*nydy + jcbp3*nxdx + icbp3;
                    if(cellQuic[idMid].c!= 0) break;
                }
            }
            int id2=kmid*nxdx*nydy +jcbm3*nxdx + icbm3;
            
            if(cellQuic[id2].c == 0){
                int im1=0;
                int im2=0;
                isign=0;
                if(cosphit>0.f){
                    im1=icbm3;
                    im2=0;//altered for GPU
                    isign=1;
                }
                else{
                    im1=icbm3;
                    im2=nx-icbm3+1;
                    isign=-1;
                }
                for(int im=im1;im<=im2;im=im+isign){//do im=im1,im2,-isign 
                    icbm3=icbm3-isign;
                    dxcbm3=dx*((icbm3-1))-xcbm3;
                    xcbm3=dx*((icbm3-1));
                    jcbm3=jcbm3+dxcbm3*sinphit/cosphit;
                    jcbp3=std::min(jcbp3,ny-1);
                    jcbm3=std::min(jcbm3,ny-1);
                    icbp3=std::min(icbp3,nx-1);
                    icbm3=std::min(icbm3,nx-1);
                    jcbp3=std::max(0,jcbp3);
                    jcbm3=std::max(0,jcbm3);
                    icbp3=std::max(0,icbp3);
                    icbm3=std::max(0,icbm3);
                    int idMid2=kmid*nxdx*nydy + jcbm3*nxdx +icbm3;
                    if(cellQuic[idMid2].c != 0) break;
                }
            }
            ycbp=ycb.at(i)+(.5*leff.at(i))*sinphit;// !  get back of the building
            xcbp=xcb.at(i)+(.5*leff.at(i))*cosphit;// !
            ycbm=ycb.at(i)-(.5*leff.at(i))*sinphit;// !  get front of the building
            xcbm=xcb.at(i)-(.5*leff.at(i))*cosphit;// !
            if(sinphi>=0.f){
                //! Note the current upstream and downstream limits for the wake non-local mixing
                //    ! are 3*lr in the downstream direction and lfx upstream in the x direction
                //  ! and lfy upstream in the y direction
                //! MAN 9/15/2005 use weff and leff appropriately
                //!mdw 7-05-2006 made changes to xcu,ycu, xcd & ycd - formerly used .5 dy or .5 dx
                xcd=xcb.at(i)+(.5*leff.at(i)+dy)*cosphi;// ! get the first point on the center line outside of the building (downstream)
                ycd=ycb.at(i)+(.5*leff.at(i)+dy)*sinphi;// !
                if(bldtype.at(i)==3){
                    xcu=xcb.at(i)-(.4*leff.at(i)+dx)*cosphi;// ! (upstream)
                    ycu=ycb.at(i)-(.4*leff.at(i)+dx)*sinphi; //!
                }
                else{
                    xcu=xcb.at(i)-(.5*leff.at(i)+0.1*dx)*cosphi;// ! (upstream) 
                    ycu=ycb.at(i)-(.5*leff.at(i)+0.1*dx)*sinphi;// !
                }
                //! end MAN 9/15/2005
                //! mdw 7-05-2006 eliminated .5 dx  or .5 dy in favor of dx & dy
                xcul=xcu-(lfr.at(i)+dx)*cosphi;// ! get upper limit of the eddie
                ycul=ycu-(lfr.at(i)+dy)*sinphi;
                xcul=std::max(xcul,0.f);
                xcul=std::min(xcul,dx*(nxdx-1));
                ycul=std::max(ycul,0.f);
                ycul=std::min(ycul,dy*(nydy-1));
                cosfac=1.f;
            }
            else{
                   //! MAN 9/15/2005 use weff and leff appropriately
                xcd=xcb.at(i)+(.5*leff.at(i)+dy)*cosphi;
                ycd=ycb.at(i)+(.5*leff.at(i)+dy)*sinphi;
                if(bldtype.at(i)==3){
                    xcu=xcb.at(i)-(.4*leff.at(i)+dx)*cosphi;// ! (upstream)
                    ycu=ycb.at(i)-(.4*leff.at(i)+dx)*sinphi;// !
                }
                else{
                    xcu=xcb.at(i)-(.5*leff.at(i)+dx)*cosphi;// ! (upstream) 
                    ycu=ycb.at(i)-(.5*leff.at(i)+dx)*sinphi;// !
                }
                //! end MAN 9/15/2005
                
                xcul=xcu+(lfr.at(i)+dx)*cosphi;// ! get upstream limit on the front cavity
                ycul=ycu+(lfr.at(i)+dy)*sinphi;// !
                xcul=std::max(xcul,0.f);
                xcul=std::min(xcul,dx*(nxdx-1));
                ycul=std::max(ycul,0.f);
                ycul=std::min(ycul,dy*(nydy-1));
                cosfac=-1.f;
            }
        }
        //!mdw 7-05-2006 change form to ixxx or jxxx =nint()+1
        temp=(xcd-dx)/dx;//altered to comply with GPU
        int icd=nint(temp)+1;// ! get indicies for the downstream center line to back of the building
        temp=(ycd-dy)/dy;//altered to comply with GPU
        int jcd=nint(temp)+1;
        //!mdw 4-16-2004 added correction for ktop+3 > nz-1
        int ktp=std::min(ktop,nzdz-1);//didn't alter here as ktop is already aligned with GPU coordinates
        float zk=0.f;
        float zbrac=0.f;
        float zkfac=0.f;
        float xcdl=0.f;
        float ycdl=0.f;
        int icdl=0;
        int jcdl=0;
        int icu=0;
        int jcu=0;
        int icul=0;
        int jcul=0;
        float urefz=0.f;
        float ds=0.f;
        float sdown=0.f;
        float sup=0.f;
        float stin=0.f;
        float istinf=0.f;
        float st=0.f;
        int istf=0;
        int isf=0;
        int isfu=0;
        float utotp=0.f;
        float utotm=0.f;
        float cosu=0.f;
        float sinv=0.f;
        int isini=0;
        float cosl=0.f;
        float sinl=0.f;
        float delutz=0.f;
        float upvpg=0.f;
        float upwpg=0.f;
        float upsqg=0.f;
        float vpsqg=0.f;
        float vpwpg=0.f;
        float wpsqg=0.f;
        float duy=0.f;
        
        for(int k=ktp;k>=0;k--){//do k=ktp,2,-1  ! Account for wake difference in the cavity
            
            zk=zm.at(k);
            if(zi.at(k)<.99*h){
                zbrac=pow( (1.f-zi.at(k)/h) , 1.5f);
            }
            else{
                zbrac=pow( (1.f-.99f),1.5f);
            }
            //zbrac=pow( (1.f-zi.at(k)/h) , 1.5f);
            //!mdw 4-16-2004 added correction for ktop+3 > nz-1
            int idupc=k*nxdx*nydy + jupc*nxdx +iupc;
            int idupcktop=(ktop+3)*nxdx*nydy + jupc*nxdx +iupc;
            int idupcnzm1=(nzdz-1)*nxdx*nydy + jupc*nxdx +iupc;
            if(ktop+3<=nzdz-1){
                zcorf.at(k)=sqrt(wind_vel[idupc].u*wind_vel[idupc].u + wind_vel[idupc].v*wind_vel[idupc].v + wind_vel[idupc].w*wind_vel[idupc].w)/
                    sqrt(wind_vel[idupcktop].u*wind_vel[idupcktop].u + wind_vel[idupcktop].v*wind_vel[idupcktop].v
                         + wind_vel[idupcktop].w*wind_vel[idupcktop].w);
            }
            else{
                zcorf.at(k)=sqrt(wind_vel[idupc].u*wind_vel[idupc].u + wind_vel[idupc].v*wind_vel[idupc].v + wind_vel[idupc].w*wind_vel[idupc].w)/
                    sqrt(wind_vel[idupcnzm1].u*wind_vel[idupcnzm1].u + wind_vel[idupcnzm1].v*wind_vel[idupcnzm1].v
                         + wind_vel[idupcnzm1].w*wind_vel[idupcnzm1].w);
            }
                        
            
            //! mdw 4-16-2004 added proper treatment of zfo
            if(zk<ht[i]+zfo[i]){
                zkfac=sqrt(1.-pow((zk/(ht[i]+zfo[i])),2));
            }
            else{
                if(k==ktp){
                    zkfac=1.;
                }
                else{
                    zkfac=0.f;
                }
                
            }
            //! mdw 7-05-2006 changed from .5 dx or .5 dy to dx & dy to be consistent with nint
            xcdl=xcd+(3.*lr.at(i)+dx)*zkfac*cosphi;// ! calculate the x,y limit of the wake as a function of height
            ycdl=ycd+(3.*lr.at(i)+dy)*zkfac*sinphi;// !
            xcdl=std::min(xcdl,dx*(nxdx));
            ycdl=std::min(ycdl,dy*(nydy));
            xcdl=std::max(xcdl,0.f);
            ycdl=std::max(ycdl,0.f);
            
            temp=(xcdl-dx)/dx;//altered for GPU, substracted dx, same below
            icdl=nint(temp)+1;// ! Calculate the indicies for i,j according to xcdl,ycdl
            temp=(ycdl-dy)/dy;
            jcdl=nint(temp)+1;
            temp=(xcu-dx)/dx;
            icu=nint(temp)+1;//   ! indicies for the upstream cavity (building)
            temp=(ycu-dy)/dy;
            jcu=nint(temp)+1;//   !
            temp=(xcul-dx)/dx;
            icul=nint(temp)+1;// ! (furthest upstream)
            temp=(ycul-dy)/dy;
            jcul=nint(temp)+1;// !!!
            //!mdw 4-16-2004 added correction for ktop+3 > nz-1
            int idktop=(ktop+3)*nxdx*nydy + jcb.at(i)*nxdx +icb.at(i);
            if(ktop+3<=nzdz-1){
                //! calculating the reference wind un-disturbed by the building
                urefz=sqrt(wind_vel[idktop].u*wind_vel[idktop].u + wind_vel[idktop].v*wind_vel[idktop].v + wind_vel[idktop].w*wind_vel[idktop].w);
            }
            else{
                urefz=sqrt(pow(ualoft,2.f)+pow(valoft,2.f));
            }
            ds=0.7*std::min(dx,dy);// ! pick a step that is small enough to not skip grid cells
            sdown=sqrt((xcdl-xcd)*(xcdl-xcd)+(ycdl-ycd)*(ycdl-ycd))+2.*ds;// ! calculate the limits for the distance measured along the centerline (rear)
            sup=sqrt((xcul-xcu)*(xcul-xcu)+(ycul-ycu)*(ycul-ycu))+2.*ds;//   ! same for the front eddy
            stin=.5*leff.at(i);//
            temp=stin/ds;
            istinf=nint(temp)+1.f;
            //!mdw 7-11-2006 changed istinf to allow replacement to center of bldg
            //!mdw 5-14-2004 corrected expression for st; older versions gave errors for wide blds
            st=sqrt((xcbp3-xcb.at(i))*(xcbp3-xcb.at(i))+(ycbp3-ycb.at(i))*(ycbp3-ycb.at(i)))+1.*ds;// ! total distance to point
			temp=(st+.333*leff.at(i))/ds;
            istf=nint(temp)+1.f;//   ! (transverse direction) 
            //!mdw 6-9-2004 extended the transverse integration to st+.333*leff
            temp=sdown/ds;
            isf=nint(temp)+1;// ! setup limits of calculations (for do loops) (along cneterline down)  
            temp=sup/ds;
            isfu=nint(temp)+1;//  ! (along centerline up) 
            if(lfr.at(i) < 0.f)isfu=0;
            
            //!mdw 4-16-2004 added correction for ktop+3 > nz-1
            
            //! Select the largest reference wind of the plus or minus side of the building
            int id1=k*nxdx*nydy + jcbp3*nxdx +icbp3;
            int id2=k*nxdx*nydy + jcbm3*nxdx +icbm3;

            utotp=sqrt(wind_vel[id1].u*wind_vel[id1].u + wind_vel[id1].v*wind_vel[id1].v + wind_vel[id1].w*wind_vel[id1].w);
            utotm=sqrt(wind_vel[id2].u*wind_vel[id2].u + wind_vel[id2].v*wind_vel[id2].v + wind_vel[id2].w*wind_vel[id2].w);
            int ik=k*nxdx*nydy + i;
            int idp=k*nxdx*nydy + jcbp3*nxdx +icbp3;
            int idm=k*nxdx*nydy + jcbm3*nxdx +icbm3;
            if(utotp>=utotm){
                uref.at(ik)=utotp+.000001;
                urefu.at(ik)=uref.at(ik)*cos(phib.at(i));
                urefv.at(ik)=uref.at(ik)*sin(phib.at(i));
                urefw.at(ik)=wind_vel[idp].w;
            }
            else{
                uref.at(ik)=utotm+.000001;
                urefu.at(ik)=uref.at(ik)*cos(phib.at(i));
                urefv.at(ik)=uref.at(ik)*sin(phib.at(i));
                urefw.at(ik)=wind_vel[idm].w;
            }
            //!!!!!!!
            cosu=(urefu.at(ik)+.000001)/uref.at(ik);
            sinv=urefv.at(ik)/uref.at(ik);
            //! downstream wake  along axis do loop for delta u
            isini=1;
            float xcell=0.f;
            float ycell=0.f;
            int icel=0;
            int jcel=0;
            float utot=0.f;
            for(int is=1;is<=isf;is++){//   do is=1,isf 
                xcell=xcd+ds*(is-1)*cosphi;
                ycell=ycd+ds*(is-1)*sinphi;
                temp=(xcell-dx)/dx;//substracted dx for GPU
                icel=nint(temp)+1;
                temp=(ycell-dy)/dy;//substracted dy for GPU
                jcel=nint(temp)+1;
                icel=std::min(nxdx-1,icel);
                icel=std::max(1,icel);//altered for GPU (2 to 1)
                jcel=std::min(nydy-1,jcel);
                jcel=std::max(1,jcel);//altered for GPU (2 to 1)
                int id=k*nxdx*nydy + jcel*nxdx +icel;
                if(cellQuic[id].c == 0 && is==1){
                    isini=2;
                }
                utot=sqrt(wind_vel[id].u*wind_vel[id].u + wind_vel[id].v*wind_vel[id].v + wind_vel[id].w*wind_vel[id].w);
                //!mdw 4-16-2004 added correction for ktop+3 > nz-1
                int iceljcel=jcel*nxdx +icel;
                int idcel=ktop*nxdx*nydy + jcel*nxdx +icel;
                
                if(k==ktp){
                    if(ktop<=nzdz-1){
                        utotktp.at(iceljcel)=utot;
                        uktop.at(iceljcel)=wind_vel[idcel].u;
                        vktop.at(iceljcel)=wind_vel[idcel].v;
                        wktop.at(iceljcel)=wind_vel[idcel].w;
                    }
                    else{
                        utotktp.at(iceljcel)=sqrt(ualoft*ualoft+valoft*valoft); //check:: compare with QP, may be a bug in QP
                        uktop.at(iceljcel)=ualoft;
                        vktop.at(iceljcel)=valoft;
                        wktop.at(iceljcel)=0.f;
                    }
                }
                //! this sets reference for vertical transfer
                utot=utot+.000001;
                int idcelk=k*nxdx*nydy +jcel*nxdx +icel;
                int ik=k*nxdx*nydy+i;
                cosl=wind_vel[idcelk].u/utot;
                sinl=wind_vel[idcelk].v/utot;
                if(cellQuic[idcelk].c > 0){
                    delutz=sqrt( pow( (wind_vel[idcelk].u-zcorf.at(k)*uktop.at(iceljcel)),2.f)
                                 +pow( (wind_vel[idcelk].v -zcorf.at(k)*vktop.at(iceljcel)),2.f)
                                 +pow( (wind_vel[idcelk].w -zcorf.at(k)*wktop.at(iceljcel)),2.f) );
                    deluc.at(ik)=sqrt( pow( (urefu.at(ik)-wind_vel[idcelk].u),2.f)
                                       +pow( (urefv.at(ik)-wind_vel[idcelk].v),2.f)
                                       +pow( (urefw.at(ik)-wind_vel[idcelk].w),2.f));
                    //!mdw 4-16-2004 added correction for ktop+3 > nz-1
                    if(k!=ktp){
                        //! Selects the largest gradient (vert or horiz transfer)
                        //! mdw 4-16-2004 added proper treatment of zfo
                        
                        if((2.*deluc.at(ik)/weff.at(i))<(utotktp.at(iceljcel)/(ht[i]+zfo[i])) &&
                           delutz>.2*zcorf.at(k)*utotktp.at(iceljcel)){// ! vertical dominates
                            ustargz.at(idcelk)=std::max(knlc*utotktp.at(iceljcel),ustargz.at(idcelk)); 
                            if(fabs(ustargz.at(idcelk)-knlc*utotktp.at(iceljcel))<1.e-05*ustargz.at(idcelk)){//!This value dominates over prev. buildings.
                                elzg.at(idcelk)=ht[i]+zfo[i];
                                upvpg=0.f;
                                upwpg=-ctau13*zbrac*ustargz.at(idcelk)*ustargz.at(idcelk);
                                upsqg=cusq*zbrac*ustargz.at(idcelk)*ustargz.at(idcelk);
                                vpsqg=cvsq*zbrac*ustargz.at(idcelk)*ustargz.at(idcelk);
                                vpwpg=0.f;
                                wpsqg=cwsq*zbrac*ustargz.at(idcelk)*ustargz.at(idcelk);
                                ustarg.at(idcelk)=ustargz.at(idcelk);
                                rotate2d(idcelk,cosphi,sinphi,upsqg,upvpg,vpsqg,wpsqg,upwpg,vpwpg);
                            }
                        }
                        else{
                            //! We use the vertical gradient as dominant if it is sharper than the horizontal
                            duy=-deluc.at(ik)*sinl*cosu+deluc.at(ik)*sinv*cosl;
                            //! we now have the delta u between the outside of the bldg and the center of the wake
                            //! mdw 6-10-2004 removed
                            if(deluc.at(ik)>.2*uref.at(ik)){
                                ustarg.at(idcelk)=std::max(ustarg.at(idcelk),knlc*deluc.at(ik));
                                if(fabs(ustarg.at(idcelk)-knlc*deluc.at(ik))<1.e-05*ustarg.at(idcelk)){// ! if the horiz is dominant calculate sigmas
                                    upvpg=0.f;
                                    //! on axis u prime v prime is zero
                                    upwpg=0.f;
                                    //! for eddy transport in uv we dont consider uw
                                    upsqg=cusq*zbrac*ustarg.at(idcelk)*ustarg.at(idcelk);
                                    wpsqg=cvsq*zbrac*ustarg.at(idcelk)*ustarg.at(idcelk);
                                    vpwpg=0.f;
                                    elzg.at(idcelk)=0.5*weff.at(i);
                                    vpsqg=cwsq*zbrac*ustarg.at(idcelk)*ustarg.at(idcelk);
                                    rotate2d(idcelk,cosphi,sinphi,upsqg,upvpg,vpsqg,wpsqg,upwpg,vpwpg);
                                }
                            }
                        }
                    }
                }
                else{
                    deluc.at(ik)=0.;
                    delutz=0.;
                }
                //! transverse do loop in downstream wake
                
                for(int ist=2;ist<=istf;ist++){//do ist=2,istf 
                    //! first direction in the transverse of the wake
                    xcelt=xcell+ds*(ist-1.f)*cosphit;
                    ycelt=ycell+ds*(ist-1.f)*sinphit;
                    temp=(xcelt-dx)/dx;
                    icelt=nint(temp)+1;
                    temp=(ycelt-dy)/dy;
                    jcelt=nint(temp)+1;
                    if(fabs(xcelt-xcell)<.5f*ds)icelt=icel;
                    if(fabs(ycelt-ycell)<.5f*ds)jcelt=jcel;
                    icelt=std::min(nxdx-1,icelt);
                    icelt=std::max(1,icelt);
                    jcelt=std::min(nydy-1,jcelt);
                    jcelt=std::max(1,jcelt);
                    int iceltjcelt=jcelt*nxdx + icelt;
                    int idceltk= k*nxdx*nydy + jcelt*nxdx +icelt;
                    if(cellQuic[idceltk].c > 0){
                        utott=sqrt(wind_vel[idceltk].u*wind_vel[idceltk].u + wind_vel[idceltk].v*wind_vel[idceltk].v
                                   + wind_vel[idceltk].w*wind_vel[idceltk].w);
                        utott=utott+.000001f;
                        //!mdw 4-16-2004 added correction for ktop+3 > nz-1

                        int idceltktop=ktop*nxdx*nydy + jcelt*nxdx +icelt;
                        if(k==ktp){
                            if(ktop<nzdz-1){
                                utotktp.at(iceltjcelt)=utott;
                                uktop.at(iceltjcelt)=wind_vel[idceltktop].u;
                                vktop.at(iceltjcelt)=wind_vel[idceltktop].v;
                                wktop.at(iceltjcelt)=wind_vel[idceltktop].w;
                            }
                            else{
                                utotktp.at(iceltjcelt)=sqrt(ualoft*ualoft+valoft*valoft);
                                uktop.at(iceltjcelt)=ualoft;
                                vktop.at(iceltjcelt)=valoft;  
                                wktop.at(iceltjcelt)=0.;
                            }
                        }
                        int ik=k*nxdx*nydy +i;
                        delut=sqrt(pow( (urefu.at(ik)-wind_vel[idceltk].u),2.f)+
                                   pow( (urefv.at(ik)-wind_vel[idceltk].v),2.f)+
                                   pow( (urefw.at(ik)-wind_vel[idceltk].w),2.f));
                        delutz=sqrt(pow( (wind_vel[idceltk].u-zcorf.at(k)*uktop.at(iceltjcelt)),2.f)
                                    +pow( (wind_vel[idceltk].v-zcorf.at(k)*vktop.at(iceltjcelt)),2.f)
                                    +pow( (wind_vel[idceltk].w-zcorf.at(k)*wktop.at(iceltjcelt)),2.f));
                        //!mdw 4-16-2004 added correction for ktop+3 > nz-1
                        if(k!=ktp){
                            //! mdw 4-16-2004 added proper treatment of zfo
                            //! mdw 6-10-2004 changed to make check on centerline rather than local value
                            int ik=k*nxdx*nydy +i;
                            if((2.*deluc.at(ik)/weff.at(i))<(utotktp.at(iceltjcelt)/(ht[i]+zfo[i])) 
                               && delutz>.2*zcorf.at(k)*utotktp.at(iceltjcelt)){
                                if(ustargz.at(idceltk)<knlc*utotktp.at(iceltjcelt)){
                                    ustargz.at(idceltk)=knlc*utotktp.at(iceltjcelt);
                                    elzg.at(idceltk)=ht[i]+zfo[i];
                                    upvpg=0.;
                                    upwpg=-ctau13*zbrac*ustargz.at(idceltk)*ustargz.at(idceltk);
                                    upsqg=cusq*zbrac*ustargz.at(idceltk)*ustargz.at(idceltk);
                                    vpsqg=cvsq*zbrac*ustargz.at(idceltk)*ustargz.at(idceltk);
                                    vpwpg=0.;
                                    wpsqg=cwsq*zbrac*ustargz.at(idceltk)*ustargz.at(idceltk);
                                    ustarg.at(idceltk)=ustargz.at(idceltk);
                                    rotate2d(idceltk,cosphi,sinphi,upsqg,upvpg,vpsqg,wpsqg,upwpg,vpwpg);
                                }
                            }
                            else{
                                // We use the vertical gradient as dominant if it is sharper than the horizontal
                                cosl=wind_vel[idceltk].u/utott;
                                sinl=wind_vel[idceltk].v/utott;
                                duy=-deluc.at(ik)*sinl*cosu+deluc.at(ik)*sinv*cosl;
                                // mdw 6-10-2004 changed check from delut (local value) to deluc.at(ik); centerline
                                if(delut>.2*uref.at(ik)){
                                    if(ustarg.at(idceltk)<knlc*deluc.at(ik)){
                                        ustarg.at(idceltk)=knlc*deluc.at(ik);
                                        upvpg=-((ist-1.f)/(istf-1.f))*ustarg.at(idceltk)*ustarg.at(idceltk);
                                        upwpg=0.;
                                        // for eddy transport in uv we dont consider uw
                                        upsqg=cusq*zbrac*ustarg.at(idceltk)*ustarg.at(idceltk);
                                        wpsqg=cvsq*zbrac*ustarg.at(idceltk)*ustarg.at(idceltk);
                                        vpwpg=0.;
                                        vpsqg=cwsq*zbrac*ustarg.at(idceltk)*ustarg.at(idceltk);
                                        elzg.at(idceltk)=.5*weff.at(i);
                                        rotate2d(idceltk,cosphi,sinphi,upsqg,upvpg,vpsqg,wpsqg,upwpg,vpwpg);
                                    }
                                }
                            }
                            if(is==isini){
                                for(int isin=isini+1;isin<=istinf;isin++){//do isin=isini+1,istinf
                                    xceln=xcelt-ds*(isin-1)*cosphi;
                                    yceln=ycelt-ds*(isin-1)*sinphi;
                                    temp=(xceln-dx)/dx;
                                    iceln=nint(temp)+1;
                                    temp=(yceln-dy)/dy;
                                    jceln=nint(temp)+1;
                                    iceln=std::min(nxdx-1,iceln);
                                    iceln=std::max(1,iceln);
                                    jceln=std::min(nydy-1,jceln);
                                    jceln=std::max(1,jceln);
                                    // mdw 3/22/2004PM added if statement to avoid replacing non-zero ustarg stuff
                                    // with zero values
                                    int idcelnk= k*nxdx*nydy + jceln*nxdx +iceln;
                                    if(ustarg.at(idceltk)>ustarg.at(idcelnk)){
                                        ustarg.at(idcelnk)=ustarg.at(idceltk);
                                        elzg.at(idcelnk)=elzg.at(idceltk);
                                        ufsqgi.at(idcelnk)=ufsqgi.at(idceltk);
                                        vfsqgi.at(idcelnk)=vfsqgi.at(idceltk);
                                        wfsqgi.at(idcelnk)=wfsqgi.at(idceltk);
                                        ufvfgi.at(idcelnk)=ufvfgi.at(idceltk);
                                        ufwfgi.at(idcelnk)=ufwfgi.at(idceltk);
                                        vfwfgi.at(idcelnk)=vfwfgi.at(idceltk);
                                    }
                                    // mdw 3/22/2004PM new endif for new if then
                                }//enddo
                            }
                        }
                    }
                    // opposite direction in the transverse of the wake
                    xcelt=xcell-ds*(ist-1.f)*cosphit;
                    ycelt=ycell-ds*(ist-1.f)*sinphit;
                    temp=(xcelt-dx)/dx;
                    icelt=nint(temp)+1; 
                    temp=(ycelt-dy)/dy;
                    jcelt=nint(temp)+1; 
                    if(fabs(xcelt-xcell)<.5*ds)icelt=icel;
                    if(fabs(ycelt-ycell)<.5*ds)jcelt=jcel;
                    icelt=std::min(nxdx-1,icelt); 
                    icelt=std::max(1,icelt);
                    jcelt=std::min(nydy-1,jcelt);
                    jcelt=std::max(1,jcelt);
                    
                    iceltjcelt=jcelt*nxdx + icelt;
                    int idceltktop=ktop*nxdx*nydy + jcelt*nxdx +icelt;
                    idceltk=k*nxdx*nydy + jcelt*nxdx +icelt;
                    if(cellQuic[idceltk].c > 0){
                        utott=sqrt(wind_vel[idceltk].u*wind_vel[idceltk].u+wind_vel[idceltk].v*wind_vel[idceltk].v
                                   +wind_vel[idceltk].w*wind_vel[idceltk].w);
                        utott=utott+.000001;
                        delut=sqrt(pow( (urefu.at(ik)-wind_vel[idceltk].u),2)
                                   + pow( (urefv.at(ik)-wind_vel[idceltk].v),2)
                                   + pow( (urefw.at(ik)-wind_vel[idceltk].w),2));
                        // mdw 4-16-2004 added correction for ktop+3 > nz-1
                        
                        if(k==ktp){
                            if(ktop<=nzdz-1){
                                utotktp.at(iceltjcelt)=utott;
                                uktop.at(iceltjcelt)=wind_vel[idceltktop].u;
                                vktop.at(iceltjcelt)=wind_vel[idceltktop].v;
                                wktop.at(iceltjcelt)=wind_vel[idceltktop].w;
                            }
                            else{
                                utotktp.at(iceltjcelt)=sqrt(ualoft*ualoft+valoft*valoft);
                                uktop.at(iceltjcelt)=ualoft;
                                vktop.at(iceltjcelt)=valoft;
                                wktop.at(iceltjcelt)=0.;
                            }
                        }
                        // mdw 4-16-2004 added correction for ktop+3 > nz-1
                        
                        if(k!=ktp){
                            delutz=sqrt(pow( (wind_vel[idceltk].u-zcorf.at(k)*uktop.at(iceltjcelt)),2)
                                        +pow( (wind_vel[idceltk].v-zcorf.at(k)*vktop.at(iceltjcelt)),2)
                                        +pow( (wind_vel[idceltk].w-zcorf.at(k)*wktop.at(iceltjcelt)),2));
                            // mdw 4-16-2004 added proper treatment of zfo
                            // mdw 6-10-2004 made check on centerline rather than local value
                            if((2.*deluc.at(ik)/weff.at(i))<(utotktp.at(iceltjcelt)/(ht[i]+zfo[i])) 
                               && delutz>.2*zcorf.at(k)*utotktp.at(iceltjcelt)){
                                if(ustargz.at(idceltk)<knlc*utotktp.at(iceltjcelt)){
                                    ustargz.at(idceltk)=knlc*utotktp.at(iceltjcelt);
                                    elzg.at(idceltk)=ht[i]+zfo[i];
                                    upvpg=0.;
                                    upwpg=-ctau13*zbrac*ustargz.at(idceltk)*ustargz.at(idceltk);
                                    upsqg=cusq*zbrac*ustargz.at(idceltk)*ustargz.at(idceltk);
                                    vpsqg=cvsq*zbrac*ustargz.at(idceltk)*ustargz.at(idceltk);
                                    vpwpg=0.;
                                    wpsqg=cwsq*zbrac*ustargz.at(idceltk)*ustargz.at(idceltk);
                                    ustarg.at(idceltk)=ustargz.at(idceltk);
                                    rotate2d(idceltk,cosphi,sinphi,upsqg,upvpg,vpsqg,wpsqg,upwpg,vpwpg);
                                }
                            }
                            else{
                                // We use the vertical gradient as dominant if it is sharper than the horizontal
                                cosl=wind_vel[idceltk].u/utott;
                                sinl=wind_vel[idceltk].v/utott;
                                duy=-deluc.at(ik)*sinl*cosu+deluc.at(ik)*sinv*cosl; 
                                // mdw 6-10-2004 made check on centerline value rather than local value
                                
                                if(delut>.2f*uref.at(ik)){
                                    if(ustarg.at(idceltk)<knlc*deluc.at(ik)){
                                        ustarg.at(idceltk)=knlc*deluc.at(ik);
                                        upvpg=((ist-1.f)/(istf-1.f))*ustarg.at(idceltk)*ustarg.at(idceltk);
                                        upwpg=0.f;
                                        // for eddy transport in uv we dont consider uw
                                        upvpg=ctau13*zbrac*((ist-1.f)/(istf-1.f))*ustarg.at(idceltk)*ustarg.at(idceltk);
                                        upwpg=0.f;
                                        // for eddy transport in uv we dont consider uw
                                        upsqg=cusq*zbrac*ustarg.at(idceltk)*ustarg.at(idceltk);
                                        wpsqg=cvsq*zbrac*ustarg.at(idceltk)*ustarg.at(idceltk);
                                        elzg.at(idceltk)=0.5f*weff.at(i);
                                        vpwpg=0.f;
                                        vpsqg=cwsq*zbrac*ustarg.at(idceltk)*ustarg.at(idceltk);
                                        rotate2d(idceltk,cosphi,sinphi,upsqg,upvpg,vpsqg,wpsqg,upwpg,vpwpg);
                                    }
                                }
                            }
                            if(is==isini){
                                for(int isin=isini+1;isin<=istinf;isin++){
                                    xceln=xcelt-ds*(isin-1)*cosphi;
                                    yceln=ycelt-ds*(isin-1)*sinphi;
                                    temp=(xceln-dx)/dx;
                                    iceln=nint(temp)+1; 
                                    temp=(yceln-dy)/dy;
                                    jceln=nint(temp)+1; 
                                    iceln=std::min(nxdx-1,iceln);
                                    iceln=std::max(1,iceln);
                                    jceln=std::min(nydy-1,jceln);
                                    jceln=std::max(1,jceln);
                                    int idcelnk=k*nxdx*nydy +jceln*nxdx +iceln;
                                    // mdw 3/22/2004pm adding new if then structure to avoid replacing non-zero
                                    // ustarg with zero ones
                                    if(ustarg.at(idceltk)>ustarg.at(idcelnk)){
                                        ustarg.at(idcelnk)=ustarg.at(idceltk);
                                        elzg.at(idcelnk)=elzg.at(idceltk); 
                                        ufsqgi.at(idcelnk)=ufsqgi.at(idceltk);
                                        vfsqgi.at(idcelnk)=vfsqgi.at(idceltk);
                                        wfsqgi.at(idcelnk)=wfsqgi.at(idceltk);
                                        ufvfgi.at(idcelnk)=ufvfgi.at(idceltk);
                                        ufwfgi.at(idcelnk)=ufwfgi.at(idceltk);
                                        vfwfgi.at(idcelnk)=vfwfgi.at(idceltk);
                                    }
                                }//enddo
                                // mdw 3/22/2004 end of new if then structure
                            }//endif
                        }//endif
                    }
                }   //lp021
            }  //lp022

            isini=1;
            for(int is=1; is<=isfu;is++){//do is=1,isfu
                // upstream front eddy along the centerline
                xcell=xcu-ds*(is-1.f)*cosphi;
                ycell=ycu-ds*(is-1.f)*sinphi;
                //mdw 7-05-2006 changed form form =nint( / ) to nint( / )+1
                temp=(xcell-dx)/dx;
                icel=nint(temp)+1; 
                temp=(ycell-dy)/dy;
                jcel=nint(temp)+1; 
                icel=std::min(nxdx-1,icel);
                icel=std::max(1,icel);
                jcel=std::min(nydy-1,jcel);
                jcel=std::max(1,jcel);
                int idcelk=k*nxdx*nydy +jcel*nxdx +icel;
                int iceljcel=jcel*nxdx +icel;
                if(cellQuic[idcelk].c == 0 && is == 1){
                    isini=2;
                }
                int idcelktop=ktop*nxdx*nydy + jcel*nxdx +icel;
                idcelk=k*nxdx*nydy + jcel*nxdx +icel;
                utot=sqrt(wind_vel[idcelk].u*wind_vel[idcelk].u+wind_vel[idcelk].v*wind_vel[idcelk].v+wind_vel[idcelk].w*wind_vel[idcelk].w);
                // mdw 1-22-2004 new lines in support of bldg infiltration
                if((k==kmid)&&(is==1))utotcl1.at(i)=utot;
                if((k==kmid)&&(utot>utotmax.at(i)))utotmax.at(i)=utot; 
                utot=utot+.000001;
                //mdw 4-16-2004 added correction for ktop+3 > nz-1
                if(k==ktp){
                    if(ktop<=nzdz-1){
                        utotktp.at(iceljcel)=utot;
                        uktop.at(iceljcel)=wind_vel[idcelktop].u;
                        vktop.at(iceljcel)=wind_vel[idcelktop].v;
                        wktop.at(iceljcel)=wind_vel[idcelktop].w;
                    }
                    else{
                        utotktp.at(iceljcel)=sqrt(ualoft*ualoft+valoft*valoft);
                        uktop.at(iceljcel)=ualoft;
                        vktop.at(iceljcel)=valoft;
                        wktop.at(iceljcel)=0.;
                    }
                }
                deluc.at(ik)=sqrt(pow( (urefu.at(ik)-wind_vel[idcelk].u),2)+
                                  pow( (urefv.at(ik)-wind_vel[idcelk].v),2)+
                                  pow( (urefw.at(ik)-wind_vel[idcelk].w),2));
                //mdw 4-16-2004 added correction for ktop+3 > nz-1
                if(k!=ktp){
                    delutz=sqrt(pow( (wind_vel[idcelk].u-zcorf.at(k)*uktop.at(iceljcel)),2)+
                                pow( (wind_vel[idcelk].v-zcorf.at(k)*vktop.at(iceljcel)),2)+
                                pow( (wind_vel[idcelk].w-zcorf.at(k)*wktop.at(iceljcel)),2));
                    deluc.at(ik)=sqrt(pow( (urefu.at(ik)-wind_vel[idcelk].u),2)+
                                      pow( (urefv.at(ik)-wind_vel[idcelk].v),2)+
                                      pow( (urefw.at(ik)-wind_vel[idcelk].w),2));
                    // Selects the largest gradient (vert or horiz transfer)
                    // mdw 4-16-2004 added proper treatment of zfo
                    if((2.*deluc.at(ik)/weff.at(i))<(utotktp.at(iceljcel)/(ht[i]+zfo[i])) 
                       && delutz>.2*zcorf.at(k)*utotktp.at(iceljcel)){ // vertical dominates
                        if(ustargz.at(idcelk)<knlc*utotktp.at(iceljcel)){ // This value dominates over prev. buildings.
                            ustargz.at(idcelk)=knlc*utotktp.at(iceljcel);
                            elzg.at(idcelk)=ht[i]+zfo[i];
                            upvpg=0.;
                            upwpg=-ctau13*zbrac*ustargz.at(idcelk)*ustargz.at(idcelk);
                            upsqg=cusq*zbrac*ustargz.at(idcelk)*ustargz.at(idcelk);
                            vpsqg=cvsq*zbrac*ustargz.at(idcelk)*ustargz.at(idcelk);
                            vpwpg=0.;
                            wpsqg=cwsq*zbrac*ustargz.at(idcelk)*ustargz.at(idcelk); // align sigmas with the overall mean wind
                            ustarg.at(idcelk)=ustargz.at(idcelk);
                            rotate2d(idcelk,cosphi,sinphi,upsqg,upvpg,vpsqg,wpsqg,upwpg,vpwpg);
                        }
                    }
                    else{
                        // We use the vertical gradient as dominant if it is sharper than the horizontal
                        cosl=wind_vel[idcelk].u/utot;
                        sinl=wind_vel[idcelk].v/utot;
                        duy=-deluc.at(ik)*sinl*cosu+deluc.at(ik)*sinv*cosl;
                        // we now have the delta u between the outside of the bldg and the center of the wake
                        if(deluc.at(ik)>.2*uref.at(ik)){
                            if(ustarg.at(idcelk)<knlc*deluc.at(ik)){
                                ustarg.at(idcelk)=knlc*deluc.at(ik);
                                upvpg=0.;
                                // on axis u prime v prime is zero
                                upwpg=0.;
                                // for eddy transport in uv we dont consider uw
                                upsqg=cusq*zbrac*ustarg.at(idcelk)*ustarg.at(idcelk);
                                wpsqg=cvsq*zbrac*ustarg.at(idcelk)*ustarg.at(idcelk);
                                vpwpg=0.;
                                elzg.at(idcelk)=0.5*weff.at(i);
                                vpsqg=cwsq*zbrac*ustarg.at(idcelk)*ustarg.at(idcelk);
                                rotate2d(idcelk,cosphi,sinphi,upsqg,upvpg,vpsqg,wpsqg,upwpg,vpwpg);
                            }
                        }
                    }
                }
                for(int ist=2;ist<=istf;ist++){//do ist=2,istf
                    // first direction in the transverse of the front eddy
                    xcelt=xcell+ds*(ist-1.f)*cosphit;
                    ycelt=ycell+ds*(ist-1.f)*sinphit;
                    //mdw 7-05-2006 changed form from nint( / ) to nint( / )+1
                    temp=(xcelt-dx)/dx;
                    icelt=nint(temp)+1;
                    temp=(ycelt-dy)/dy;
                    jcelt=nint(temp)+1;
                    if(fabs(xcelt-xcell)<.5*ds)icelt=icel;
                    if(fabs(ycelt-ycell)<.5*ds)jcelt=jcel;
                    //mdw 7-11-2006 check added to use closest axis cell
                    icelt=std::min(nxdx-1,icelt);
                    icelt=std::max(1,icelt);
                    jcelt=std::min(nydy-1,jcelt);
                    jcelt=std::max(1,jcelt);
                    int idceltk=k*nxdx*nydy + jcelt*nxdx +icelt;
                    int idceltktop=ktop*nxdx*nydy + jcelt*nxdx +icelt;
                    int iceltjcelt=jcelt*nxdx + icelt;
                    utott=sqrt(wind_vel[idceltk].u*wind_vel[idceltk].u+wind_vel[idceltk].v*wind_vel[idceltk].v
                               +wind_vel[idceltk].w*wind_vel[idceltk].w);
                    utott=utott+.000001;
                    delut=sqrt(pow( (urefu.at(ik)-wind_vel[idceltk].u),2)+
                               pow( (urefv.at(ik)-wind_vel[idceltk].v),2)+
                               pow( (urefw.at(ik)-wind_vel[idceltk].w),2));
                    //mdw 4-16-2004 added correction for ktop+3 > nz-1
                    if(k==ktp){
                        if(ktop<=nzdz-1){
                            utotktp.at(iceltjcelt)=utott;
                            uktop.at(iceltjcelt)=wind_vel[idceltktop].u;
                            vktop.at(iceltjcelt)=wind_vel[idceltktop].v;
                            wktop.at(iceltjcelt)=wind_vel[idceltktop].w;
                        }
                        else{
                            utotktp.at(iceltjcelt)=sqrt(ualoft*ualoft+valoft*valoft);
                            uktop.at(iceltjcelt)=ualoft;
                            vktop.at(iceltjcelt)=valoft;
                            wktop.at(iceltjcelt)=0.;
                        }
                    }
                    //mdw 4-16-2004 added correction for ktop+3 > nz-1
                    if(k!=ktp){
                        delutz=sqrt(pow( (wind_vel[idceltk].u-zcorf.at(k)*uktop.at(iceltjcelt)),2)+
                                    pow( (wind_vel[idceltk].v-zcorf.at(k)*vktop.at(iceltjcelt)),2)+
                                    pow( (wind_vel[idceltk].w-zcorf.at(k)*wktop.at(iceltjcelt)),2));
                        // mdw 4-16-2004 added proper treatment of zfo
                        // mdw 6-10-2004 made check on centerline deluc rather than local delut
                        if((2.*deluc.at(ik)/weff.at(i))<(utotktp.at(iceltjcelt)/(ht[i]+zfo[i])) 
                           && delutz>.2*zcorf.at(k)*utotktp.at(iceltjcelt)){
                            if(ustargz.at(idceltk)<knlc*utotktp.at(iceltjcelt)){
                                ustargz.at(idceltk)=knlc*utotktp.at(iceltjcelt);
                                elzg.at(idceltk)=ht[i]+zfo[i];
                                upvpg=0.;
                                upwpg=-ctau13*zbrac*ustargz.at(idceltk)*ustargz.at(idceltk);
                                upsqg=cusq*zbrac*ustargz.at(idceltk)*ustargz.at(idceltk);
                                vpsqg=cvsq*zbrac*ustargz.at(idceltk)*ustargz.at(idceltk);
                                vpwpg=0.;
                                wpsqg=cwsq*zbrac*ustargz.at(idceltk)*ustargz.at(idceltk);
                                ustarg.at(idceltk)=ustargz.at(idceltk);
                                rotate2d(idceltk,cosphi,sinphi,upsqg,upvpg,vpsqg,wpsqg,upwpg,vpwpg);
                            }
                        }
                        else{
                            // We use the vertical gradient as dominant if it is sharper than the horizontal
                            cosl=wind_vel[idceltk].u/utott;
                            sinl=wind_vel[idceltk].v/utott;
                            duy=-deluc.at(ik)*sinl*cosu+deluc.at(ik)*sinv*cosl;
                            // mdw 6-10-2004 made check on centerline rather than local value
                            if(delut>.2*uref.at(ik)){
                                if(ustarg.at(idceltk)<knlc*deluc.at(ik)){
                                    ustarg.at(idceltk)=knlc*deluc.at(ik);
                                    // for eddy transport in uv we dont consider uw
                                    upvpg=-ctau13*zbrac*((ist-1.f)/(istf-1.f))*ustarg.at(idceltk)*ustarg.at(idceltk);
                                    upwpg=0.;
                                    // for eddy transport in uv we dont consider uw
                                    upsqg=cusq*zbrac*ustarg.at(idceltk)*ustarg.at(idceltk);
                                    wpsqg=cvsq*zbrac*ustarg.at(idceltk)*ustarg.at(idceltk);
                                    vpwpg=0.;
                                    vpsqg=cwsq*zbrac*ustarg.at(idceltk)*ustarg.at(idceltk);
                                    elzg.at(idceltk)=0.5*weff.at(i);
                                    rotate2d(idceltk,cosphi,sinphi,upsqg,upvpg,vpsqg,wpsqg,upwpg,vpwpg);
                                }
                            }
                        }
                        if(is==isini){
                            for(int isin=isini+1;isin<=istinf;isin++){//do isin=isini+1,istinf
                                xceln=xcelt+ds*(isin-1)*cosphi;
                                yceln=ycelt+ds*(isin-1)*sinphi;
                                temp=(xceln-dx)/dx;
                                iceln=nint(temp)+1; 
                                temp=(yceln-dy)/dy;
                                jceln=nint(temp)+1; 
                                iceln=std::min(nxdx-1,iceln);
                                iceln=std::max(1,iceln);
                                jceln=std::min(nydy-1,jceln);
                                jceln=std::max(1,jceln);
                                
                                int idcelnk=k*nxdx*nydy + jceln*nxdx +iceln;
                                int idcelnktop=ktop*nxdx*nydy + jceln*nxdx +iceln;
                                int icelnjceln=jceln*nxdx + iceln;
                                // mdw 3/22/2004pm added new if then structure to prevent replacing non-zero
                                // ustarg s with zero ones
                                if(ustarg.at(idceltk)>ustarg.at(idcelnk)){
                                    ustarg.at(idcelnk)=ustarg.at(idceltk);
                                    elzg.at(idcelnk)=elzg.at(idceltk);
                                    ufsqgi.at(idcelnk)=ufsqgi.at(idceltk);
                                    vfsqgi.at(idcelnk)=vfsqgi.at(idceltk);
                                    wfsqgi.at(idcelnk)=wfsqgi.at(idceltk);
                                    ufvfgi.at(idcelnk)=ufvfgi.at(idceltk);
                                    ufwfgi.at(idcelnk)=ufwfgi.at(idceltk);
                                    vfwfgi.at(idcelnk)=vfwfgi.at(idceltk);
                                }
                                // mdw 3/22/2004pm end  of new if then structure
                            }//enddo
                        }
                    }
                    // opposite direction in the transverse of the front eddy
                    xcelt=xcell-ds*(ist-1.f)*cosphit;
                    ycelt=ycell-ds*(ist-1.f)*sinphit;
                    //mdw 7-05-2006 changed form from nint( / ) to nint( / )+1
                    temp=(xcelt-dx)/dx;
                    icelt=nint(temp)+1; 
                    temp=(ycelt-dy)/dy;
                    jcelt=nint(temp)+1; 
                    
                    if(fabs(xcelt-xcell)<.5*ds)icelt=icel;
                    if(fabs(ycelt-ycell)<.5*ds)jcelt=jcel;
                    //mdw 7-11-2006 check added to use closest axis cell
                    icelt=std::min(nxdx-1,icelt);
                    icelt=std::max(1,icelt);
                    jcelt=std::min(nydy-1,jcelt);
                    jcelt=std::max(1,jcelt);
                    iceltjcelt=jcelt*nxdx + icelt;
                    idceltktop=ktop*nxdx*nydy+jcelt*nxdx +icelt;
                    idceltk=k*nxdx*nydy +jcelt*nxdx +icelt;
                    utott=sqrt(wind_vel[idceltk].u*wind_vel[idceltk].u+wind_vel[idceltk].v*wind_vel[idceltk].v
                               +wind_vel[idceltk].w*wind_vel[idceltk].w);
                    
                    utott=utott+.000001;
                    //mdw 4-16-2004 added correction for ktop+3 > nz-1
                    if(k==ktp){
                        if(ktop<=nzdz-1){
                            utotktp.at(iceltjcelt)=utott;
                            uktop.at(iceltjcelt)=wind_vel[idceltktop].u;
                            vktop.at(iceltjcelt)=wind_vel[idceltktop].v;
                            wktop.at(iceltjcelt)=wind_vel[idceltktop].w;
                        }
                        else{
                            utotktp.at(iceltjcelt)=sqrt(ualoft*ualoft+valoft*valoft);
                            uktop.at(iceltjcelt)=ualoft;
                            vktop.at(iceltjcelt)=valoft;
                            wktop.at(iceltjcelt)=0.;
                        }
                    }
                    delut=sqrt(pow( (urefu.at(ik)-wind_vel[idceltk].u),2)
                               +pow( (urefv.at(ik)-wind_vel[idceltk].v),2)
                               +pow( (urefw.at(ik)-wind_vel[idceltk].w),2));
                    //mdw 4-16-2004 added correction for ktop+3 > nz-1
                    if(k!=ktp){
                        delutz=sqrt(pow( (wind_vel[idceltk].u-zcorf.at(k)*uktop.at(iceltjcelt)),2)+
                                    pow( (wind_vel[idceltk].v-zcorf.at(k)*vktop.at(iceltjcelt)),2)+
                                    pow( (wind_vel[idceltk].w-zcorf.at(k)*wktop.at(iceltjcelt)),2));
                        // mdw 4-16-2004 added proper treatment of zfo
                        // mdw 6-10-2004 made check on centerline rather than local value
                        if((2.*deluc.at(ik)/weff.at(i))<(utotktp.at(iceltjcelt)/(ht[i]+zfo[i])) 
                           &&delutz>.2*zcorf.at(k)*utotktp.at(iceltjcelt)){
                            
                            if(ustargz.at(idceltk)<knlc*utotktp.at(iceltjcelt)){
                                ustargz.at(idceltk)=knlc*utotktp.at(iceltjcelt);
                                elzg.at(idceltk)=ht[i]+zfo[i];
                                upvpg=0.;
                                upwpg=-ctau13*zbrac*ustargz.at(idceltk)*ustargz.at(idceltk);
                                upsqg=cusq*zbrac*ustargz.at(idceltk)*ustargz.at(idceltk);
                                vpsqg=cvsq*zbrac*ustargz.at(idceltk)*ustargz.at(idceltk);
                                vpwpg=0.;
                                wpsqg=cwsq*zbrac*ustargz.at(idceltk)*ustargz.at(idceltk);
                                ustarg.at(idceltk)=ustargz.at(idceltk);
                                rotate2d(idceltk,cosphi,sinphi,upsqg,upvpg,vpsqg,wpsqg,upwpg,vpwpg);
                            }
                        }
                        else{
                            // We use the vertical gradient as dominant if it is sharper than the horizontal
                            cosl=wind_vel[idceltk].u/utott;
                            sinl=wind_vel[idceltk].v/utott;
                            duy=-deluc.at(ik)*sinl*cosu+deluc.at(ik)*sinv*cosl;
                            // mdw 6-10-2004 made check on centerline rather than local (delut) value
                            if(delut>.2*uref.at(k)){
                                if(ustarg.at(idceltk)<knlc*deluc.at(ik)&&ustargz.at(idceltk)<knlc*deluc.at(ik)){
                                    ustarg.at(idceltk)=knlc*deluc.at(ik);
                                    
                                    // for eddy transport in uv we dont consider uw
                                    float tau13=0.;
                                    upvpg=-tau13*zbrac*((ist-1.f)/(istf-1.f))*ustarg.at(idceltk)*ustarg.at(idceltk);//check:: might be bug in QP
                                    upwpg=0.;
                                    // for eddy transport in uv we dont consider uw
                                    upsqg=cusq*zbrac*ustarg.at(idceltk)*ustarg.at(idceltk);
                                    wpsqg=cvsq*zbrac*ustarg.at(idceltk)*ustarg.at(idceltk);
                                    vpwpg=0.;
                                    elzg.at(idceltk)=0.5*weff.at(i);
                                    vpsqg=cwsq*zbrac*ustarg.at(idceltk)*ustarg.at(idceltk);
                                    rotate2d(idceltk,cosphi,sinphi,upsqg,upvpg,vpsqg,wpsqg,upwpg,vpwpg);
                                }
                            }
                        }
                        if(is==isini){
                            for(int isin=isini+1;isin<=istinf;isin++){//do isin=isini+1,istinf{
                                xceln=xcelt+ds*(isin-1)*cosphi;
                                yceln=ycelt+ds*(isin-1)*sinphi;
                                temp=(xceln-dx)/dx;
                                iceln=nint(temp)+1; 
                                temp=(yceln-dy)/dy;
                                jceln=nint(temp)+1; 
                                iceln=std::min(nxdx-1,iceln);
                                iceln=std::max(1,iceln);
                                jceln=std::min(nydy-1,jceln);
                                jceln=std::max(1,jceln);
                                int idcelnk=k*nxdx*nydy + jceln*nxdx +iceln;
                                int idcelnktop=ktop*nxdx*nydy + jceln*nxdx +iceln;
                                int icelnjceln=jceln*nxdx + iceln;
                                // mdw 3/22/2004pm added new if then structure to prevent replacing non-zero
                                // ustargs with zero ones
                                if(ustarg.at(idceltk)>ustarg.at(idcelnk)){
                                    ustarg.at(idcelnk)=ustarg.at(idceltk);
                                    
                                    elzg.at(idcelnk)=elzg.at(idceltk);
                                    ufsqgi.at(idcelnk)=ufsqgi.at(idceltk);
                                    vfsqgi.at(idcelnk)=vfsqgi.at(idceltk);
                                    wfsqgi.at(idcelnk)=wfsqgi.at(idceltk);
                                    ufvfgi.at(idcelnk)=ufvfgi.at(idceltk);
                                    ufwfgi.at(idcelnk)=ufwfgi.at(idceltk);
                                    vfwfgi.at(idcelnk)=vfwfgi.at(idceltk);
                                }
                                // mdw 3/22/2004pm end of new if then structure
                            }
                        }
                    }

                }//   lp023
            }//   lp024
        }//   lp025
        float xpent1, ypent1;
        int npentx,npenty,ipent1,ipent2,jpent1,jpent2,ibuild;
        
        ibuild=1;
        switch(bldtype.at(i)){
        case(3):
            xpent1=xfo[i]-wti[i]*.2f-dx;
            temp = ((.4f*wti[i])/dx); 
            npentx=nint(temp)+1; 
            ypent1=yfo[i]-wti[i]*.2-dy;
            temp=((.4f*wti[i])/dy);
            npenty=nint(temp)+1; 
            temp = (xpent1-dx)/dx;
            ipent1=nint(temp)+1; 
            ipent2=ipent1+npentx;
            temp= ((ypent1-dy)/dy);
            jpent1=nint(temp)+1; 
            jpent2=jpent1+npenty;
            
            for(int icel=ipent1;icel<=ipent2;icel++){
                for(int jcel=jpent1;jcel<=jpent2;jcel++){
                    for(int k=ktp;k>=0;k--){
                        int idcelk=k*nxdx*nydy + jcel*nxdx +icel;    
                        int iceljcel=jcel*nxdx +icel;
                        
                        utot=sqrt(wind_vel[idcelk].u*wind_vel[idcelk].u+wind_vel[idcelk].v*wind_vel[idcelk].v+wind_vel[idcelk].w*wind_vel[idcelk].w);
                        utot=utot+.000001;
                        
                        //mdw 4-16-2004 added correction for ktop+3 > nz-1
                        if(k==ktp){
                            if(ktop<=nzdz){
                                int idcelktop=ktop*nxdx*nydy + jcel*nxdx +icel;
                                int iceljcel=jcel*nxdx +icel; 
                                utotktp.at(iceljcel)=utot;
                                uktop.at(iceljcel)=wind_vel[idcelktop].u;
                                vktop.at(iceljcel)=wind_vel[idcelktop].v;
                                wktop.at(iceljcel)=wind_vel[idcelktop].w;
                            }
                            else{
                                utotktp.at(iceljcel)=sqrt(ualoft*ualoft+valoft*valoft);
                                uktop.at(iceljcel)=ualoft;
                                vktop.at(iceljcel)=valoft;
                                wktop.at(iceljcel)=0.;
                            }
                        }
                        if(k!=ktp && cellQuic[idcelk].c != 0){
                            // MAN 9/14/2005 pentagon courtyard nonlocal mixing fix
                            delutz=sqrt((wind_vel[idcelk].u-uktop.at(iceljcel))*(wind_vel[idcelk].u-uktop.at(iceljcel))
                                        +(wind_vel[idcelk].v-vktop.at(iceljcel))*(wind_vel[idcelk].v-vktop.at(iceljcel))+ 
                                        (wind_vel[idcelk].w-wktop.at(iceljcel))*(wind_vel[idcelk].w-wktop.at(iceljcel)));
                            if(delutz>.2*utotktp.at(iceljcel)){ // vertical dominates
                                // end MAN 9/14/2005           
                                if(ustargz.at(idcelk)<knlc*utotktp.at(iceljcel)){ // This value dominates over prev. buildings.
                                    ustargz.at(idcelk)=knlc*utotktp.at(iceljcel);
                                    elzg.at(idcelk)=ht[i]+zfo[i];
                                    upvpg=0.;
                                    upwpg=-ustargz.at(idcelk)*ustargz.at(idcelk);
                                    upsqg=6.25*ustargz.at(idcelk)*ustargz.at(idcelk);
                                    vpsqg=(4./6.25)*upsqg;
                                    vpwpg=0.;
                                    wpsqg=1.69*ustargz.at(idcelk)*ustargz.at(idcelk); // align sigmas with the overall mean wind
                                    upvpg=0.;
                                    upwpg=-ctau13*zbrac*ustargz.at(idcelk)*ustargz.at(idcelk);
                                    upsqg=cusq*zbrac*ustargz.at(idcelk)*ustargz.at(idcelk);
                                    vpsqg=cvsq*zbrac*ustargz.at(idcelk)*ustargz.at(idcelk);
                                    vpwpg=0.;
                                    wpsqg=cwsq*zbrac*ustargz.at(idcelk)*ustargz.at(idcelk); // align sigmas with the overall mean wind
                                    ustarg.at(idcelk)=ustargz.at(idcelk);
                                    rotate2d(idcelk,cosphi,sinphi,upsqg,upvpg,vpsqg,wpsqg,upwpg,vpwpg);
                                }
                            }
                        }
                    }
                    check1=check1+1;
                }
                check2=check2+1;
            }
            check3=check3+1;
            break;
        case(4):
        case(5):
            
            float x0=xfo[i]+0.5*lti[i]*cos(gamma.at(i)*pi/180.);
            float y0=yfo[i]+0.5*lti[i]*sin(gamma.at(i)*pi/180.);
            
            float x1=xfo[i]+0.5*wti[ibuild]*sin(gamma.at(i)*pi/180.);
            float y1=yfo[i]-0.5*wti[ibuild]*cos(gamma.at(i)*pi/180.);
            float x2=x1+lti[i]*cos(gamma.at(i)*pi/180.);
            float y2=y1+lti[i]*sin(gamma.at(i)*pi/180.);
            float x4=xfo[i]-0.5*wti[i]*sin(gamma.at(i)*pi/180.);
            float y4=yfo[i]+0.5*wti[i]*cos(gamma.at(i)*pi/180.);
            float x3=x4+lti[i]*cos(gamma.at(i)*pi/180.);
            float y3=y4+lti[i]*sin(gamma.at(i)*pi/180.);
            float temp1 = std::min(x1,x2);
            float temp2 = std::min(temp1,x3);
            float temp3 = std::min(temp2,x4);
            int icelmin = int(temp3/dx);
            temp1 = std::max(x1,x2);
            temp2 = std::max(temp1,x3);
            temp3 = std::max(temp2,x4);
            int icelmax = int(temp3/dx);

            temp1 = std::min(y1,y2);
            temp2 = std::min(temp1,y3);
            temp3 = std::min(temp2,y4);
            int jcelmin = int(temp3/dy);
            temp1 = std::max(y1,y2);
            temp2 = std::max(temp1,y3);
            temp3 = std::max(temp2,y4);
            int jcelmax = int(temp3/dy);
            
            for(int icel=icelmin;icel<=icelmax+1;icel++){
                for(int jcel=jcelmin; jcel<=jcelmax+1;jcel++){
                    float xc=(((icel)-0.5)*dx-x0)*cos(gamma.at(i)*pi/180.)+
                        (((jcel)-0.5)*dy-y0)*sin(gamma.at(i)*pi/180.);
                    float yc=-(((icel)-0.5)*dx-x0)*sin(gamma.at(i)*pi/180.)+
                        (((jcel)-0.5)*dy-y0)*cos(gamma.at(i)*pi/180.);
					int kk=0;
                    for(int k=1;k<=ktp;k++){
                        kk=k;
                        if(zfo[i]<z.at(k))break;
                    }
                    int kzfo=kk;
                    
                    for(k=ktp;k>=kzfo;k--){ 
                        dz=dz_array.at(k);
                        int incourt=0;
                        int idcelk=k*nxdx*nydy + jcel*nxdx +icel;
                        int idcelktop=ktop*nxdx*nydy + jcel*nxdx +icel;
                        int iceljcel=jcel*nxdx +icel;
                        if(cellQuic[idcelk].c != 0){
                            utot=sqrt(wind_vel[idcelk].u*wind_vel[idcelk].u+wind_vel[idcelk].v*wind_vel[idcelk].v+wind_vel[idcelk].w*wind_vel[idcelk].w)+.000001;
                            if(bldtype.at(i) == 4){
                                if(xc > -0.5*lti[i] && xc < 0.5*lti[i] && 
                                   yc > -0.5*wti[i] && yc < 0.5*wti[i]){
                                    incourt=1;
                                }
                            }
                            else{
                                float rc=sqrt((xc*xc)+(yc*yc));
                                float tc=atan2(yc,xc);
                                if(rc < 0.25*lti[i]*wti[i]/
                                   sqrt((pow( (0.5f*lti[i]*sin(tc)),2.f))+(pow( (0.5f*wti[i]*cos(tc)),2.f)))){
                                    incourt=1;
                                }
                            }
                        }
                        else{
                            continue;
                        }
                        //mdw 4-16-2004 added correction for ktop+3 > nz-1
                        if(incourt == 1){
                            if(k==ktp){
                                if(ktop<=nz-1){
                                    utotktp.at(iceljcel)=utot;
                                    uktop.at(iceljcel)=wind_vel[idcelktop].u;
                                    vktop.at(iceljcel)=wind_vel[idcelktop].v;
                                    wktop.at(iceljcel)=wind_vel[idcelktop].w;
                                }
                                else{
                                    utotktp.at(iceljcel)=sqrt(ualoft*ualoft+valoft*valoft);
                                    uktop.at(iceljcel)=ualoft;
                                    vktop.at(iceljcel)=valoft;
                                    wktop.at(iceljcel)=0.;
                                }
                            }
                            if(k!=ktp && cellQuic[idcelk].c != 0){
                                // MAN 9/14/2005 pentagon courtyard nonlocal mixing fix
                                delutz=sqrt((wind_vel[idcelk].u-uktop.at(iceljcel))*(wind_vel[idcelk].u-uktop.at(iceljcel))
                                            +(wind_vel[idcelk].v-vktop.at(iceljcel))*(wind_vel[idcelk].v-vktop.at(iceljcel))+ 
                                            (wind_vel[idcelk].w-wktop.at(iceljcel))*(wind_vel[idcelk].w-wktop.at(iceljcel)));
                                if(delutz>.2*utotktp.at(iceljcel)){ // vertical dominates
                                    // end MAN 9/14/2005              
                                    if(ustargz.at(idcelk)<knlc*utotktp.at(iceljcel)){ // This value dominates over prev. buildings.
                                        ustargz.at(idcelk)=knlc*utotktp.at(iceljcel);
                                        elzg.at(idcelk)=ht[i]+zfo[i];
                                        upvpg=0.;
                                        upwpg=-ustargz.at(idcelk)*ustargz.at(idcelk);
                                        upsqg=6.25*ustargz.at(idcelk)*ustargz.at(idcelk);
                                        vpsqg=(4./6.25)*upsqg;
                                        vpwpg=0.;
                                        wpsqg=1.69*ustargz.at(idcelk)*ustargz.at(idcelk); // align sigmas with the overall mean wind
                                        upvpg=0.;
                                        upwpg=-ctau13*zbrac*ustargz.at(idcelk)*ustargz.at(idcelk);
                                        upsqg=cusq*zbrac*ustargz.at(idcelk)*ustargz.at(idcelk);
                                        vpsqg=cvsq*zbrac*ustargz.at(idcelk)*ustargz.at(idcelk);
                                        vpwpg=0.;
                                        wpsqg=cwsq*zbrac*ustargz.at(idcelk)*ustargz.at(idcelk); // align sigmas with the overall mean wind
                                        ustarg.at(idcelk)=ustargz.at(idcelk);
                                        rotate2d(idcelk,cosphi,sinphi,upsqg,upvpg,vpsqg,wpsqg,upwpg,vpwpg);
                                    }
                                }
                            }
                        }
                    }
                    check1=check1+1;
                }
                check2=check2+1;
            }
            check3=check3+1;
        }
        check=check+1;
        
         
    }//for loop for buildings for non-local mixing ends
    
    // Following code is for local mixing-Balli -06/14/09
    // calculate distance to ground and walls if within 2 cells
    float zbrac=0.;
    float m_roof=6.;
    float eps=0.;
    float sigu=0.;
    float sigv=0.;
    float sigw=0.;
    float upwp=0.;
    float delym=0.;
    float delxm=0.;
    float u3psq=0.;
    float v3psq=0.;
    float w3psq=0.;
    float upvp=0.;
    float vpwp=0.;
    float ufwf=0.;
    float ufvf=0.;
    float vfwf=0.;
    float utotm=0.;
    float utotp=0.;
    float dutotdxp=0.;
    float dutotdxm=0.;
    float dutotdxa=0.;
    float dutotdyp=0.;
    float dutotdym=0.;
    float dutotdyc=0.;
    float dutotdya=0.;
    float x_b=0.;
    float y_b=0.;
    float dwallg=0.;
    float elzv=0.;
    float xloc=0.;
    float ufsq=0.;
    float vfsq=0.;
    float wfsq=0.;
    float dwall=0.;
    float ufsqb=0.;
    float wfsqb=0.;
    float vfsqb=0.;
    
    
    
    for(int j=0;j<nydy;j++){//altered for GPU, in QP it was ->do j=1,ny-1 -Balli(06/14/09)
        for(int i=0;i<nxdx;i++){//altered for GPU, in QP it was ->do i=1,nx-1 -Balli(06/14/09)
            int ij=j*nxdx+i;
            if(hgtveg.at(ij) > 0.){
                for(int  kk=1;kk<=nzdz;kk++){
                    kcantop=kk;
                    if(hgtveg.at(ij) <= z.at(kk))break;
                }
                int idcan=kcantop*nxdx*nydy+j*nxdx+i;
                int idcan1=(kcantop+1)*nxdx*nydy+j*nxdx+i;
                ucantop=.5*sqrt(wind_vel[idcan].u*wind_vel[idcan].u+wind_vel[idcan].v*wind_vel[idcan].v+wind_vel[idcan].w*wind_vel[idcan].w)+
                    .5*sqrt(wind_vel[idcan1].u*wind_vel[idcan1].u+wind_vel[idcan1].v*wind_vel[idcan1].v+wind_vel[idcan1].w*wind_vel[idcan1].w);
            }
            for(int k=0;k<nzdz;k++){ //altered for GPU, in QP it was ->do k=2,nz-1 -Balli(06/14/09)
                sigu=0.;
                sigv=0.;
                sigw=0.;
                
                int id=k*nxdx*nydy +j*nxdx +i;
                int idkm1=(k-1)*nxdx*nydy +j*nxdx +i;
                int kp1=(k+1)*nxdx*nydy +j*nxdx +i;
                
                int row = k / (numInRow);
                int texidx = row * width * nydy * 4 +
                    j * width * 4 +
                    k % (numInRow) * nxdx * 4 +
                    i * 4;
                
                sigui.at(id)=0.;
                sigvi.at(id)=0.;
                sigwi.at(id)=0.;


                //Balli:New stuff- vertical varying grid
                dz=dz_array.at(k);
                elcanopy=0.;
                int klim=1;
                
                if(cellQuic[id].c==8){
                    zbrac=1.;
                }
                else{
                    if(zi.at(k)<.99*h){
                        zbrac=pow( (1.f-zi.at(k)/h),1.5f);
                    }
                    else{
                        zbrac=pow( (1.f-.99f),1.5f);
                    }
                }
                dzm.at(id)=std::max(zm.at(k)-hgt.at(ij),0.f);
                if(dzm.at(id)>zi.at(k)-std::max((hgtveg.at(ij)-elcanopy/kkar),0.f)&&(zi.at(k)-std::max((hgtveg.at(ij)-elcanopy/kkar),0.f)>0.f))
                    dzm.at(id)=zi.at(k)-std::max((hgtveg.at(ij)-elcanopy/kkar),0.f);
                // MAN 9/21/2005 roof top mixing length fix
                eleff.at(id)=dzm.at(id);
                if(cellQuic[id].c==8)eleff.at(id)=elcanopy*std::min(1.,(dzm.at(id)-z0)/(.3*hgtveg.at(ij)))/kkar;
                int kdif=k-1;
                eleff.at(id)=std::max(zm.at(k)-hgt.at(ij)*pow( (hgt.at(ij)/zm.at(k)),m_roof),0.f);
                if(zi.at(k)-std::max((hgtveg.at(ij)-elcanopy/kkar),0.f)>0.f)
                    eleff.at(id)=std::min(eleff.at(id),zi.at(k)-std::max((hgtveg.at(ij)-elcanopy/kkar),0.f));
                if(cellQuic[id].c==8)eleff.at(id)=elcanopy*std::min(1.,(zi.at(k)-z0)/(.3*hgtveg.at(ij)))/kkar;
                klim=nzdz;
                // calculation of ustar in the vertical
                if(cellQuic[idkm1].c == 0 && cellQuic[id].c != 0){
                    utot=sqrt(wind_vel[id].u*wind_vel[id].u+wind_vel[id].v*wind_vel[id].v*+wind_vel[id].w*wind_vel[id].w);
                    if(cellQuic[id].c!=8){
                        if(rcl>0){
                            phim=1.+4.7*rcl*0.5*dz;
                            psim=-4.7*rcl*0.5*dz;
                        }
                        else{
                            phim=pow( (1.f-15.f*rcl*0.5f*dz),(-.25f));
                            psim=2.*log((1.+1./phim)/2.)+log((1.+1./pow(phim,2.f))/2.)-2.*atan(1./phim)+pi/2.;
                        }
                    }
                    else{
                        phim=1.;
                        psim=0.;
                        elz.at(id)=elcanopy*std::min(1.,(.5*dz-z0)*kkar/(.3*hgtveg.at(ij)));
                        ustar=elz.at(id)*utot/(.5*dz-z0);
                        dutotdzi.at(id)=ustar/elz.at(id);
                        ustarz.at(id)=ustar;
                    }
                }
                else{
                    if(cellQuic[id].c != 0){
                        utotu=sqrt(wind_vel[kp1].u*wind_vel[kp1].u+wind_vel[kp1].v*wind_vel[kp1].v+wind_vel[kp1].w*wind_vel[kp1].w);
                        utot=sqrt(wind_vel[id].u*wind_vel[id].u+wind_vel[id].v*wind_vel[id].v*+wind_vel[id].w*wind_vel[id].w);
                        if(fabs(dutotdzi.at(id))>1.e-06 && cellQuic[id].c!=8 && dzm.at(id)>2.*dz){
                            elz.at(id)=kkar*utot/fabs(dutotdzi.at(id));
                            // MAN 9/21/2005 roof top mixing length fix
                            if((kkar*eleff.at(id))<elz.at(id))
                                elz.at(id)=kkar*eleff.at(id);
                            else
                                elz.at(id)=kkar*eleff.at(id);
                            if(cellQuic[id].c==8){
                                elz.at(id)=elcanopy*std::min(1.,(zi.at(k)-z0)/(.3*hgtveg.at(ij)));
                                eleff.at(id)=elz.at(id)/kkar;
                            }
                        }
                        if(k < nz-1){
                            if((cellQuic[kp1].c!=8)&&(cellQuic[id].c==8)){
                                if(fabs(dutotdzi.at(id))>1.e-06){
                                    elz.at(id)=kkar*utot/fabs(dutotdzi.at(id));
                                    if((kkar*eleff.at(id))<elz.at(id)) elz.at(id)=kkar*eleff.at(id);
                                }
                            }
                        }
                        // We have just put in the vortex mixing length for the last cell in the canopy
                        if(cellQuic[idkm1].c!=8){
                            if(rcl>0){
                                phim=1.+4.7*rcl*eleff.at(id);
                                psim=-4.7*rcl*eleff.at(id);
                            }
                            else{
                                phim=pow( (1.f-15.f*rcl*eleff.at(id)),(-.25f));
                                psim=2.*log((1.+1./phim)/2.)+log((1.+1./pow(phim,2.f))/2.)-2.*atan(1./phim)+pi/2.;
                            }
                            ustar=kkar*eleff.at(id)*fabs(dutotdzi.at(id))/phim;
                            ustarz.at(id)=ustar;
                        }
                        else{
                            utotl=sqrt(wind_vel[idkm1].u*wind_vel[idkm1].u+wind_vel[idkm1].v*wind_vel[idkm1].v+wind_vel[idkm1].w*wind_vel[idkm1].w);
                            dutotdzi.at(id)=(utotu-utotl)/(2.*dz);
                            //corrected wrong gradient at the vegetative canopy top 12/22/2008
                            if(cellQuic[id].c!=8){
                                elz.at(id)=kkar*.5*dz;
                            }
                            else{
                                elz.at(id)=elcanopy*std::min(1.,(zi.at(k)-z0)/(.3*hgtveg.at(ij)));
                            }
                            ustar=elz.at(id)*fabs(dutotdzi.at(id));
                            ustarz.at(id)=ustar;
                        }
                        if(cellQuic[idkm1].c!=8){
                            if(rcl>0){
                                phim=1.+4.7*rcl*eleff.at(id);
                                psim=-4.7*rcl*eleff.at(id);
                            }
                            else{
                                phim=pow( (1.f-15.f*rcl*eleff.at(id)),(-.25f));
                                psim=2.*log((1.+1./phim)/2.)+log((1.+1./pow(phim,2.f))/2.)-2.*atan(1./phim)+pi/2.;
                            }
                            ustar=kkar*eleff.at(id)*fabs(dutotdzi.at(id))/phim;
                            ustarz.at(id)=ustar;
                        }
                    }
                }
                // for neutral conditions sigw is only dependent on ustar
                
                // for vertical downward distance (dzm)
                klim=nzdz-1;
                
                float dutotdn=0.f;
                float dutotds=0.f;
                
                sig[id].u = 0.0f;
                sig[id].v = 0.0f;   
                sig[id].w = 0.0f;   
                
                tau[id].t11   = 0.0f;        
                tau[id+1].t22 = 0.0f;        
                tau[id+2].t33 = 0.0f;
                tau[id+3].t13 = 0.0f;
                //Make tau's a texture so that they can be visualized as horizontal layers in the domain
                dataTau[texidx] = 0.0f;   
                dataTau[texidx+1] =0.0f;  
                dataTau[texidx+2] =0.0f;  
                dataTau[texidx+3] = 0.0f; 
                
                dataTwo[texidx]   = 0.0f;
                dataTwo[texidx+1] =  0.0f;
                dataTwo[texidx+2] =  0.0f;
                dataTwo[texidx+3] =  0.0f;
                
                dataWind[texidx]   = wind_vel[id].u;
                dataWind[texidx+1] = wind_vel[id].v;
                dataWind[texidx+2] = wind_vel[id].w;
                dataWind[texidx+3] = 0.;
                if(k>0){
                    dzp.at(id)=10.f*dz+z.at(klim)-z.at(k-1)+.5f*dz;
                }
                else{
                    dzp.at(id)=10.f*dz+z.at(klim)+.5f*dz;
                }
                for(int kk=k;kk<=klim;kk++){//do kk=k,klim
                    int idkk=kk*nxdx*nydy +j*nxdx +i;
                    int celltypeidkk=0;
                    if(idkk>=0){
                        celltypeidkk=cellQuic[idkk].c;
                    }
                    if(celltypeidkk == 0){
                        dzp.at(id)=.5*dz+(kk-k-1)*dz;
                        break;
                    }
                }
                // for distance to the left (dxm)
                int ilim=0;
                dxm.at(id)=.5*dx+i*dx+(nxdx+1)*dx;//altered for GPU [added 1 to nxdx as in QP nx is nx+1 actually] -Balli(06/14/09)
                for(int ii=i;ii>ilim;ii--){//do ii=i,ilim,-1
                    // calculation of the distance to the wall in the negative x direction
                    int idii=k*nxdx*nydy +j*nxdx +ii;
                    int celltypeidii=1;//as in x or y direction we assume fluid in all directions out of domain
                    if(idii>=0){
                        celltypeidii=cellQuic[idii].c;
                    }                        
                    if(celltypeidii == 0){
                        dxm.at(id)=.5*dx+(i-ii-1)*dx;
                        break;
                    }
                }
                // for distance to the right (dxp)
                ilim=nxdx-1;
                dxp.at(id)=.5*dx+(ilim-i)*dx+(nxdx+1)*dx;
                for(int ii=i;ii<=ilim;ii++){// ii=i,ilim
                    // calculation of the distance to the wall in the positive x direction
                    int idii=k*nxdx*nydy +j*nxdx +ii;
                    int celltypeidii=1;//as in x or y direction we assume fluid in all directions out of domain
                    if(idii>=0){
                        celltypeidii=cellQuic[idii].c;
                    }
                    
                    if(celltypeidii == 0){
                        dxp.at(id)=.5*dx+(ii-i-1)*dx;
                        break;
                    }
                }
                // for distance  from the back (dym)
                int jlim=0;
                dym.at(id)=.5*dy+j*dy+(nydy+1)*dy; //added 1 to nydy ,same reason as in x-directiom, see above-Balli(06/14/09)
                for(int jj=j;jj>jlim;jj--){//do jj=j,jlim,-1
                    // calculation of the distance to the wall in the negative y direction
                    int idjj=k*nxdx*nydy +jj*nxdx +i;
                    int celltypeidjj=1;//as in x or y direction we assume fluid in all directions out of domain
                    if(idjj>=0){
                        celltypeidjj=cellQuic[idjj].c;
                    }
                    if(celltypeidjj == 0){
                        dym.at(id)=.5*dy+(j-jj-1)*dy;
                        break;
                    }
                }
                // for distance to the front  (dyp)
                jlim=nydy-1;
                dyp.at(id)=.5*dy+(jlim-j)*dy+(nydy+1)*dy; //added 1 to nydy ,same reason as in x-directiom, see above-Balli(06/14/09)
                for(int jj=j;jj<=jlim;jj++){//do jj=j,jlim
                    // calculation of the distance to the wall in the positive x direction
                    int idjj=k*nxdx*nydy +jj*nxdx +i;
                    int celltypeidjj=1;//as in x or y direction we assume fluid in all directions out of domain
                    if(idjj>=0){
                        celltypeidjj=cellQuic[idjj].c;
                    }
                    if(celltypeidjj == 0){
                        dyp.at(id)=.5*dy+(jj-j-1)*dy;
                        break;
                    }
                }
                // we need to calculate the largest change in utot
                if(cellQuic[id].c == 0){
                    eps=0.;
                    sigu=0.;
                    sigv=0.;
                    sigw=0.;
                    upwp=0.;
                    elz.at(id)=0.;
                    eleff.at(id)=0.;
                    sig[id].u = 0.0f;
                    sig[id].v = 0.0f;   
                    sig[id].w = 0.0f;
                }
                if(cellQuic[id].c != 0){//for all fuid cells
                    // first we set up parameters for cells near boundary
                    if(j<1||j>=ny-1||i<1||i>=nx-1){//boundary cells
                        // calculation of near-boundary values of u*y, ly, dely, and the
                        // gradients of speed in the x and y directions
                        delym=(nydy+1)*dy;
                        utot=mag(wind_vel[id]);
                        sigvi.at(id)=0.;
                        delxm=dx*(nxdx-1);
                        dutotdxi.at(id)=0.;
                        dutotdyi.at(id)=0.;
                        elz.at(id)=kkar*eleff.at(id);
                        dutotdni.at(id)=dutotdzi.at(id);

                        detang(0,id,dutotds,dutotdn,i,j,k);
                        if(rcl>0){
                            phim=1.+4.7*rcl*eleff.at(id);
                            psim=-4.7*rcl*eleff.at(id);
                        }
                        else{
                            phim=pow( (1.-15.*rcl*eleff.at(id)),(-.25) );
                            psim=2.*log((1.+1./phim)/2.)+log((1.+1./pow(phim,2.f))/2.)-2.*atan(1./phim)+pi/2.;
                        }
                        ustarz.at(id)=elz.at(id)*dutotdni.at(id)/phim; // calculate local ustar
                        ustarz.at(id)=std::max(ustarz.at(id),3.e-02f);
                        u3psq=cusq*zbrac*ustarz.at(id)*ustarz.at(id);   //// (u''')^2
                        v3psq=cvsq*zbrac*ustarz.at(id)*ustarz.at(id);   //...
                        w3psq=cwsq*zbrac*ustarz.at(id)*ustarz.at(id); //...
                        upwp=-ctau13*zbrac*ustarz.at(id)*ustarz.at(id); // -tau13
                        upvp=0.;
                        vpwp=0.;
                        if(rcl<0 && zi.at(k)<.99*h){
                            u3psq=u3psq+.6*(ustarz.at(id)*ustarz.at(id))*pow( (-h*rcl),(2.f/3.f));
                            v3psq=v3psq+.6*(ustarz.at(id)*ustarz.at(id))*pow( (-h*rcl),(2.f/3.f));
                            w3psq=w3psq+3.3*(ustarz.at(id)*ustarz.at(id))*pow( (-zi.at(k)*rcl),(2.3f))*pow( (1.f-.8f*zi.at(k)/h),2.f);
                            upwp=upwp*pow( (1.f-zi.at(k)/h),(.5f*rcl*h/(1.0f-rcl*h)) );
                        }
                        else{
                            if(rcl<0){
                                u3psq=u3psq+.6*(ustarz.at(id)*ustarz.at(id))*pow( (-h*rcl),(2.f/3.f));
                                v3psq=v3psq+.6*(ustarz.at(id)*ustarz.at(id))*pow( (-h*rcl),(2.f/3.f));
                                w3psq=w3psq+3.3*(ustarz.at(id)*ustarz.at(id))*pow( (-.99f*h*rcl),(2.3f))*pow( (1.f-.8f*.99f),2.f);
                                upwp=upwp*pow( (1.f-.99f),(.5f*rcl*h/(1.0f-rcl*h)) );  
                            }
                        }
                        rotu3psq(id,u3psq,utot,upvp,upwp,vpwp,v3psq,w3psq,ufsqb,wfsqb,vfsqb,ufvf,ufwf,vfwf); // rotate values back into the orig. grid
                        ufwfi.at(id)=ufwf;
                        ufvfi.at(id)=ufvf;
                        vfwfi.at(id)=vfwf;
                        ustarij.at(id)=ustarz.at(id);
                        sigui.at(id)=sqrt(u3psq);
                        sigvi.at(id)=sqrt(v3psq);
                        sigwi.at(id)=sqrt(w3psq);
                        upwpi.at(id)=upwp;
                        // along the boundaries we make y effects negligible
                    }
                    else{
                        utot=mag(wind_vel[id]);
                        // away from boundaries u*y, ly, dely, and gradients
                        int idim1=k*nxdx*nydy +j*nxdx +(i-1);
                        int idip1=k*nxdx*nydy +j*nxdx +(i+1);
                        if(cellQuic[idim1].c != 0 && cellQuic[id].c != 0  && cellQuic[idip1].c != 0){
                            //mdw 3-08-2004 start changes for highest gradient
                            utot=mag(wind_vel[id]);
                            utotm=sqrt(wind_vel[idim1].u*wind_vel[idim1].u+wind_vel[idim1].v*wind_vel[idim1].v+wind_vel[idim1].w*wind_vel[idim1].w);
                            utotp=sqrt(wind_vel[idip1].u*wind_vel[idip1].u+wind_vel[idip1].v*wind_vel[idip1].v+wind_vel[idip1].w*wind_vel[idip1].w);

                            dutotdxp=(utotp-utot)/dx;
                            dutotdxm=(utot-utotm)/dx;
                            dutotdxa=std::max(fabs(dutotdxp),fabs(dutotdxm));
                            if(dutotdxa==fabs(dutotdxm)){
                                dutotdxi.at(id)=dutotdxm;
                            }
                            else{
                                dutotdxi.at(id)=dutotdxp;
                            }
                            // mdw 3-08-2004end changes
                        }
                        else{
                            if(cellQuic[id].c == 0){ ////BALLI
                                dutotdxi.at(id)=0.;
                            }
                            else{
                                if(cellQuic[idim1].c == 0){ ////BALLI
                                    dutotdxi.at(id)=2.*mag(wind_vel[id])/dx;
                                    dutotdxi.at(id)=mag(wind_vel[id])
                                        /(log((.5*dx)/z0)*(.5*dx));
                                }
                                else{
                                    dutotdxi.at(id)=-mag(wind_vel[id])
                                        /(log((.5*dx)/z0)*(.5*dx));
                                }
                            }
                        }
                        
                        int idjm1=k*nxdx*nydy +(j-1)*nxdx +i;
                        int idjp1=k*nxdx*nydy +(j+1)*nxdx +i;
                        if(cellQuic[id].c != 0 && cellQuic[idjm1].c != 0 && cellQuic[idjp1].c != 0){
                            //mdw 3-08-2008 start gradient changes
                            utot=mag(wind_vel[id]);
                            utotm=sqrt(wind_vel[idjm1].u*wind_vel[idjm1].u+wind_vel[idjm1].v*wind_vel[idjm1].v+wind_vel[idjm1].w*wind_vel[idjm1].w);
                            utotp=sqrt(wind_vel[idjp1].u*wind_vel[idjp1].u+wind_vel[idjp1].v*wind_vel[idjp1].v+wind_vel[idjp1].w*wind_vel[idjp1].w);
                            dutotdyc=0.5*(utotp-utotm)/dy;
                            dutotdyp=(utotp-utot)/dy;
                            dutotdym=(utot-utotm)/dy;
                            dutotdya=std::max(fabs(dutotdyp),fabs(dutotdym));
                            if(dutotdya==fabs(dutotdym)){
                                dutotdyi.at(id)=dutotdym;
                            }
                            else{
                                dutotdyi.at(id)=dutotdyp;
                            }
                            // mdw 3-08-2004end changes
                        }
                        else{
                            if(cellQuic[id].c == 0){
                                dutotdyi.at(id)=0.;
                            }
                            else{
                                if(cellQuic[idjm1].c == 0){
                                    dutotdyi.at(id)=mag(wind_vel[id])
                                        /(log((.5*dy)/z0)*(.5*dy));
                                }
                                else{
                                    dutotdyi.at(id)=-mag(wind_vel[id])
                                        /(log((.5*dy)/z0)*(.5*dy));
                                }
                            }
                        }
                    }
                    detang(0,id,dutotds,dutotdn,i,j,k); // Calculates the parameters fot the triple rotatation of coord sys.
                    dwall=std::min(std::min(eleff.at(id),std::min(dxm.at(id),dxp.at(id))),std::min(dym.at(id),dyp.at(id)));
                    dwall=std::min(dwall,dzm.at(id));
                    elz.at(id)=kkar*dwall; // length scale based on distance to wall

                    if(cellQuic[id].c !=8)elz.at(id)=kkar*dwall; // length scale based on distance to wall
                    if(fabs(dutotdni.at(id))>1.e-6){
                        x_b=std::min(dxm.at(id),dxp.at(id));
                        if(x_b>std::max(del_b,dx)) x_b=0;
                        y_b=std::min(dym.at(id),dyp.at(id));
                        if(y_b>std::max(del_b,dy)) y_b=0;
                        dwallg=fabs(dutotdyi.at(id))*y_b+fabs(dutotdxi.at(id))*x_b;
                        dwallg=dwallg+fabs(dutotdzi.at(id))*eleff.at(id);
                        dwallg=dwallg/dutotdni.at(id);
                        elzv=kkar*utot/dutotdni.at(id); // length scale based on distance to null wind
                        if(dwallg*kkar<elzv && (x_b+y_b)>0.) {
                            // mdw 6-29-2006 changed test so that must be near vertical wall
                            elz.at(id)=kkar*dwallg; // pick the smallest length scale
                        }
                        else{
                            // mdw 6-30-2006 changed test so that vortex test does not override normal stuff
                            if(elzv<=elz.at(id)){
                                elz.at(id)=elzv;
                            }
                        }
                    }
                    if(cellQuic[id].c!=8){
                        if(rcl>0){
                            phim=1.+4.7*rcl*eleff.at(id);
                            psim=-4.7*rcl*eleff.at(id);
                        }
                        else{
                            phim=pow( (1.f-15.f*rcl*eleff.at(id)),(-.25f) );
                            psim=2.*log((1.+1./phim)/2.)+log((1.+1./pow(phim,2.f))/2.f)-2.*atan(1./phim)+pi/2.;
                        }
                    }
                    else{
                        phim=1.;
                        psim=0.;
                        elz.at(id)=elcanopy*std::min(1.f,(dzm.at(id)-z0)*kkar/(.3f*hgtveg.at(ij)));
                    }
                    ustarz.at(id)=elz.at(id)*dutotdni.at(id)/phim; // calculate local ustar
                    ustarz.at(id)=std::max(ustarz.at(id),3.e-02f);
                    //mdw 6-23-2004 adjust for vertical structure
                    u3psq=cusq*zbrac*ustarz.at(id)*ustarz.at(id);   // (u''')^2
                    v3psq=cvsq*zbrac*ustarz.at(id)*ustarz.at(id);   //...
                    w3psq=cwsq*zbrac*ustarz.at(id)*ustarz.at(id); //...
                    upwp=-ctau13*zbrac*ustarz.at(id)*ustarz.at(id); // -tau13
                    upvp=0.;
                    vpwp=0.;
                    xloc=1.;
                    if(ustarz.at(id)>xloc*ustarg.at(id)){
                        if(rcl<0. && zi.at(k)<.99*h){
                            u3psq=u3psq+.6*(ustarz.at(id)*ustarz.at(id))*pow( (-h*rcl),(2.f/3.f));
                            v3psq=v3psq+.6*(ustarz.at(id)*ustarz.at(id))*pow( (-h*rcl),(2.f/3.f));
                            w3psq=w3psq+3.3*(ustarz.at(id)*ustarz.at(id))*pow( (-zi.at(k)*rcl),(2.3f))*pow( (1.f-.8f*zi.at(k)/h),2.f);
                            upwp=upwp*pow( (1.f-zi.at(k)/h),(.5f*rcl*h/(1.f-rcl*h)) );
                        }
                        else{
                            if(rcl<0.){
                                u3psq=u3psq+.6*(ustarz.at(id)*ustarz.at(id))*pow( (-h*rcl),(2.f/3.f));
                                v3psq=v3psq+.6*(ustarz.at(id)*ustarz.at(id))*pow( (-h*rcl),(2.f/3.f));
                                w3psq=w3psq+3.3*(ustarz.at(id)*ustarz.at(id))*pow( (-.99f*h*rcl),(2.3f))*pow( (1.f-.8f*.99f),2.f);
                                upwp=upwp*pow( (1.f-.99f),(.5f*rcl*h/(1.f-rcl*h)) );
                            }
                        }
                        rotu3psq(id,u3psq,utot,upvp,upwp,vpwp,v3psq,w3psq,ufsqb,wfsqb,vfsqb,ufvf,ufwf,vfwf); // rotate values back into the orig. grid
                        ufwfi.at(id)=ufwf;
                        ufvfi.at(id)=ufvf;
                        vfwfi.at(id)=vfwf;
                        ustarij.at(id)=ustarz.at(id);
                        sigui.at(id)=sqrt(u3psq);
                        sigvi.at(id)=sqrt(v3psq);
                        sigwi.at(id)=sqrt(w3psq);
                        upwpi.at(id)=upwp;
                    }
                    else{ // non-local dominates (possible place for into of TPT)
                        ufsq=ufsqgi.at(id);
                        vfsq=vfsqgi.at(id);
                        wfsq=wfsqgi.at(id);
                        ufvf=ufvfgi.at(id);
                        ufwf=ufwfgi.at(id);
                        vfwf=vfwfgi.at(id);
                        sigui.at(id)=sqrt(ufsq);
                        sigvi.at(id)=sqrt(vfsq);
                        sigwi.at(id)=sqrt(wfsq);
                        ufwfi.at(id)=ufwf;
                        ufvf=0.;
                        ufvfi.at(id)=ufvf;
                        vfwfi.at(id)=vfwf;
                        //mdw 7-25-2005 corrections for axis rotation with non-local mixing
                        ustarij.at(id)=ustarg.at(id);
                        rotufsq(id,u3psq,upwp,v3psq,w3psq,ufsq,ufvf,ufwf,vfsq,vfwf,wfsq);
                        sigui.at(id)=sqrt(u3psq);
                        sigvi.at(id)=sqrt(v3psq);
                        sigwi.at(id)=sqrt(w3psq);
                        upwpi.at(id)=upwp;
                    }
                    sigu=sigui.at(id);
                    sigv=sigvi.at(id);
                    sigw=sigwi.at(id);
                    if(zi.at(k)<=.99*h){
                        eps=pow(ustarij.at(id),3.f)*(1.f-.75f*zi.at(k)*rcl)*pow((1.f-.85f*zi.at(k)/h),(1.5f))/eleff.at(id); // calculate epsilon for grid cell centers
                    }
                    else{
                        eps=pow(ustarij.at(id),3.f)*(1.f-.75f*.99f*h*rcl)*pow((1.f-.85f*.99f),(1.5f))/eleff.at(id); 
                    }
                    
                }
                epsi.at(id)=eps;
                
                // erp June 18, 2009  shader needs sigu sigv
                // convert them to taus and load them onto the shaders
                //tau13= ufwf; tau12 = ufvf; tau23 = vfwf need to be written to shaders
                turbfield<<std::setw(7) << std::setprecision(3)<< std::scientific<<xi.at(i)<<"   "
                         <<yi.at(j)<<"   "<<zi.at(k)<<"   "<< sigu<<"  "<<sigv<<"  "<<sigw<<"  "<<elz.at(id)<<"  "<<
                    eleff.at(id)<<"  "<<eps<<std::endl;
                
                sig[id].u = sigu;   //sigU
                sig[id].v = sigv;   //sigV
                sig[id].w = sigw;   //sigW
                
                float tau11=sigu*sigu;
                float tau22=sigv*sigv;
                float tau33=sigw*sigw;
                float tau13=ustarij.at(id)*ustarij.at(id);
                
                tau[id].t11   = tau11;             //Tau11
                tau[id+1].t22 = tau22;             //Tau22
                tau[id+2].t33 = tau33;             //Tau33
                tau[id+3].t13 = tau13;             //Tau13
                //Make tau's a texture so that they can be visualized as horizontal layers in the domain

                
                dataTwo[texidx]   =  sigu;
                dataTwo[texidx+1] =  sigv;
                dataTwo[texidx+2] =  sigw;
                dataTwo[texidx+3] =  ustarij.at(id);
                
                data3[texidx]   = dutotdxi.at(id);
                data3[texidx+1] = dutotdyi.at(id);
                data3[texidx+2] = dutotdzi.at(id);
                data3[texidx+3] = 0.;

                data4[texidx]   = dzm.at(id);
                data4[texidx+1] = dzp.at(id);
                data4[texidx+2] = dym.at(id);
                data4[texidx+3] = dyp.at(id);
                
                data[texidx]   = dxm.at(id);
                data[texidx+1] = dxp.at(id);
                data[texidx+2] = .00987;
                data[texidx+3] = .00986;
                
                ustar=std::max(ustarij.at(id),0.003f);
                float ustar3=ustar*ustar*ustar;
                dataWind[texidx+3] = 0.;
            }//   lp027
        }//      lp028
    }//         lp029
    
    createTexture(lambda, GL_RGBA32F_ARB, width,height, dataTwo);
    createTexture(windField, GL_RGBA32F_ARB, width, height, dataWind);
    createTexture(tauTex, GL_RGBA32F_ARB, width, height, dataTau);
    createTexture(tau_dz, GL_RGBA32F_ARB, width,height, data);
    
    delete[] dataTwo;
    delete[] dataWind;
    delete[] dataTau;
    delete[] data;
    std::vector<float>dsigwdni,dsigvdni,dsigudni,dupwpdni;
    dsigwdni.resize(nzdz*nydy*nxdx);
    dsigvdni.resize(nzdz*nydy*nxdx);
    dsigudni.resize(nzdz*nydy*nxdx);
    dupwpdni.resize(nzdz*nydy*nxdx);
    ani.resize(nzdz*nydy*nxdx);
    bni.resize(nzdz*nydy*nxdx);
    cni.resize(nzdz*nydy*nxdx);

    std::ofstream dsigwdnOut;
    dsigwdnOut.open("dsigwdn.dat");
    float dsigwdx=0.;
    float dsigwdy=0.;
    float dsigudx=0.;
    float dsigudy=0.;
    float dsigvdx=0.;
    float dsigvdy=0.;
    float dupwpdx=0.;
    float dupwpdy=0.;
    float dsigwdz=0.;
    float dsigvdz=0.;
    float dsigudz=0.;
    float dupwpdz=0.;
    float dsigwdn=0.;
    float dsigvdn=0.;
    float dsigudn=0.;
    float dupwpdn=0.;

    for(int k=0; k<nzdz;k++){//do k=2,nz-1
        for(int j=0;j<nydy;j++){//do j=1,ny-1
            for(int i=0;i<nxdx;i++){//do i=1,nx-1
                int id=k*nxdx*nydy + j*nxdx + i;
                
                int row = k / (numInRow);
                int texidx = row * width * nydy * 4 +
                    j * width * 4 +
                    k % (numInRow) * nxdx * 4 +
                    i * 4;
                
                
                if(cellQuic[id].c != 0){
                    int idim1=k*nxdx*nydy +j*nxdx +(i-1);
                    int idip1=k*nxdx*nydy +j*nxdx +(i+1);
                    
                    int idjm1=k*nxdx*nydy +(j-1)*nxdx +i;
                    int idjp1=k*nxdx*nydy +(j+1)*nxdx +i;
                    
                    int idkm1=(k-1)*nxdx*nydy +j*nxdx +i;
                    int idkp1=(k+1)*nxdx*nydy +j*nxdx +i;
                    
                    if(j<1||j>=nydy-1||i<1||i>=nxdx-1){
                        dsigwdx=0.;
                        dsigwdy=0.;
                        dsigudx=0.;
                        dsigudy=0.;
                        dsigvdx=0.;
                        dsigvdy=0.;
                        dupwpdx=0.;
                        dupwpdy=0.;
                    }
                    //
                    // calculate the gradient of sigma w normal to the flow using a CDD
                    //
                    utot=mag(wind_vel[id]);
                    if(dxm.at(id)>=dx && dxp.at(id)>=dx && i!=0 && i!=nxdx-1){
                        if(idim1<0){
                            dsigwdx=.5*(sigwi.at(idip1)-sigwi.at(id))/dx;
                            dsigvdx=.5*(sigvi.at(idip1)-sigvi.at(id))/dx;
                            dsigudx=.5*(sigui.at(idip1)-sigui.at(id))/dx;
                            dupwpdx=.5*(upwpi.at(idip1)-upwpi.at(id))/dx;
                        }
                        else{
                            dsigwdx=.5*(sigwi.at(idip1)-sigwi.at(idim1))/dx;
                            dsigvdx=.5*(sigvi.at(idip1)-sigvi.at(idim1))/dx;
                            dsigudx=.5*(sigui.at(idip1)-sigui.at(idim1))/dx;
                            dupwpdx=.5*(upwpi.at(idip1)-upwpi.at(idim1))/dx;							  
                        }
                    }
                    else{
                        if(i==1||i==nxdx-1){
                            dsigwdx=0.;
                            dsigvdx=0.;
                            dsigudx=0.;
                            dupwpdx=0.;
                        }
                        else{
                            if(dxm.at(id)<dx && dxp.at(id)>dx){
                                //mdw 11-21-2005 modified if statements to address particle in 3-walled cells
                                dsigwdni.at(id)=(sigwi.at(idip1)-sigwi.at(id))/dx;
                                dsigvdni.at(id)=(sigvi.at(idip1)-sigvi.at(id))/dx;
                                dsigudni.at(id)=(sigui.at(idip1)-sigui.at(id))/dx;
                                dupwpdni.at(id)=(upwpi.at(idip1)-upwpi.at(id))/dx;
                                dsigwdni.at(id)=0.;
                                dsigvdni.at(id)=0.;
                                dsigudni.at(id)=0.;
                                dupwpdni.at(id)=0.;
                                sigwi.at(id)=std::max(sigwi.at(idip1),sigwi.at(id));
                                sigvi.at(id)=std::max(sigvi.at(idip1),sigvi.at(id));
                                sigui.at(id)=std::max(sigui.at(idip1),sigui.at(id));
                                ustarij.at(id)=std::max(ustarij.at(idip1),ustarij.at(id));
                                if(fabs(upwpi.at(id))<fabs(upwpi.at(idip1))){
                                    upwpi.at(id)=upwpi.at(idip1);
                                }
                                else{
                                    upwpi.at(idip1)=upwpi.at(id);
                                }
                            }
                            if(dxp.at(id)<dx && dxm.at(id)>dx){
                                //mdw 11-21-2005 modified if statements to address particle in 3-walled cells
                                if(idim1<0){
                                    dsigwdni.at(id)=0.;//(sigwi.at(id)-sigwi.at(id))/dx;
                                    dsigvdni.at(id)=0.;//(sigvi.at(id)-sigvi.at(id))/dx;
                                    dsigudni.at(id)=0.;//(sigui.at(id)-sigui.at(id))/dx;
                                    dupwpdni.at(id)=0.;//(upwpi.at(id)-upwpi.at(id))/dx;
                                    sigwi.at(id)=std::max(0.f,sigwi.at(id));
                                    sigvi.at(id)=std::max(0.f,sigvi.at(idim1));
                                    sigui.at(id)=std::max(0.f,sigui.at(id));
                                    ustarij.at(id)=std::max(0.f,ustarij.at(id));
                                }
                                else{
                                    dsigwdni.at(id)=(sigwi.at(idim1)-sigwi.at(id))/dx;
                                    dsigvdni.at(id)=(sigvi.at(idim1)-sigvi.at(id))/dx;
                                    dsigudni.at(id)=(sigui.at(idim1)-sigui.at(id))/dx;
                                    dupwpdni.at(id)=(upwpi.at(idim1)-upwpi.at(id))/dx;
                                    sigwi.at(id)=std::max(sigwi.at(idim1),sigwi.at(id));
                                    sigvi.at(id)=std::max(sigvi.at(idim1),sigvi.at(idim1));
                                    sigui.at(id)=std::max(sigui.at(idim1),sigui.at(id));
                                    ustarij.at(id)=std::max(ustarij.at(idim1),ustarij.at(id));
                                }
                                dsigwdni.at(id)=0.f;
                                dsigvdni.at(id)=0.f;
                                dsigudni.at(id)=0.f;
                                dupwpdni.at(id)=0.f;
                                
                                if(idim1<0){
                                    if(fabs(upwpi.at(id))<fabs(0.f)){
                                        upwpi.at(id)=upwpi.at(id);
                                    }
                                }
                                else{
                                    if(fabs(upwpi.at(id))<fabs(upwpi.at(idim1))){
                                        upwpi.at(id)=upwpi.at(idim1);
                                    }
                                    else{
                                        upwpi.at(idim1)=upwpi.at(id);
                                    }
                                }
                            }  
                        }
                    }
                    
                    if(dym.at(id)>=dy && dyp.at(id)>=dy && j!=0 && j!=ny-1){
                        if(idjm1<0){
                            dsigwdy=.5*(sigwi.at(idjp1)-sigwi.at(id))/dy;
                            dsigvdy=.5*(sigvi.at(idjp1)-sigvi.at(id))/dy;
                            dsigudy=.5*(sigui.at(idjp1)-sigui.at(id))/dy;
                            dupwpdy=.5*(upwpi.at(idjp1)-upwpi.at(id))/dy;
                        }
                        else{
                            dsigwdy=.5*(sigwi.at(idjp1)-sigwi.at(idjm1))/dy;
                            dsigvdy=.5*(sigvi.at(idjp1)-sigvi.at(idjm1))/dy;
                            dsigudy=.5*(sigui.at(idjp1)-sigui.at(idjm1))/dy;
                            dupwpdy=.5*(upwpi.at(idjp1)-upwpi.at(idjm1))/dy;
                        }
                    }
                    else{
                        if(j==1||j==ny-1){
                            dsigwdy=0.;
                        }
                        else{
                            if(dym.at(id)<dy && dyp.at(id)>dy){
                                //mdw 11-21-2006 modified if statements to address particle in 3-walled cells
                                dsigwdni.at(id)=(sigwi.at(idjp1)-sigwi.at(id))/dy;
                                dsigvdni.at(id)=(sigvi.at(idjp1)-sigvi.at(id))/dy;
                                dsigudni.at(id)=(sigui.at(idjp1)-sigui.at(id))/dy;
                                dupwpdni.at(id)=(upwpi.at(idjp1)-upwpi.at(id))/dy;
                                dsigwdni.at(id)=0.;
                                dsigvdni.at(id)=0.;
                                dsigudni.at(id)=0.;
                                dupwpdni.at(id)=0.;
                                sigwi.at(id)=std::max(sigwi.at(idjp1),sigwi.at(id));
                                sigvi.at(id)=std::max(sigvi.at(idjp1),sigvi.at(id));
                                sigui.at(id)=std::max(sigui.at(idjp1),sigui.at(id));
                                ustarij.at(id)=std::max(ustarij.at(idjp1),ustarij.at(id));
                                if(fabs(upwpi.at(id))<fabs(upwpi.at(idjp1))){
                                    upwpi.at(id)=upwpi.at(idjp1);
                                }
                                else{
                                    upwpi.at(idjp1)=upwpi.at(id);
                                }
                            }
                            if(dyp.at(id)<dy && dym.at(id)>dy){
                                //mdw 11-21-2005 modified if statements to address particle in 3-walled cells
                                if(idjm1<0){
                                    dsigwdni.at(id)=0.;//(sigwi.at(id)-sigwi.at(id))/dy;
                                    dsigvdni.at(id)=0.;//(sigvi.at(id)-sigvi.at(id))/dy;
                                    dsigudni.at(id)=0.;//(sigui.at(id)-sigui.at(id))/dy;
                                    dupwpdni.at(id)=0.;//(upwpi.at(id)-upwpi.at(id))/dy;
                                    
                                    sigwi.at(id)=std::max(0.f,sigwi.at(id));
                                    sigvi.at(id)=std::max(0.f,sigvi.at(id));
                                    sigui.at(id)=std::max(0.f,sigui.at(id));
                                    ustarij.at(id)=std::max(0.f,ustarij.at(id));
                                }
                                else{
                                    dsigwdni.at(id)=(sigwi.at(idjm1)-sigwi.at(id))/dy;
                                    dsigvdni.at(id)=(sigvi.at(idjm1)-sigvi.at(id))/dy;
                                    dsigudni.at(id)=(sigui.at(idjm1)-sigui.at(id))/dy;
                                    dupwpdni.at(id)=(upwpi.at(idjm1)-upwpi.at(id))/dy;
                                    
                                    sigwi.at(id)=std::max(sigwi.at(idjm1),sigwi.at(id));
                                    sigvi.at(id)=std::max(sigvi.at(idjm1),sigvi.at(id));
                                    sigui.at(id)=std::max(sigui.at(idjm1),sigui.at(id));
                                    ustarij.at(id)=std::max(ustarij.at(idjm1),ustarij.at(id));
                                }
                                
                                dsigwdni.at(id)=0.f;
                                dsigvdni.at(id)=0.f;
                                dsigudni.at(id)=0.f;
                                dupwpdni.at(id)=0.f;
                                if(idjm1<0){
                                    if(fabs(upwpi.at(id))<fabs(0.f)){
                                        upwpi.at(id)=upwpi.at(idjm1);
                                    }
                                }
                                else{
                                    if(fabs(upwpi.at(id))<fabs(upwpi.at(idjm1))){
                                        upwpi.at(id)=upwpi.at(idjm1);
                                    }
                                    else{
                                        upwpi.at(idjm1)=upwpi.at(id);
                                    }
                                }
				
                            }
                        }
                    }
                    if(dzm.at(id)>dz && k!=nzdz-1 && dzp.at(id)>dz){
                        if(idkm1<0){
                            dsigwdz=(sigwi.at(idkp1))/(dz_array.at(k)+0.5+dz_array.at(k+1));
                            dsigvdz=(sigvi.at(idkp1))/(dz_array.at(k)+0.5+dz_array.at(k+1));
                            dsigudz=(sigui.at(idkp1))/(dz_array.at(k)+0.5+dz_array.at(k+1));
                            dupwpdz=(upwpi.at(idkp1))/(dz_array.at(k)+0.5+dz_array.at(k+1));
                        }
                        else{
                            dsigwdz=(sigwi.at(idkp1)-sigwi.at(idkm1))/(dz_array.at(k)+.5*dz_array.at(k-1)+dz_array.at(k+1));
                            dsigvdz=(sigvi.at(idkp1)-sigvi.at(idkm1))/(dz_array.at(k)+.5*dz_array.at(k-1)+dz_array.at(k+1));
                            dsigudz=(sigui.at(idkp1)-sigui.at(idkm1))/(dz_array.at(k)+.5*dz_array.at(k-1)+dz_array.at(k+1));
                            dupwpdz=(upwpi.at(idkp1)-upwpi.at(idkm1))/(dz_array.at(k)+.5*dz_array.at(k-1)+dz_array.at(k+1));
                        }
                    }
                    if(dzm.at(id)<=dz  && k!=nzdz-1 && dzp.at(id)>dz){
                        dsigwdn=(sigwi.at(idkp1)-sigwi.at(id))/(dz_array.at(k)+.5+dz_array.at(k+1));
                        dsigvdn=(sigvi.at(idkp1)-sigvi.at(id))/(dz_array.at(k)+.5+dz_array.at(k+1));
                        dsigudn=(sigui.at(idkp1)-sigui.at(id))/(dz_array.at(k)+.5+dz_array.at(k+1));
                        dupwpdn=(upwpi.at(idkp1)-upwpi.at(id))/(dz_array.at(k)+.5+dz_array.at(k+1));
                        //mdw 9-26-2005 force dsigwdn to be zero near surface
                        dsigwdn=0.;
                        dsigudn=0.;
                        dsigvdn=0.;
                        dupwpdn=0.;
                        sigui.at(id)=std::max(sigui.at(idkp1),sigui.at(id));
                        sigvi.at(id)=std::max(sigvi.at(idkp1),sigvi.at(id));
                        sigwi.at(id)=std::max(sigwi.at(idkp1),sigwi.at(id));
                        ustarij.at(id)=std::max(ustarij.at(idkp1),ustarij.at(id));
                        if(fabs(upwpi.at(id))<fabs(upwpi.at(idkp1))){
                            upwpi.at(id)=upwpi.at(idkp1);
                        }
                        else{
                            upwpi.at(idkp1)=upwpi.at(id);
                        }
                    }
                    if(dzp.at(id)<=dz && k!=nzdz-1 && dzm.at(id)>dz){
                        //mdw 9-26-2005 force dsigwdn to be zero near surface
                        dsigwdn=0.;
                        dsigudn=0.;
                        dsigvdn=0.;
                        dupwpdn=0.;
                        if(idkm1<0){
                            sigui.at(id)=std::max(0.f,sigui.at(id));
                            sigvi.at(id)=std::max(0.f,sigvi.at(id));
                            sigwi.at(id)=std::max(0.f,sigwi.at(id));
                            ustarij.at(id)=std::max(0.f,ustarij.at(id));
                        }
                        else{
                            sigui.at(id)=std::max(sigui.at(idkm1),sigui.at(id));
                            sigvi.at(id)=std::max(sigvi.at(idkm1),sigvi.at(id));
                            sigwi.at(id)=std::max(sigwi.at(idkm1),sigwi.at(id));
                            ustarij.at(id)=std::max(ustarij.at(idkm1),ustarij.at(id));
                        }
                        if(idkm1<0){
                            if(fabs(upwpi.at(id))<0.f){
                                upwpi.at(id)=0.;
                            }
                        }
                        else{
                            if(fabs(upwpi.at(id))<fabs(upwpi.at(idkm1))){
                                upwpi.at(id)=upwpi.at(idkm1);
                            }
                            else{
                                upwpi.at(idkm1)=upwpi.at(id);
                            }
                        }
                        
                    }
                    // erp June 18, 2009: only need diswdni used to calculate tau gradients
                    // export this value to be used, send to shaders
                    if((dxm.at(id)>=dx)&&(dxp.at(id)>=dx)&&(dym.at(id)>=dy)&& 
                       (dyp.at(id)>=dy)&&(dzm.at(id)>=dz)&&(dzp.at(id)>=dz)){
                        dsigwdn=ani.at(id)*dsigwdx+bni.at(id)*dsigwdy+cni.at(id)*dsigwdz;
                        dsigvdn=ani.at(id)*dsigvdx+bni.at(id)*dsigvdy+cni.at(id)*dsigvdz;
                        dsigudn=ani.at(id)*dsigudx+bni.at(id)*dsigudy+cni.at(id)*dsigudz;
                        dupwpdn=ani.at(id)*dupwpdx+bni.at(id)*dupwpdy+cni.at(id)*dupwpdz;
                    }
                    dsigwdni.at(id)=dsigwdn;
                    dsigvdni.at(id)=dsigvdn;
                    dsigudni.at(id)=dsigudn;
                    dupwpdni.at(id)=dupwpdn;
                    
                    // limiting form for near wall circumstances
                    data3[texidx+3] = dsigwdni.at(id);
                }
                dsigwdnOut<<xi.at(i)<<" "<<yi.at(j)<<" "<<zi.at(k)<<" "<<dsigwdni.at(id)<<std::endl;
            }//   lp030
        }//   lp031
    } //  lp032
    //500 line code ends
    createTexture(duvw_dz, GL_RGBA32F_ARB, width,height, data3);
    createTexture(dxyz_wall, GL_RGBA32F_ARB, width,height, data4);
    delete[] data3;
    delete[] data4;
        
  }//end turbinit

#endif
}
