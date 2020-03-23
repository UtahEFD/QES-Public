#include "PolyBuilding.h"

// These take care of the circular reference
#include "URBInputData.h"
#include "URBGeneralData.h"
#include "TURBGeneralData.h"

void PolyBuilding::NonLocalMixing (const URBInputData* UID, URBGeneralData* UGD, TURBGeneralData* TGD) 
{
    
    const float sigUOrg   = 2.0;
    const float sigVOrg   = 2.0;
    const float sigWOrg   = 1.3;
    const float sigUConst = sigUOrg;
    const float sigVConst = sigVOrg;
    const float sigWConst = sigWOrg;
    
    float ustar,ustarH,ustarV;
    int id;
    
    /*
    if(ustarH>ustarV){
        ustar=ustarH;
        Lm=wth.at(ibld);
    }
    else{
        ustar=ustarV;
        Lm=0.75*hgt.at(ibld);
    }
    */
    
    float ustar2 = ustar*ustar;
    TGD->tau11[id] = sigUConst*sigUConst*ustar2;
    TGD->tau22[id] = sigVConst*sigVConst*ustar2;
    TGD->tau33[id] = sigWConst*sigWConst*ustar2;
    if(ustarH>ustarV) {
        ustar=ustarH;
        float ustar2 = ustar*ustar;
        
        TGD->tau11[id] = sigUConst*sigUConst*ustar2;
        TGD->tau22[id] = sigVConst*sigVConst*ustar2;
        TGD->tau33[id] = sigWConst*sigWConst*ustar2;
        
        TGD->tau12[id] = ustar2; //to finish (projection with wind dir)
        TGD->tau23[id] = ustar2; //to finish (projection with wind dir)
        TGD->tau13[id] = 0;
        
        TGD->Lm[id] = 0; //effective width
        
    } else {
        ustar=ustarV;
        float ustar2 = ustar*ustar;
        
        TGD->tau11[id] = sigUConst*sigUConst*ustar2;
        TGD->tau22[id] = sigVConst*sigVConst*ustar2;
        TGD->tau33[id] = sigWConst*sigWConst*ustar2;
        
        TGD->tau12[id] = 0;
        TGD->tau23[id] = -ustar2;//to finish (projection with wind dir)
        TGD->tau13[id] = -ustar2;//to finish (projection with wind dir)
        
        TGD->Lm[id] = 0; //height
    }

    TGD->tke[id]=0.5*(TGD->tau11[id]+TGD->tau22[id]+TGD->tau33[id]);
    TGD->CoEps[id]=5.7* pow(ustar,3.0)/(TGD->Lm[id]); 

    return;
    
}
