#include "TURBWallBuilding.h"

void TURBWallBuilding::defineWalls(URBGeneralData *UGD,TURBGeneralData *TGD) {
    // fill array with cellid  of cutcell cells
    get_cutcell_wall_id(UGD,icellflag_cutcell);
    // fill itrublfag with cutcell flag
    set_cutcell_wall_flag(TGD,iturbflag_cutcell);
  
    // [FM] temporary fix -> use stairstep within the cut-cell 
    get_stairstep_wall_id(UGD,icellflag_building);
    set_stairstep_wall_flag(TGD,iturbflag_stairstep);

    /*
      [FM] temporary fix -> when cut-cell treatement is implmented 
      if(cutcell_wall_id.size()==0) {
      get_stairstep_wall_id(UGD,icellflag_building);
      set_stairstep_wall_flag(TGD,iturbflag_stairstep);
      }else{
      use_cutcell = true;
      }
    */
  
    return;
}


void TURBWallBuilding::setWallsBC(URBGeneralData *UGD,TURBGeneralData *TGD){

    /*
      This function apply the loglow at the wall for building
      Note:
      - only stair-step is implemented
      - need to do: cut-cell for building
    */

    if(!use_cutcell) {
    
        for (size_t id=0; id < stairstep_wall_id.size(); ++id){
            set_loglaw_stairstep_at_id_cc(UGD,TGD,stairstep_wall_id[id],icellflag_building,UGD->z0);
        }
    } else {
        //[FM] temporary fix because the cut-cell are messing with the wall
        //at the terrain
        for(size_t i=0; i < cutcell_wall_id.size(); i++) {
            int id_cc=cutcell_wall_id[i];
            TGD->S11[id_cc]=0.0;
            TGD->S12[id_cc]=0.0;
            TGD->S13[id_cc]=0.0;
            TGD->S22[id_cc]=0.0;
            TGD->S23[id_cc]=0.0;
            TGD->S33[id_cc]=0.0;
            TGD->Lm[id_cc]=0.0;
        }
    } 
}

