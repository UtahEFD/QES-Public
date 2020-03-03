#include "TURBWallTerrain.h"

void TURBWallTerrain::defineWalls(URBGeneralData *UGD,TURBGeneralData *TGD) {
  // fill array with cellid  of cutcell cells
  get_cutcell_wall_id(UGD,icellflag_cutcell);
  // fill itrublfag with cutcell flag
  set_cutcell_wall_flag(TGD,iturbflag_cutcell);

  // [FM] temporary fix -> use stairstep within the cut-cell
  get_stairstep_wall_id(UGD,icellflag_terrain);
  set_stairstep_wall_flag(TGD,iturbflag_stairstep);
  
  /*
    [FM] temporary fix -> when cut-cell treatement is implmented 
    if(cutcell_wall_id.size()==0) {
    get_stairstep_wall_id(UGD,icellflag_terrain);
    set_stairstep_wall_flag(TGD,iturbflag_stairstep);
    } else {
    use_cutcell = true; 
    }
  */
  
  return;
}


void TURBWallTerrain::setWallsBC(URBGeneralData *UGD,TURBGeneralData *TGD){

  /*
    This function apply the loglow at the wall for terrain
    Note:
    - only stair-step is implemented
    - need to do: cut-cell for terrain
  */
  int nx = UGD->nx;
  int ny = UGD->ny;
  
  if(!use_cutcell) {
    float z0=0.01;
    for (size_t id=0; id < stairstep_wall_id.size(); ++id){
      
      int id_cc=stairstep_wall_id[id];
      int k = (int)(id_cc / ((nx-1)*(ny-1)));
      int j = (int)((id_cc - k*(nx-1)*(ny-1))/(nx-1));
      int i = id_cc -  j*(nx-1) - k*(nx-1)*(ny-1);
      int id_2d = i + j*nx;

      // set_loglaw_stairstep_at_id_cc need z0 at the cell center
      // -> z0 is averaged over the 4 face of the cell
      z0=0.25*(UGD->z0_domain_u.at(id_2d)+UGD->z0_domain_u.at(id_2d+1)+
               UGD->z0_domain_v.at(id_2d)+UGD->z0_domain_v.at(id_2d+nx));

      set_loglaw_stairstep_at_id_cc(UGD,TGD,id_cc,icellflag_terrain,z0);
    }
  } else {
    // [FM] temporary fix because the cut-cell are messing with the wall
    // at the terrain
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

