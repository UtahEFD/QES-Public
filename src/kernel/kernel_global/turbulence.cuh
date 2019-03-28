/*
 * turbulence.cuh
 * This file is part of CUDAPLUME
 *
 * Copyright (C) 2012 - Alex
 *
 * CUDAPLUME is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * CUDAPLUME is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with CUDAPLUME. If not, see <http://www.gnu.org/licenses/>.
 */

 #ifndef __TURBULENCE_CUH_H__
 #define __TURBULENCE_CUH_H__ 
 
 
struct turbulence
{
  float CoEps;
  int cellType;
  float3 windData;
  //copy CoEps end//////////////////////////////////////////////////////////////
  float3 eigVal; 
  float3  ka0; 
  float3  g2nd; 
////////////////  matrix 9////////////////
  float3  eigVec1;
  float3  eigVec2;
  float3  eigVec3;
  float3  eigVecInv1;
  float3  eigVecInv2;
  float3  eigVecInv3;
  float3  lam1;
  float3  lam2;
  float3  lam3;
//////////////// matrix6 ////////////////
  float3  sig1;
  float3  sig2;
  float3  taudx1;
  float3  taudx2; 
  float3  taudy1;
  float3  taudy2; 
  float3  taudz1;
  float3  taudz2;
};


__host__
inline std::ostream &operator<<(std::ostream& os, const float4 &f4)
{
  os << " " << f4.x << ' ' << f4.y << ' ' << f4.z<< ' ' << f4.w <<std::endl;
//   os << " " << f4.x << ' ' << f4.y << ' ' << f4.z << ' ' << f4.w << " "<<std::endl;
  return os;
}
__host__
inline std::ostream &operator<<(std::ostream& os, const float3 &f3)
{
  os << " [ " << f3.x << ' ' << f3.y << ' ' << f3.z << " ] ";
  return os;
}

__host__
inline std::ostream &operator<<(std::ostream& os, const turbulence &turb)
{
  os << "{ CoEps" << turb.CoEps << " cellType" << turb.cellType << " windData" << turb.windData 
     << " eigVal" << turb.eigVal << " ka0" << turb.ka0 << " g2nd" << turb.g2nd << " eigVec1" << turb.eigVec1 
     << " eigVec2" << turb.eigVec2 << " eigVec3" << turb.eigVec3 << " eigVecInv1" << turb.eigVecInv1 
     << " eigVecInv2" << turb.eigVecInv2 << " eigVecInv3" << turb.eigVecInv3 << " lam1" << turb.lam1 
     << " lam2" << turb.lam2 << " lam3" << turb.lam3 << " sig1" << turb.sig1 << " sig2" << turb.sig2 
     << " taudx1" << turb.taudx1 << " taudx2" << turb.taudx2 << " taudy1" << turb.taudy1 << " taudy2" 
     << turb.taudy2 << " taudz1" << turb.taudz1 << " taudz2" << turb.taudz2 << " }"<<std::endl;
  return os;
}
 
 #endif /* __TURBULENCE_CUH_H__ */
 
