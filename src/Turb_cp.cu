/*
 * Turb_cp.cu
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

 #ifndef __TURB_CP_H__
 #define __TURB_CP_H__
 
#include <thrust/host_vector.h>
// #include <thrust/device_vector.h>
#include <thrust/copy.h>
#include <thrust/fill.h>
#include <thrust/sequence.h>
// #include <thrust/transform.h>
// #include "thrust/for_each.h"
// #include "thrust/iterator/zip_iterator.h"
#include "bw/Eulerian.h"
#include "bw/Dispersion.h"
#include "GL_funs.hpp" 
#include "kernel/kernel_global/turbulence.cuh" 
 

 
uint3 gridSize;

struct copyTurbData1
{
  template <typename Tuple>
  __host__ 
  void operator()(Tuple t)
  {
   //  Eulerian::wind wd = thrust::get<1>(t); 
    // double f1 = thrust::get<1>(t);
    thrust::get<0>(t).CoEps = thrust::get<1>(t); // make_float4(wd.u, wd.v, wd.w, 0.f); 
    thrust::get<0>(t).cellType = thrust::get<2>(t).c; // make_float4(wd.u, wd.v, wd.w, 0.f); 
     Eulerian::wind wd = thrust::get<3>(t);  
    thrust::get<0>(t).windData = make_float3(wd.u, wd.v, wd.w); // make_float4(wd.u, wd.v, wd.w, 0.f); 
    Eulerian::vec3 vec = thrust::get<4>(t); 
    thrust::get<0>(t).ka0 = make_float3(vec.e11, vec.e21, vec.e31); 
    vec = thrust::get<5>(t); 
    thrust::get<0>(t).g2nd = make_float3(vec.e11, vec.e21, vec.e31); 
    
    Eulerian::matrix9 matr = thrust::get<6>(t); 
    thrust::get<0>(t).eigVec1 = make_float3(matr.e11, matr.e12, matr.e13 ); //make_float3(matr.e11, matr.e12, matr.e13); 
    thrust::get<0>(t).eigVec2 = make_float3(matr.e21, matr.e22, matr.e23 ); //make_float3(matr.e21, matr.e22, matr.e23); 
    thrust::get<0>(t).eigVec3 = make_float3(matr.e31, matr.e32, matr.e33 ); 
    
    matr = thrust::get<7>(t); 
    thrust::get<0>(t).eigVecInv1 = make_float3(matr.e11, matr.e12, matr.e13 ); //make_float3(matr.e11, matr.e12, matr.e13); 
    thrust::get<0>(t).eigVecInv2 = make_float3(matr.e21, matr.e22, matr.e23 ); //make_float3(matr.e21, matr.e22, matr.e23); 
    thrust::get<0>(t).eigVecInv3 = make_float3(matr.e31, matr.e32, matr.e33 );  
     
  }
};

struct copyTurbData2
{
  template <typename Tuple>
  __host__ 
  void operator()(Tuple t)
  {  
    Eulerian::matrix6 
    matr = thrust::get<1>(t); 
    thrust::get<0>(t).taudx1 = make_float3(matr.e11, matr.e12, matr.e13);//  make_float3(matr.e11, matr.e12, matr.e13); 
    thrust::get<0>(t).taudx2 = make_float3(matr.e22, matr.e23, matr.e33); 
    
    matr = thrust::get<2>(t); 
    thrust::get<0>(t).taudy1 = make_float3(matr.e11, matr.e12, matr.e13);//  make_float3(matr.e11, matr.e12, matr.e13); 
    thrust::get<0>(t).taudy2 = make_float3(matr.e22, matr.e23, matr.e33); 
    
    matr = thrust::get<3>(t); 
    thrust::get<0>(t).taudz1 = make_float3(matr.e11, matr.e12, matr.e13);//  make_float3(matr.e11, matr.e12, matr.e13); 
    thrust::get<0>(t).taudz2 = make_float3(matr.e22, matr.e23, matr.e33);  
    
    matr = thrust::get<4>(t); 
    thrust::get<0>(t).sig1 = make_float3(matr.e11, matr.e12, matr.e13);//  make_float3(matr.e11, matr.e12, matr.e13); 
    thrust::get<0>(t).sig2 = make_float3(matr.e22, matr.e23, matr.e33); 
    
    Eulerian::diagonal dia = thrust::get<5>(t); 
    thrust::get<0>(t).eigVal = make_float3(dia.e11, dia.e22, dia.e33); 
  }
};

void turb_cp_2ndEdition(util &utl, Eulerian &eul, dispersion &disp, 
	     thrust::host_vector<turbulence> &turb)
{
  thrust::for_each(
    thrust::make_zip_iterator(
      thrust::make_tuple(turb.begin(), eul.CoEps.begin(), eul.CellType.begin(), eul.windVec.begin(),
			eul.ka0.begin(), eul.g2nd.begin(), eul.eigVec.begin(), eul.eigVecInv.begin() 
      )
    ),
    thrust::make_zip_iterator(
      thrust::make_tuple(turb.end(), eul.CoEps.end(), eul.CellType.end(),eul.windVec.end(),
			 eul.ka0.end(), eul.g2nd.end(), eul.eigVec.end(), eul.eigVecInv.end()  
      )
    ),
    copyTurbData1()
  ); 
  
  thrust::for_each(
    thrust::make_zip_iterator(
      thrust::make_tuple(turb.begin(), eul.taudx.begin(), eul.taudy.begin(), eul.taudz.begin(), eul.sig.begin(),
			 eul.eigVal.begin()
      )
    ),
    thrust::make_zip_iterator(
      thrust::make_tuple(turb.end(), eul.taudx.end(), eul.taudy.end(), eul.taudz.end(), eul.sig.begin(),
			 eul.eigVal.end()
      )
    ),
    copyTurbData2()
  ); 
  
}
struct makefloat3
{
  template <typename Tuple>
  __host__ 
  void operator()(Tuple t)
  {
     float x = thrust::get<0>(t);
     float y = thrust::get<1>(t);
     float z = thrust::get<2>(t);
    thrust::get<3>(t) = make_float3(x, y, z); 
  }
}; 
struct copyWinddata
{
  template <typename Tuple>
  __host__ 
  void operator()(Tuple t)
  {
     Eulerian::wind wd = thrust::get<0>(t); 
    thrust::get<1>(t) = make_float4(wd.u, wd.v, wd.w, 0.f); 
//     thrust::get<1>(t) = make_float4(2.f, 0.f, 0.f, 0.f); 
  }
};
struct copyEigVal
{
  template <typename Tuple>
  __host__ 
  void operator()(Tuple t)
  {
     Eulerian::diagonal diag = thrust::get<0>(t); 
    thrust::get<1>(t) = make_float4(diag.e11, diag.e22, diag.e33,0.f);//  make_float3(diag.e11, diag.e22, diag.e33); 
  } 
};
struct copyka0
{
  template <typename Tuple>
  __host__ 
  void operator()(Tuple t)
  {
     Eulerian::vec3 vec = thrust::get<0>(t); 
    thrust::get<1>(t) = make_float4(vec.e11, vec.e21, vec.e31, 0.f); //make_float3(vec.e11, vec.e21, vec.e31); 
  }
};
struct copyMatrix9 
{
  template <typename Tuple>
  __host__ 
  void operator()(Tuple t)
  {
     Eulerian::matrix9 matr = thrust::get<0>(t); 
    thrust::get<1>(t) = make_float4(matr.e11, matr.e12, matr.e13, 0.f); //make_float3(matr.e11, matr.e12, matr.e13); 
    thrust::get<2>(t) = make_float4(matr.e21, matr.e22, matr.e23, 0.f); //make_float3(matr.e21, matr.e22, matr.e23); 
    thrust::get<3>(t) = make_float4(matr.e31, matr.e32, matr.e33, 0.f); //make_float3(matr.e31, matr.e32, matr.e33); 
  }
}; 
struct copyMatrix6
{
  template <typename Tuple>
  __host__ 
  void operator()(Tuple t)
  {
     Eulerian::matrix6 matr = thrust::get<0>(t); 
    thrust::get<1>(t) = make_float4(matr.e11, matr.e12, matr.e13, 0.f);//  make_float3(matr.e11, matr.e12, matr.e13); 
    thrust::get<2>(t) = make_float4(matr.e22, matr.e23, matr.e33, 0.f);//// make_float3(matr.e22, matr.e23, matr.e33);  
  }
};
struct copyPrime
{
  template <typename Tuple>
  __host__ 
  void operator()(Tuple t)
  {
     dispersion::matrix vec3 = thrust::get<0>(t); 
    thrust::get<1>(t) = make_float4(vec3.x, vec3.y, vec3.y, 0.f);//  make_float3(matr.e11, matr.e12, matr.e13); 
   }
};
struct makeBuildings
{
  template <typename Tuple>
  __host__ 
  void operator()(Tuple t)
  {
     float x = thrust::get<0>(t);
     float y = thrust::get<1>(t);
     float z = thrust::get<2>(t); 
     float w = thrust::get<4>(t);
     float h = thrust::get<5>(t);
     float l = thrust::get<6>(t); 
    
    thrust::get<3>(t) = make_float3(x, y - (w/2.f), z);
    thrust::get<7>(t) = make_float3(x +l, y + (w/2.f), z+h);
  }
};

 
void turb_cp(util &utl, Eulerian &eul, dispersion &disp, 
	     thrust::host_vector<float4> &windData, 
	     thrust::host_vector<float4> &eigVal, 
	     thrust::host_vector<float4> &ka0, 
	     thrust::host_vector<float4> &g2nd, 
      ////////////////  matrix 9////////////////
	     thrust::host_vector<float4> &eigVec1,
	     thrust::host_vector<float4> &eigVec2,
	     thrust::host_vector<float4> &eigVec3,
	     thrust::host_vector<float4> &eigVecInv1,
	     thrust::host_vector<float4> &eigVecInv2,
	     thrust::host_vector<float4> &eigVecInv3,
	     thrust::host_vector<float4> &lam1,
	     thrust::host_vector<float4> &lam2,
	     thrust::host_vector<float4> &lam3,
      //////////////// matrix6 ////////////////
	     thrust::host_vector<float4> &sig1,
	     thrust::host_vector<float4> &sig2,
	     thrust::host_vector<float4> &taudx1,
	     thrust::host_vector<float4> &taudx2, 
	     thrust::host_vector<float4> &taudy1,
	     thrust::host_vector<float4> &taudy2, 
	     thrust::host_vector<float4> &taudz1,
	     thrust::host_vector<float4> &taudz2
	    )
{ 
     
 
  domain = make_uint3(utl.nx, utl.ny, utl.nz);
  std::cout << "utl.nx " << utl.nx<<"  utl.ny " << utl.ny<<"  utl.nz " << utl.nz << "\n";
  gridSize = make_uint3(domain.x, domain.y, domain.z); 
  origin = make_float3(0.f, 0.f, 0.f);  
  
 /*
  thrust::host_vector<float> hxfo(utl.xfo.begin(), utl.xfo.end());*/ 
  int size = utl.xfo.size();   
  lowCorners.resize(size); highCorners.resize(size);
  
//copy Buidling data  
  thrust::for_each(
    thrust::make_zip_iterator(
      thrust::make_tuple(utl.xfo.begin(), utl.yfo.begin(), utl.zfo.begin(), lowCorners.begin(),
			 utl.hgt.begin(), utl.wth.begin(), utl.len.begin(), highCorners.begin() )),
    thrust::make_zip_iterator( 
      thrust::make_tuple(utl.xfo.end(), utl.yfo.end(), utl.zfo.end(), lowCorners.end(),
			 utl.hgt.end(), utl.wth.end(), utl.len.end(), highCorners.end() )),
    makeBuildings()
  ); 
//copy Buidling end//////////////////////////////////////////////////////////////
  
//   float4 pointSource = make_float4(utl.xSrc, utl.ySrc, utl.zSrc, utl.rSrc);
//   for(size_t i = 0; i < hyfo.size(); i++)
//     std::cout << "D[" << i << "] = " << lowCorners[i].x<<" "<< lowCorners[i].y<< " "<<lowCorners[i].z<<  std::endl; 
//   return 1;  
/*
 *copy prime here
 *
 */  
/* prime.resize(disp.prime.size());
  thrust::for_each(
    thrust::make_zip_iterator(
      thrust::make_tuple(disp.prime.begin(), prime.begin() )),
    thrust::make_zip_iterator(
      thrust::make_tuple(disp.prime.end(), prime.end() )),
    copyPrime()
  ); */ 
 
//copy windData start//////////////////////////////////////////////////////////////
  windData.resize(eul.windVec.size());
  thrust::for_each(
    thrust::make_zip_iterator(
      thrust::make_tuple(eul.windVec.begin(), windData.begin() )),
    thrust::make_zip_iterator(
      thrust::make_tuple(eul.windVec.end(), windData.end() )),
    copyWinddata()
  ); 
//copy windData end//////////////////////////////////////////////////////////////
    
//copy CellType start////////////////////////////////////////////////// ////////////
//   thrust::host_vector<int> CellType(eul.CellType.begin(), eul.CellType.end()); 
//copy CellType end//////////////////////////////////////////////////////////////

    
//copy eig_val start//////////////////////////////////////////////////////////////
//   thrust::host_vector<float3> eigVal;
  eigVal.resize(eul.eigVal.size());
  
  thrust::for_each( 
    thrust::make_zip_iterator(
      thrust::make_tuple(eul.eigVal.begin(), eigVal.begin() )),
    thrust::make_zip_iterator(
      thrust::make_tuple(eul.eigVal.end(), eigVal.end() )),
    copyEigVal()
  ); 
  //std::cout << "windVec.size() " << eul.eigVal.size()<<"  eul.size(): " << eul.nx*eul.ny*eul.nz << "\n";
  eul.eigVal.clear();
   
//copy eig_val end//////////////////////////////////////////////////////////////
  
//copy ka0 g2nd start//////////////////////////////////////////////////////////////
//   thrust::host_vector<float3> ka0;
  ka0.resize(eul.ka0.size());
//   thrust::host_vector<float3> g2nd;
  g2nd.resize(eul.g2nd.size());
  
  thrust::for_each(
    thrust::make_zip_iterator(
      thrust::make_tuple(eul.ka0.begin(), ka0.begin() )),
    thrust::make_zip_iterator(
      thrust::make_tuple(eul.ka0.end(), ka0.end() )),
    copyka0()
  ); 
  eul.ka0.clear();
   
  thrust::for_each(
    thrust::make_zip_iterator(
      thrust::make_tuple(eul.g2nd.begin(), g2nd.begin() )),
    thrust::make_zip_iterator(
      thrust::make_tuple(eul.g2nd.end(), g2nd.end() )),
    copyka0()
  );  
  eul.g2nd.clear();
//copy ka0 g2nd end////////////////////////////////////////////////////////////// 
  
//copy eigVec start//////////////////////////////////////////////////////////////
//copy eigVec1 includes e11, e12, e13 in matrix, similarly eigVec2 and eigVec3//////////////////////////////////////////
//   thrust::host_vector<float3> eigVec1;
//   thrust::host_vector<float3> eigVec2;
//   thrust::host_vector<float3> eigVec3;
  eigVec1.resize(eul.eigVec.size());
  eigVec2.resize(eul.eigVec.size());
  eigVec3.resize(eul.eigVec.size()); 
  
  thrust::for_each(
    thrust::make_zip_iterator(
      thrust::make_tuple(eul.eigVec.begin(), eigVec1.begin(), eigVec2.begin(), eigVec3.begin() )),
    thrust::make_zip_iterator(
      thrust::make_tuple(eul.eigVec.end(), eigVec1.end(), eigVec2.end(), eigVec3.end() )),
    copyMatrix9()
  );  
//     for(size_t i = 0; i < eigVec1.size(); i++){
//     std::cout << eigVec1[i].x<<" "<< eigVec2[i].y<< " "<<eigVec3[i].z<<  std::endl; 
//     std::cout << eul.eigVec[i].e11<<" "<< eul.eigVec[i].e22<< " "<<eul.eigVec[i].e33<<  std::endl; 
//     }
  eul.eigVec.clear();
//  
//copy eigVec end//////////////////////////////////////////////////////////////
  
//copy eigVecInv start//////////////////////////////////////////////////////////////
//copy eigVecInv1 includes e11, e12, e13 in matrix, similarly eigVecInv2 and eigVecInv3//////////////////////////////////////////
//   thrust::host_vector<float3> eigVecInv1;
//   thrust::host_vector<float3> eigVecInv2;
//   thrust::host_vector<float3> eigVecInv3;
  eigVecInv1.resize(eul.eigVecInv.size());
  eigVecInv2.resize(eul.eigVecInv.size());
  eigVecInv3.resize(eul.eigVecInv.size()); 
  
  thrust::for_each(
    thrust::make_zip_iterator(
      thrust::make_tuple(eul.eigVecInv.begin(), eigVecInv1.begin(), eigVecInv2.begin(), eigVecInv3.begin() )),
    thrust::make_zip_iterator(
      thrust::make_tuple(eul.eigVecInv.end(), eigVecInv1.end(), eigVecInv2.end(), eigVecInv3.end() )),
    copyMatrix9()
  );  
//     for(size_t i = 0; i < eigVecInv1.size(); i++){
//     std::cout << eigVecInv1[i].x<<" "<< eigVecInv2[i].y<< " "<<eigVecInv3[i].z<<  std::endl; 
//     std::cout << eul.eigVecInv[i].e11<<" "<< eul.eigVecInv[i].e22<< " "<<eul.eigVecInv[i].e33<<  std::endl; 
//     }
  eul.eigVecInv.clear();
//  
//copy eigVecInv end//////////////////////////////////////////////////////////////
  
//copy lam start//////////////////////////////////////////////////////////////
//copy lam1 includes lam11, lam12, lam13 in matrix, similarly lam2 and lam3//////////////////////////////////////////
//   thrust::host_vector<float3> lam1;
//   thrust::host_vector<float3> lam2;
//   thrust::host_vector<float3> lam3;
  lam1.resize(eul.lam.size());
  lam2.resize(eul.lam.size());
  lam3.resize(eul.lam.size()); 
  
  thrust::for_each(
    thrust::make_zip_iterator(
      thrust::make_tuple(eul.lam.begin(), lam1.begin(), lam2.begin(), lam3.begin() )),
    thrust::make_zip_iterator(
      thrust::make_tuple(eul.lam.end(), lam1.end(), lam2.end(), lam3.end() )),
    copyMatrix9()
  );  
  eul.lam.clear();
//     for(size_t i = 0; i < lam1.size(); i++){
//     std::cout << lam1[i].x<<" "<< lam2[i].y<< " "<<lam3[i].z<<  std::endl; 
//     std::cout << eul.lam[i].e11<<" "<< eul.lam[i].e22<< " "<<eul.lam[i].e33<<  std::endl; 
//     }
//  
//copy lam end//////////////////////////////////////////////////////////////
  
//copy sig start//////////////////////////////////////////////////////////////
//copy sig includes e11,e12,e13 //sig1 has e11 e12 e13
		      //  e22,e23//sig2 has e22 e23 e33
		      //      e33 in matrix/////////////////////////////////////////
//   thrust::host_vector<float3> sig1;
//   thrust::host_vector<float3> sig2; 
  sig1.resize(eul.sig.size());
  sig2.resize(eul.sig.size()); 
  
  thrust::for_each(
    thrust::make_zip_iterator(
      thrust::make_tuple(eul.sig.begin(), sig1.begin(), sig2.begin() )),
    thrust::make_zip_iterator(
      thrust::make_tuple(eul.sig.end(), sig1.end(), sig2.end() )),
    copyMatrix6()
  );   
//   for(int i=0; i<eul.sig.size(); i++)
//   { 
//     std::cout << sig1[i].x<<" "<< sig2[i].x<< " "<<sig2[i].z<<  std::endl; 
//     std::cout << eul.sig[i].e11<<" "<< eul.sig[i].e22<< " "<<eul.sig[i].e33<<  std::endl; 
//   }
  eul.sig.clear();
//  
//copy sig end//////////////////////////////////////////////////////////////
  
//copy taudx start//////////////////////////////////////////////////////////////
//copy taudx includes e11,e12,e13 //taudx1 has e11 e12 e13
		      //  e22,e23//taudx2 has e22 e23 e33
		      //      e33 in matrix/////////////////////////////////////////
//   thrust::host_vector<float3> taudx1;
//   thrust::host_vector<float3> taudx2; 
  taudx1.resize(eul.taudx.size());
  taudx2.resize(eul.taudx.size()); 
  
  thrust::for_each(
    thrust::make_zip_iterator(
      thrust::make_tuple(eul.taudx.begin(), taudx1.begin(), taudx2.begin() )),
    thrust::make_zip_iterator(
      thrust::make_tuple(eul.taudx.end(), taudx1.end(), taudx2.end() )),
    copyMatrix6()
  );   
  eul.taudx.clear();
//  
//copy taudx end//////////////////////////////////////////////////////////////
//   
//copy taudy start//////////////////////////////////////////////////////////////
//copy taudy includes e11,e12,e13 //taudy1 has e11 e12 e13
		      //  e22,e23//taudy2 has e22 e23 e33
		      //      e33 in matrix/////////////////////////////////////////
//   thrust::host_vector<float3> taudy1;
//   thrust::host_vector<float3> taudy2; 
  taudy1.resize(eul.taudy.size());
  taudy2.resize(eul.taudy.size()); 
  
  thrust::for_each(
    thrust::make_zip_iterator(
      thrust::make_tuple(eul.taudy.begin(), taudy1.begin(), taudy2.begin() )),
    thrust::make_zip_iterator(
      thrust::make_tuple(eul.taudy.end(), taudy1.end(), taudy2.end() )),
    copyMatrix6()
  );   
  eul.taudy.clear();
//  
//copy taudy end//////////////////////////////////////////////////////////////
//   
//copy taudz start//////////////////////////////////////////////////////////////
//copy taudz includes e11,e12,e13 //taudz1 has e11 e12 e13
		      //  e22,e23//taudz2 has e22 e23 e33
		      //      e33 in matrix/////////////////////////////////////////
//   thrust::host_vector<float3> taudz1;
//   thrust::host_vector<float3> taudz2; 
  taudz1.resize(eul.taudz.size());
  taudz2.resize(eul.taudz.size()); 
  
  thrust::for_each(
    thrust::make_zip_iterator(
      thrust::make_tuple(eul.taudz.begin(), taudz1.begin(), taudz2.begin() )),
    thrust::make_zip_iterator(
      thrust::make_tuple(eul.taudz.end(), taudz1.end(), taudz2.end() )),
    copyMatrix6()
  );   
  eul.taudz.clear();
//  
//copy taudz end//////////////////////////////////////////////////////////////
// 
}
 #endif /* __TURB_CP_H__ */
 
