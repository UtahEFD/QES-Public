/*
 * reflection.cu
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

#ifndef __REFLECTION_CU__
#define __REFLECTION_CU__

#include "../kernel_global/matrix3X3.cu"
#include "../kernel_global/ConstParams.cuh" 

__host__ __device__ 
inline float dist(const float3 &posA, const float3 &posB)
{
  return sqrtf( (posA.x-posB.x)*(posA.x-posB.x) + 
		(posA.y-posB.y)*(posA.y-posB.y) + 
		(posA.z-posB.z)*(posA.z-posB.z) );
}

__host__ __device__ 
inline float3 getNormal(const float3 &posOld, const float3 &posNew, float &reflection_longth)
{//hard coding cell size, gona remove it when test done
  if(floor(posNew.x) == floor(posOld.x) + 1.f)//1.f should be the cell width
  {
    reflection_longth = (posNew.x - floor(posNew.x)) * dist(posOld, posNew) / (posNew.x - posOld.x);
    return make_float3(-1.f, 0.f, 0.f);
  }
  if(floor(posNew.x) == floor(posOld.x) - 1.f)
  {
    reflection_longth = (floor(posOld.x) - posNew.x) * dist(posOld, posNew) / (posOld.x - posNew.x);
    return make_float3(1.f,  0.f, 0.f);
  }
  
  if(floor(posNew.y) == floor(posOld.y) + 1.f)//1.f should be the cell length
  {
    reflection_longth = (posNew.y - floor(posNew.y)) * dist(posOld, posNew) / (posNew.y - posOld.y);
    return make_float3(0.f, -1.f, 0.f);
  }
  if(floor(posNew.y) == floor(posOld.y) - 1.f)
  {
    reflection_longth = (floor(posOld.y) - posNew.y) * dist(posOld, posNew) / (posOld.y - posNew.y);
    return make_float3(0.f,  1.f, 0.f);
  }
  
  if(floor(posNew.z) == floor(posOld.z) + 1.f)//1.f should be the cell height
  {
    reflection_longth = (posNew.z - floor(posNew.z)) * dist(posOld, posNew) / (posNew.z - posOld.z);
    return make_float3(0.f, 0.f, -1.f);
  }
  if(floor(posNew.z) == floor(posOld.z) - 1.f)
  {
    reflection_longth = (floor(posOld.z) - posNew.z) * dist(posOld, posNew) / (posOld.z - posNew.z);
    return make_float3(0.f, 0.f,  1.f);
  }
  
  return make_float3(0.f, 0.f,  0.f);// for the bug case if particle moves more than one cell size
}

__host__ __device__ 
inline float3 getReflectPos(const float3 &posOld, const float3 &posNew, const float& radius)
{//hard coding cell size, gona remove it when test done
  float3 pos = posNew;
  if(floor(posNew.x) == floor(posOld.x) + 1.f)//1.f should be the cell width
  { 
    pos.x = 2*floor(posNew.x) - posNew.x - radius;
    //return make_float3(2*floor(posNew.x) - posNew.x, posNew.y, posNew.z);
  }
  if(floor(posNew.x) == floor(posOld.x) - 1.f)
  { 
    pos.x = 2*floor(posOld.x) - posNew.x + radius;
    //return make_float3(2*floor(posOld.x) - posNew.x, posNew.y, posNew.z);
  }
  
  if(floor(posNew.y) == floor(posOld.y) + 1.f)//1.f should be the cell length
  {
    pos.y = 2*floor(posNew.y) - posNew.y - radius;
    //return make_float3(posNew.x, 2*floor(posNew.y) - posNew.y, posNew.z);
  }
  if(floor(posNew.y) == floor(posOld.y) - 1.f)
  {
    pos.y = 2*floor(posOld.y) - posNew.y + radius;
    //return make_float3(posNew.x, 2*floor(posOld.y) - posNew.y, posNew.z);
  }
  
  if(floor(posNew.z) == floor(posOld.z) + 1.f)//1.f should be the cell height
  {
    pos.z = 2*floor(posNew.z) - posNew.z - radius;
    //return make_float3(posNew.x, posNew.y, 2*floor(posNew.z) - posNew.z);
  }
  if(floor(posNew.z) == floor(posOld.z) - 1.f)
  {
     pos.z = 2*floor(posOld.z) - posNew.z + radius;
    //return make_float3(posNew.x, posNew.y, 2*floor(posOld.z) - posNew.z);
  }
  
  return pos;
  //return make_float3(0.f, 0.f,  0.f);// for the bug case if particle moves more than one cell size
}

// __host__ __device__ 
// inline float3 reflection(const float3 &posOld, const float3 &posNew)
// {
//   float reflection_longth;
//   float3 normal_dir   = normalize(getNormal(posOld, posNew, reflection_longth));
//   float3 incoming_dir = normalize(posNew - posOld);
//   float3 reflect_dir  = incoming_dir - 2.f * dot(incoming_dir, normal_dir) * normal_dir;
//   float3 lerpON = lerp(posNew, posOld, reflection_longth);
//   //return make_float3(reflection_longth,reflection_longth,reflection_longth);//
//   return lerpON;// + reflect_dir*reflection_longth;
// }


#endif /* __REFLECTION_CU__ */

