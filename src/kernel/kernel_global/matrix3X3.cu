/*
 * matrix3X3.cu
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

#ifndef __MATRIX3X3_CU__
#define __MATRIX3X3_CU__  


////////////////////////matrix3X3 start/////////////////////////////////////////////////
struct matrix3X3
{
  float e11, e12, e13,
	e21, e22, e23,
	e31, e32, e33;
};
	// convert vec1,vec2, vec3 to matrix3X3
__device__ __host__
inline matrix3X3 make_matrix3X3(const float3 &vec1, const float3 &vec2, const float3 &vec3)
{
  matrix3X3 mat;
  mat.e11 = vec1.x; mat.e12 = vec1.y; mat.e13 = vec1.z; 
  mat.e21 = vec2.x; mat.e22 = vec2.y; mat.e23 = vec2.z; 
  mat.e31 = vec3.x; mat.e32 = vec3.y; mat.e33 = vec3.z; 
  
  return mat;
}
	      /////matrix3X3  * vec3X1/////
__device__ __host__
inline float3 mat_X_vec(const matrix3X3 &mat, const float3 &vec3)
{
  return 
    make_float3(mat.e11*vec3.x + mat.e12*vec3.y + mat.e13*vec3.z,
		mat.e21*vec3.x + mat.e22*vec3.y + mat.e23*vec3.z,
		mat.e31*vec3.x + mat.e32*vec3.y + mat.e33*vec3.z);
}
__device__ __host__
inline float3 operator*(const matrix3X3 &mat, const float3 &vec3)
{
  return 
    make_float3(mat.e11*vec3.x + mat.e12*vec3.y + mat.e13*vec3.z,
		mat.e21*vec3.x + mat.e22*vec3.y + mat.e23*vec3.z,
		mat.e31*vec3.x + mat.e32*vec3.y + mat.e33*vec3.z);
}

/////////////////////matrix3X3 end/////////////////////////////////////////// 
 

//----------------overwrite == and !=  start/ 
__host__  __device__ 
inline bool isANum(const float3 &f31)
{
  return (f31.x==f31.x) && (f31.y==f31.y) && (f31.z==f31.z);
}

__host__  __device__ 
inline bool isANum(const float4 &f41)
{
  return (f41.x==f41.x) && (f41.y==f41.y) && (f41.z==f41.z) && (f41.w==f41.w);
}

__host__  __device__ 
inline bool operator>(const float3 &f31, const float3 &f32)
{
  return (f31.x>f32.x) && (f31.y>f32.y) && (f31.z>f32.z);
}

__host__  __device__ 
inline bool operator==(const float3 &f31, const float3 &f32)
{
  return (f31.x==f32.x) && (f31.y==f32.y) && (f31.z==f32.z);
}

__host__  __device__ 
inline bool operator!=(const float3 &f31, const float3 &f32)
{
  return !(f31==f32);
}

__host__  __device__ 
inline bool operator==(const float4 &f41, const float4 &f42)
{
  return (f41.x==f42.x) && (f41.y==f42.y) && (f41.z==f42.z) && (f41.w==f42.w);
}

__host__  __device__ 
inline bool operator!=(const float4 &f41, const float4 &f42)
{
  return !(f41==f42);
}  
__host__  __device__ 
inline bool operator>(const float4 &f41, const float4 &f42)
{
  return (f41.x>f42.x) || (f41.y>f42.y) || (f41.z>f42.z) || (f41.w>f42.w);
}  
 

//////////////////////////same sign test start/////////////////////////////////////////////   
__host__  __device__ 
inline bool isSameSign(const float &x, const float &y)
{
  if(x>=0 && y>=0)
    return true;
  if(x<=0 && y<=0)
    return true;
   
  return false;
  //return fabs(x+y) == fabs(x) + fabs(y);    
}
__host__  __device__ 
inline bool isAllSameSign(const float3 &f31, const float3 &f32)
{
  return isSameSign(f31.x, f32.x) && isSameSign(f31.y, f32.y) && isSameSign(f31.z, f32.z);    
}
__host__  __device__ 
inline bool isHasSameSign(const float3 &f31, const float3 &f32)
{
  return isSameSign(f31.x, f32.x) || isSameSign(f31.y, f32.y) || isSameSign(f31.z, f32.z);    
}
//////////////////////////same sign test start/////////////////////////////////////////////  
__host__  __device__ 
inline float3 exp(const float3 &f31)
{
  return make_float3(exp(f31.x), exp(f31.y), exp(f31.z));
}  

__host__  __device__ 
inline float4 make_float4(const float3 &f31)
{
  return make_float4(f31.x, f31.y, f31.z, 0.f);
} 

__host__  __device__ 
inline float4 make_float4(const float &f1, const float3 &f31)
{
  return make_float4(f1, f31.x, f31.y, f31.z);
} 

 
//---------------------------overwrite == and != end///-


//-----------------------get the minmum value start///-
__host__  __device__ 
inline float minf4(const float &x, const float &y, const float &z, const float &w)
{
  float t = x < y ? x : y;  
  t = t < z ? t : z;    
  return t < w ? t : w;
}   


__host__  __device__ 
inline float minf4(const float3 &f3, const float &w)
{     
  return minf4(f3.x, f3.y, f3.z, w);
}
  
__host__  __device__ 
inline float minf3(const float &x, const float &y, const float &z)
{
  float t = x < y ? x : y;  
  return t < z ? t : z;    
}
__host__  __device__ 
inline float minf3(const float3 &f3)
{ 
  return minf3(f3.x, f3.y, f3.z);    
}
  
__host__  __device__ 
inline float minf2(const float &x, const float &y)
{
  return x < y ? x : y;    
}
__device__ __host__
inline float3 sqrtf(const float3 &f3)
{
  return make_float3(sqrtf(f3.x), sqrtf(f3.y), sqrtf(f3.z));
}

//////////////////////////get the minmum value end/////////////////////////////////////////////  

 


#endif /* __MATRIX3X3_CU__ */

 
