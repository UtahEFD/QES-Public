#ifndef FLOAT3_OPERATIONS
#define FLOAT3_OPERATIONS 1

#include <cmath>
#include <vector_types.h>

inline float3 operator+(float3 const& a, float3 const& b)
{
  float3 result = {a.x + b.x, a.y + b.y, a.z + b.z};
  return result;
}

inline float3 operator-(float3 const& a, float3 const& b)
{
  float3 result = {a.x - b.x, a.y - b.y, a.z - b.z};
  return result;
}

inline float operator*(float3 const& a, float3 const& b)
{
  return a.x*b.x + a.y*b.y + a.z*b.z;
}

inline float3 operator*(float3 const& a, float const& s)
{
  float3 result = {a.x*s, a.y*s, a.z*s};
  return result;
}

inline float3 operator*(float const& s, float3 const& a)
{
  return a*s;
}

__device__ __host__ inline float mag(float3 const& v)
{
  return sqrt(v.x*v.x + v.y*v.y + v.z*v.z);
}

__device__ __host__ inline float3 normal(float3 const& v)
{
  float v_lngth = mag(v);
  float3 result = {v.x / v_lngth, v.y / v_lngth, v.z / v_lngth};
  return result;
}

__device__ __host__ inline void setEqual(float4& a, float3 const& b)
{
  a.x = b.x;
  a.y = b.y;
  a.z = b.z;
  // a.w --> don't do.
}

__device__ __host__ inline void setEqual(float3& a, float4 const& b)
{
  a.x = b.x;
  a.y = b.y;
  a.z = b.z;
  // a.w --> doesn't have.
}

#endif
