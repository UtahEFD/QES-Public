#pragma once

/*
 *Basic definition of a ray 
 */
#ifndef RAY_H
#define RAY_H

#include "Vector3.h"
#include <vector>
class Ray{
  private:
      float origin_x, origin_y, origin_z;
      Vector3<float> dirVec;
      
  public:
      Ray(float origin_x, float origin_y, float origin_z, Vector3<float> dirVec);

      Ray(float origin_x, float origin_y, float origin_z);

      float getOriginX();
      float getOriginY();
      float getOriginZ();
      Vector3<float> getDirection();
      void setDir(Vector3<float> dir);
};


#endif
