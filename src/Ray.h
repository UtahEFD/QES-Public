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
      Vector3<float> inverse_dir;  //inverse of dirVec
      std::vector<int> signs;
  public:
      Ray(float origin_x, float origin_y, float origin_z, Vector3<float> dirVec);

      Ray(float origin_x, float origin_y, float origin_z);

      float getOriginX();
      float getOriginY();
      float getOriginZ();
      Vector3<float> getDirection();
      Vector3<float> getInverseDir();
      std::vector<int> getSigns();
      void setDir(Vector3<float> dir);
};


#endif
