#pragma once

/*
 *Basic definition of a ray 
 */

#include <vector>
#include "Vec3D.h"

class Ray{
  private:
      float origin_x, origin_y, origin_z;
      Vector3<float> dirVec;

  public:
      Ray(float origin_x, float origin_y, float origin_z, Vector3<float> dirVec);

      Ray(float origin_x, float origin_y, float origin_z);

    ~Ray() {}

      float getOriginX();
      float getOriginY();
      float getOriginZ();
      Vector3<float> getDirection();

      void setDir(const Vector3<float> &dir);
};


#endif
