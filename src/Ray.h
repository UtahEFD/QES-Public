#pragma once

/*
 *Basic definition of a ray 
 */

#include "Vec3D.h"

class Ray{
  private:
    float origin_x, origin_y, origin_z;
    Vec3D dirVec;

  public:
    Ray(float o_x, float o_y, float o_z, Vec3D &dVec) 
        : origin_x(o_x), origin_y(o_y), origin_z(o_z), dirVec(dVec)
    {
    }
    
    Ray(float o_x, float o_y, float o_z)
        : origin_x(o_x), origin_y(o_y), origin_z(o_z)
    {
        dirVec[0] = 0.0;;
        dirVec[1] = 0.0;
        dirVec[2] = 1.0;
    }
    
    ~Ray() {}

    float getOriginX() const {return origin_x;}
    float getOriginY() const {return origin_y;}
    float getOriginZ() const {return origin_z;};

    Vec3D getDirection() const {return dirVec;}

    void setDir(const Vec3D &dir) {
        dirVec[0] = dir[0];
        dirVec[1] = dir[1];
        dirVec[2] = dir[2];
    }
};
