#pragma once

/*
 *Used to store information about intersections
 *Can add other information about the BVH node it hits as needed 
 */

#ifndef HR_H
#define HR_H

#include <limits>

class HitRecord{
  private:
   bool isHit;
   void* hitNode;
   float hitDist;  //distance from ray origin to hit point 
  public:

   HitRecord(void* hitNode, bool isHit);
   HitRecord(void* hitNode, bool isHit, float hitDist);
   
   void* getHitNode();
   float getHitDist();
   bool getIsHit();
};

#endif
