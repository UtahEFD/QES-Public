#pragma once

/*
 *Used to store information about intersections
 *Can add other information about the BVH node it hits as needed 
 */

#ifndef HR_H
#define HR_H
class HitRecord{
  private:
   void* hitNode;
   float hitDist;  //distance from ray origin to hit point 
  public:
   
   HitRecord(void* hitNode, float hitDist);
   void* getHitNode();
   float getHitDist();
   
};

#endif
