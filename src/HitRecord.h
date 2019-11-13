/*
 *Used to store information about intersections
 *Can add other information about the BVH node it hits as needed 
 */

#ifndef HR_H
#define HR_H
class HitRecord{
  private:
   //BVH* hitNode;
   void* hitNode;
   float hitDist;  //distance from ray origin to hit point 
  public:
   //HitRecord(BVH* hitNode, hitDist);
   HitRecord(void* hitNode, hitDist);
   //BVH* getHitNode();
   void* getHitNode();
   float getHitDist();
   
};

#endif
