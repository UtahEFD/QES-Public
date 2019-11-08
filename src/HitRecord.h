/*
 *Used to store information about intersections
 *Can add other information about the BVH node it hits as needed 
 */


class HitRecord(){
  private:
   BVH* hitNode;
   float hitDist;  //distance from ray origin to hit point 
  public:
   HitRecord(BVH* hitNode, hitDist){
      this.hitNode = hitNode;
      this.hitDist = hitDist;
   }

   BVH* getHitNode(){return hitnode;}
   float getHitDist(){return hitDist;}
   
};
