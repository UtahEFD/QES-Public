#include "Mesh.h"

float Mesh::getHeight(float x, float y)
{
   return tris->heightToTri(x,y);
}

void Mesh::calculateMixingLength(){

   //add a reference to list of all fluid cells in domain

   //TODO: reference to fluid cells
   for(int i=0, i< /*all fluid cells in domiain*/, i++) {

      //Create a SphereDirections obj. that will contain all the
      //directions to iterate over
      SphereDirections sd = new SphereDirections();
      float mixLength = std::numeric_limits<float>::infinity();

      BVH* nextBVH = tris;

      //TODO: find out how cells are actually called
      Cell currCell = cell[i];

      //TODO: might have to calc the cell's midpoint (or center of mass)
      Ray ray = new Ray(currCell.centerX, currCell.centerY, currCell.centerZ);

      HitRecord* hit=NULL;
      //for all possible directions, determine the distance
      for(int j=0, j < sd.getNumDirVec(), j++){
         ray = new Ray(currCell.center X, currCell.centerY, currCell.centerZ, ds.getNextDir());
         ray.setNextDir(ds.getNextDir());

         //stop if nextBVH is a leaf node or if it didn't intersect anything
         while(!nextBVH.getIsLeaf() && nextBVH != NULL){
            HitRecord* checkLeft = nextBVH.getLeftBox().rayBoxIntersect();
            HitRecord* checkRight = nextBVH.getRightBox().rayBoxIntersect();

            if(checkLeft != NULL && checkRight == NULL){
               nextBVH = nextBVH.getLeftBox();
            }else if(checkLeft == NULL && checkRight != NULL){
               nextBVH = nextBVH.getRightBox();
            }else if (checkLeft != NULL && checkRight != NULL){
               if(checkLeft.getHitDist() < checkRight.getHitDist()){
                  nextBVH = nextBVH.getLeftBox();
               }else{
                  nextBVH = nextBVH.getRightBox();
               }
            }else{
               std::cout<<"Ray is outside of bounds. Did not hit any of the bounding boxes"<<endl;
               nextBVH == NULL;
            }
         }
         //At this point BVH should be a leaf or NULL
         //if nextBVH is a leaf, it should check if it's the smallest
         //dist currently
         if(nextBVH != NULL){
            hit = nextBVH.rayTriIntersect();
            if(nextBVH != NULL & hit != NULL){
               if(hit.getHitDist() < mixLength){
                  mixLength = hit.getDist();
               }
            }
         }
      }

      //TODO: find proper call
      cell.mixingLength = mixLength;
     std:cout<<"Mixing length for this cell is"<<maxLength<<std::endl;

   } //fluid loop end

}
