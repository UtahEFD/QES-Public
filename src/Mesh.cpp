#include "Mesh.h"

float Mesh::getHeight(float x, float y)
{
   return tris->heightToTri(x,y);
}

void Mesh::calculateMixingLength(){

   //add a reference to list of all fluid cells in domain

   //TODO:
   for(int i=0, i< /*all fluid cells in domiain*/, i++) {

      //Create a SphericalDirections obj. that will contain all the
      //dir to iterate over
      SphericalDirections sd = new SphericalDirections();
      float mixLength = std::numeric_limits<float>::infinity();
      nextBVH = tris;
      //TODO: find out how cells are actually called
      Cell currCell = cell[i];
      Ray ray = new Ray(currCell.centerX, currCell.centerY, currCell.centerZ);
      HitRecord* hit=NULL;
      for(int j=0, j < sd.getNumDirVec(), j++){
         ray = new Ray(currCell.center X, currCell.centerY, currCell.centerZ, ds.getNextDir());
         ray.setNextDir(ds.getNextDir());

         while(!nextBVH.getIsLeaf() && nextBVH == NULL){
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
               cout<<"Ray is outside of bounds. Hit any of the bounds."<<endl;
               nextBVH == NULL;
            }
         }
         //At this point BVH should be a leaf or non-NULL
         hit = nextBVH.rayTriIntersect();
         if(nextBVH != NULL & hit != NULL){
            if(hit.getHitDist() < mixLength){
               mixLength = hit.getDist();
            }
         }
      }

      cell.mixingLength = mixLength;
     std:cout<<"Mixing length for this cell is"<<maxLength<<std::endl;

   } //fluid loop end

}
