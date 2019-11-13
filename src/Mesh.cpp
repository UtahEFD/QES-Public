#include "Mesh.h"

float Mesh::getHeight(float x, float y)
{
   return tris->heightToTri(x,y);
}

vector<float> Mesh::calculateMixingLength(int dimX, int dimY,int dimZ, float dx, float dy, float dz, const vector<int> &icellflag){

   vector<float> mixingLengthList(dimX*dimY*dimZ);

   for(int k = 0; k< dimz - 1; k++) {
      for(int j = 0; j < dimY - 1; j++){
         for(int i = 0; i < dimX -1; i++){

            //calculate icell index
            int icell_idx = i + j*(dimX-1) + k*(dimY-1) * (dimX-1);

            if(icellflag[icell_idx] == 1){

               SphereDirections sd = new SphereDirections();
               float mixLength = std::numeric_limits<float>::infinity();

               //ray's origin = cell's center
               Ray ray = new Ray((i+0.5)*dx, (j+0.5)*dy, (k+0.5)*dz);

               HitRecord* hit;
               BVH* nextBVH;

               //for all possible directions, determine the distance
               for(int m = 0, m < sd.getNumDirVec(), m++){

                  ray.setNextDir(ds.getNextDir());
                  nextBVH* = tris;
                  hit = NULL;

                  //stop if nextBVH is a leaf node or if it didn't intersect anything
                  while(!nextBVH.getIsLeaf() && nextBVH != NULL){
                     HitRecord *checkLeft = nextBVH.getLeftBox().rayBoxIntersect();
                     HitRecord *checkRight = nextBVH.getRightBox().rayBoxIntersect();

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
            }
           std:cout<<"Mixing length for this cell is"<<mixLength<<std::endl;
            //add to list of vectors
            mixingLengthList.push_back(mixLength);
         }
      }
   }
   return mixingLengthList;
}
