#include "Mesh.h"

float Mesh::getHeight(float x, float y)
{
   return tris->heightToTri(x,y);
}

void Mesh::calculateMixingLength(){

   //TODO:
   for(/*For all domain cells that are fluid*/) {

      //Create a SphericalDirections obj. that will contain all the
      //dir to iterate over
      SphericalDirections sd = new SphericalDirections();
      float mixLength;

      for(/*all rays in the set, test for intersection*/){



         if(intersects){
            if(intersectLength < mixLength){
               mixLength = intersectLength;
            }
         }
         cell.mixingLength = mixLength;
        std:cout<<"Mixing length for this cell is"<<maxLength<<std::endl;
      }
   } //fluid loop end

}
