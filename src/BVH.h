/****************************************************************************
 * Copyright (c) 2021 University of Utah
 * Copyright (c) 2021 University of Minnesota Duluth
 *
 * Copyright (c) 2021 Behnam Bozorgmehr
 * Copyright (c) 2021 Jeremy A. Gibbs
 * Copyright (c) 2021 Fabien Margairaz
 * Copyright (c) 2021 Eric R. Pardyjak
 * Copyright (c) 2021 Zachary Patterson
 * Copyright (c) 2021 Rob Stoll
 * Copyright (c) 2021 Pete Willemsen
 *
 * This file is part of QES-Winds
 *
 * GPL-3.0 License
 *
 * QES-Winds is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, version 3 of the License.
 *
 * QES-Winds is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with QES-Winds. If not, see <https://www.gnu.org/licenses/>.
 ****************************************************************************/

/** @file BVH.h */

#pragma once

#include <vector>

#include "Triangle.h"
#include "HitRecord.h"

#define GETMIN(x,y) ( (x) < (y) ? (x) : (y))
#define GETMAX(x,y) ( (x) > (y) ? (x) : (y))


#include "Ray.h"

using std::vector;

/**
 * @class BVH
 * @brief Bounding Volume Hierarchy data structure.
 *
 * Organizes Triangle objects spacially allowing for fast access based on location.
 *
 * @sa Triangle
 */
class BVH
{
private:
   BVH* leftBox;  /**< left child box */
   BVH* rightBox; /**< right child box */

   bool isLeaf;   /**< true if leaf of the tree */
   Triangle* tri; /**< Triangle put in the hierarchy */


   /**
    * Sorts Bounding Boxes.
    *
    * @param list the list of bounding boxes that should be sorted
    * @param type identifier for what dimension the boxes are being sorted by
    */
   static void mergeSort(std::vector<BVH *>& list, const int type);

public:
   float xmin, xmax, ymin, ymax, zmin, zmax;

   /**
    * Creates a bounding box encasing two child bounding boxes.
    *
    * @param l left child box
    * @param r right child box
    */
   BVH(BVH* l, BVH* r);

   /**
    * Creates a bounding box encasing a Triangle, this marks the box
    * as a leaf meaning it is at the bottom of the tree.
    *
    * @param t the Triangle to be put in the heirarchy
    */
   BVH(Triangle* t);

   /**
    * Creates a bounding volume hierarchy from a list of bounding boxes.
    * Height is used to determine the spacial ordering.
    *
    * @param m list of bounding boxes
    * @param height current depth in the tree
    */
   BVH(std::vector<BVH *> m, int height);

   /**
    * Takes a point in the xy-plane and finds what Triangle is directly above
    * it. It returns how many meters exist between the point and the ground.
    *
    * @param x x-position
    * @param y y-position
    * @return distance from the point to the triangle directly above it
    */
   float heightToTri(float x, float y);

   /**
    * Creates a BVH structure from a vector of models.
    *
    * @param tris list of Triangle objects that will be placed in the structure
    */
   static BVH* createBVH(const std::vector<Triangle*> &tris);

   /**
    * Takes a 3D ray and determines if it intersects this BVH
    * node's bounding box.
    *
    * @param ray ray to potential hit
    * @return true if a hit is found; false otherwise
    */
   bool rayBoxIntersect(const Ray &ray);

   /**
    * Determines if the ray hit the expected parameters and
    * updates the HitRecord with info on the hit.
    *
    * @param ray ray to potential hit
    * @param rec the HitRecord to be updated with hit details
    * @return true if hit is found; false otherwise
    */
   bool rayHit(const Ray &ray, HitRecord& rec);
};

inline bool BVH::rayBoxIntersect(const Ray &ray){
   float originX = ray.getOriginX();
   float originY = ray.getOriginY();
   float originZ = ray.getOriginZ();
   // Vector3<float> dir = ray.getDirection();
   Vec3D dir = ray.getDirection();

   float tMinX, tMaxX, tMinY, tMaxY, tMinZ, tMaxZ;

   //calc tMinX and tMaxX
   float aX = 1/dir[0];
   if(aX >= 0){
      tMinX = aX*(xmin - originX);
      tMaxX= aX*(xmax - originX);
   }else{
      tMinX = aX*(xmax - originX);
      tMaxX = aX*(xmin - originX);
   }

   //calc tMinY and tMaxY
   float aY = 1/dir[1];
   if(aY >= 0){
      tMinY = aY*(ymin - originY);
      tMaxY = aY*(ymax - originY);
   }else{
      tMinY = aY*(ymax - originY);
      tMaxY = aY*(ymin - originY);
   }

   //calc tMinZ and tMaxZ
   float aZ = 1/dir[2];
   if(aZ >= 0){
      tMinZ = aZ*(zmin - originZ);
      tMaxZ = aZ*(zmax - originZ);
   }else{
      tMinZ = aZ*(zmax - originZ);
      tMaxZ = aZ*(zmin - originZ);
   }


   if(tMinX > tMaxY || tMinY > tMaxX ||
      tMinY > tMaxZ || tMinZ > tMaxY ||
      tMinZ > tMaxX || tMinX > tMaxZ){
      return false;
   }else{
      return true;
   }
}


inline bool BVH::rayHit(const Ray &ray, HitRecord& rec)
{
   if(!rayBoxIntersect(ray)) {
      return false;
   }

   if(isLeaf) {

      //if(!tri){
      //std::cout<<"Must not be a tri."<<std::endl;  // this is an error!
      // }

      float t0 = 0.00001;
      //float t1 = std::numeric_limits<float>::infinity();
      float t1 = 999999999.9;

      return tri->rayTriangleIntersect(ray, rec, t0, t1);

   } else {  //not a leaf node

      bool leftHit = false, rightHit = false;
      HitRecord lrec, rrec;

      //prob don't need to check left due to BVH construction
      //left should never be null for anything except a leaf node
      if(leftBox){
         leftHit = leftBox->rayHit(ray, lrec);
      }

      if(rightBox){
         rightHit = rightBox->rayHit(ray, rrec);
      }

      // std::cout << "Left Hit: " << leftHit << ", Right Hit: " << rightHit << std::endl;

      if(leftHit && rightHit){
         if(lrec.hitDist < rrec.hitDist){
            rec = lrec;
         }else{
            rec = rrec;
         }
         return true;
      }else if(leftHit){
         rec = lrec;
         return true;
      }else if(rightHit){
         rec = rrec;
         return true;
      }else{
         // std::cout << "false only else case" << std::endl;
         return false;
      }
   }
}
