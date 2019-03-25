 /*
* ConstParams.cuh
* This file is part of CUDAPLUME
*
* Copyright (C) 2012 - Alex 
*
*
* CUDAPLUME is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
* Lesser General Public License for more details.
*
* You should have received a copy of the GNU Lesser General Public License
* along with CUDAPLUME. If not, see <http://www.gnu.org/licenses/>.
*/

 
#ifndef __CONSTANTPARAMS_H__
#define __CONSTANTPARAMS_H__

// #define USE_TEX 0
// 
// #if USE_TEX
// #define FETCH(t, i) tex1Dfetch(t##Tex, i)
// #else
// #define FETCH(t, i) t[i]
// #endif

#include "vector_types.h"
#include "Source.cuh"

typedef unsigned int uint;

struct Building
{
  float3 lowCorner;
  float3 highCorner;
};

// struct Source
// {
//   enum SourceType{LINE, SPHERE};
//   
//   float3 sourceOrigin;
//   SourceType type;
// };


// simulation parameters
struct ConstParams {
//     bool isFirsttime; 
//     uint currentNumBodies; //currentnumBodies
    
    float3 domain;
    float3 origin;
    
    Building building;//buildings in the scene;
//     float3 sourceOrigin;
    Source source;
    
    
    float particleRadius;  
    
//     //cell// texture memory alreay include the Cell info
//     uint numCells;//how many cells
//     float3 cellSize;  
//     float globalDamping;
};

#endif
