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
 
 
#include "Source.cuh"

typedef unsigned int uint;

struct Building
{
  float3 lowCorner;
  float3 highCorner;
};
 
// simulation parameters
struct ConstParams {  
    uint3 domain;
    float3 origin; 
    
    Building building; 
    Source source; 
    float particleRadius;   
};


//
static __constant__ ConstParams g_params;  

#endif
