 /*
* Source.cuh
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

#ifndef __SOURCE_H__
#define __SOURCE_H__

#include "vector_types.h" 

enum SourceType{POINTSOURCE, LINESOURCE, SPHERESOURCE};

struct SphereSource
{
  float3 ori;
  float rad;
} ;

struct LineSource
{
  float3 start;
  float3 end;
};

union SourceInfo
{
  SphereSource sph;
  LineSource ln;
  float3 pt;
};
 

struct Source
{
  SourceType type; 
  SourceInfo info;
  float speed;
};

#endif