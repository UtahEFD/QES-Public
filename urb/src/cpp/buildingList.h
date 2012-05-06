/**
*	Author: Andrew Larson <lars2865@d.umn.edu>
* Borned: May 7th, 2010
* Reason: Advanced operations for building lists needed.
* Source: My brain.
*/

#ifndef URBBUILDING_LIST
#define URBBUILDING_LIST 1

#include <cfloat>
#include <algorithm>

#include "building.h"
#include "../util/index3D.h"

// TODO fix this for memory leaks, not thinking about it now.
// TODO make vegetations a different strucuture??

namespace QUIC
{
  using namespace std;

  class urbBuildingList : public vector<urbBuilding>
  {	
	
  public:
	
    urbBuildingList() : vector<urbBuilding>() {}
    virtual ~urbBuildingList() {}

    virtual void sort();

    // Given i, j and k, finds the building at that index in the domain.
    // Returns NULL if no building found.
    urbBuilding& getBuildingAt(index3D const& ndx3d);
    urbBuilding& getBuildingAt(int const& i, int const& j, int const& k);

    virtual float averageHeight() const;

    float3 getBuildingLocation(int const& bndx) const;
    float3 getBuildingDimensions(int const& bndx) const;
	
    double closestBuilding(index3D const& loc) const;

  protected:
		
    virtual void swapBuildings(urbBuilding& b1, urbBuilding& b2);
  };
}

#endif

