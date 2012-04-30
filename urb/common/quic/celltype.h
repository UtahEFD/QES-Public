#ifndef CELLTYPE
#define CELLTYPE 1

namespace QUIC
{
	enum CellType 
	{
		SOLID        =  0, // Building 
		FLUID        =  1, 
		UPWIND       =  2, 
		ROOFTOP      =  3, 
		NEARWAKE     =  4,
		FARWAKE      =  5,
		CANYON       =  6,
		VEGETATION   =  8,
		INTERSECTION =  9,
		GARAGE       = 10
	};
}

#endif
