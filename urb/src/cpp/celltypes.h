/**
*	Author: Andrew Larson <lars2865@d.umn.edu>
* Reason: Better typing for the conversion to C++ from Fortran.
*/

#ifndef CELLTYPE_MATRIX
#define CELLTYPE_MATRIX

#include <vector_types.h>

#include "celltype.h"

namespace QUIC
{
	typedef struct celltypes
	{
		CellType* c;
		
		int3 dim;
	} celltypes;
}

#endif

