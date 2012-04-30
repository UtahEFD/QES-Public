/**
*	Author: Andrew Larson <lars2865@d.umn.edu>
* Reason: Better typing for the conversion to C++ from Fortran.
*/

#ifndef CELLTYPE_FIELD
#define CELLTYPE_FIELD

#include <vector_types.h>

#include "celltype.h"
#include "cellDims.h"
#include "domainDims.h"

namespace QUIC
{
	typedef struct celltypeField
	{
		CellType* c;
		
		uint3 dmn;
		cellDims cll;
	} celltypeField;
}

#endif

