/*	
*	Class Definition
*/
#ifndef SIVELAB_SAMPLE_OSG
#define SIVELAB_SAMPLE_OSG
#include "Gui.h"
#include <osgViewer/Viewer>
#include <osgDB/ReadFile>
namespace siveLAB {
	class SampleOSG : public Gui {
	public:
		SampleOSG(int h, int w);
		int display();
	private:
	};
}

#endif
