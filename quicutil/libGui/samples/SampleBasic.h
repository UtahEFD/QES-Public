/*	
*	Class Definition
*/
#ifndef SLUI_SAMPLE_BASIC
#define SLUI_SAMPLE_BASIC
#include "Gui.h"
namespace SLUI {
	class SampleBasic : public Gui {
	public:
		SampleBasic(int h, int w);
		int display();
	private:
		void sampleFunc();
		void drawCube();
	};
}

#endif
