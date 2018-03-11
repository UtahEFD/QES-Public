/*	
*	Class Definition
*/
#ifndef SLUI_SAMPLE_CLICKWORLD
#define SLUI_SAMPLE_CLICKWORLD
#include "Gui.h"
namespace SLUI {
	class SampleClickworld : public Gui {
	public:
		SampleClickworld(int h, int w);
		int display();
	private:
		void generateCubes();
		void cellFunction(std::vector<unsigned int> selected);
	};
}

#endif
