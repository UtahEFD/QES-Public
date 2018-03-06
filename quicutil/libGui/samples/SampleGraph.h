/*	
*	Class Definition
*/
#ifndef SLUI_SAMPLE_GRAPH
#define SLUI_SAMPLE_GRAPH
#include "Gui.h"
namespace SLUI {
	class SampleGraph : public Gui {
	public:
		SampleGraph(int h, int w);
		int display();
	private:
		void createGraph1();
		void createGraph2();
		void createGraph3();
	};
}

#endif
