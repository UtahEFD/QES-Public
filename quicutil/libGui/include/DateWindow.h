/* File: DateWindow.h
 * Author: Matthew Overby
 *
 * NOTE:
 * This class isn't meant to be used for your own purposes.
 * If you look at the implementation file you'll see references
 * to options that were never initialized in Gui.cpp.
 * You should really only use it as a reference on how to
 * interact with and use the calendar.
 */

#ifndef SLUI_DATEWINDOW_H
#define SLUI_DATEWINDOW_H

#include "Window.h"
#include "OptionTracker.h"
#include "Calendar.h"
#include <stdio.h>
#include <time.h>

namespace SLUI {

class DateWindow : public Window {

	public:
		static const int Margin = 160;

		DateWindow(int newId, WindowController *winC, OptionTracker *opT, 
			EventTracker *evT, sf::RenderWindow *app);

		~DateWindow();

		void draw();

		void checkEvent();

		void close();

		void open();

		void highlight();

		void update();

		void resizeEvent();

	private:
		Button *closeButton;
		Button *applyButton;
		Button *useToday;
		Button *latButton;
		Button *longButton;
		Button *hourButton;
		Button *minuteButton;

		OptionTracker *m_optTracker;
		Calendar *m_calendar;
};

}

#endif

