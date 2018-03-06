/* File: Calendar.h
 * Author: Matthew Overby
 *
 * TODO:
 * Needs comments!
 * A more generic calendar class would be extremely useful.
 * However, a functional implementation of how this class
 * can be used is in the DateWindow class.
 */

#ifndef SLUI_CALENDAR_H
#define SLUI_CALENDAR_H

#include <cstdio>
#include "Widget.h"
#include "EventTracker.h"
#include "OptionTracker.h"

namespace SLUI {

// TODO DateButton should derive from widget
struct DateButton {

	DateButton();
	DateButton( bool act, std::string day, sf::RenderWindow *app );
	void setPosition( float x, float y );
	void draw();
	void highlight();
	bool isMouseOver();

	float m_posX, m_width;
	float m_posY, m_height;
	
	int num_date;
	bool active, alwaysActive;
	bool mouseOver;
	sf::String m_date;
	sf::Shape m_highlight;
	sf::Shape m_background;

	sf::RenderWindow *m_App;
};

struct CalDate {
	int month; // 1-12
	int year; // 1980+
	int day; // 1-7
};

class Calendar : public Widget {

	public:
		Calendar( EventTracker *evt, sf::RenderWindow *app );

		void setDate( int y, int m, int d );

		CalDate getDate();

		void draw();

		void checkEvent();

		void highlight();

		void setPosition( float x, float y );

		void setToDefault();

		bool active;
		bool updated;

	private:
		void initCalButtons();

		int getDayOfWeek(int year, int month, int day);

		void generateCalData();

		EventTracker *m_eventTracker;
		sf::Shape m_inactiveBG;
		DateButton m_dates[7][7];

		std::map< int, std::string > month_names;
		sf::String month;
		sf::String year;
		int curr_month; // 1-12
		int curr_year; // 1980+
		int curr_day; // 1-7

		sf::Shape month_incIcon;
		sf::Shape month_decIcon;
		sf::Shape month_decLab;
		sf::Shape month_incLab;

		sf::Shape year_incIcon;
		sf::Shape year_decIcon;
		sf::Shape year_decLab;
		sf::Shape year_incLab;

		static const int lrButtonWidth = 10;
		static const int lrButtonHeight = 20;
		static const int lrButtonSpace = 5;
		int monthIconSpacing;
};

}

#endif


