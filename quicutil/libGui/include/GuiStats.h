/* File: GuiStats.h
 * Author: Matthew Overby
 *
 * NOTE:
 * This class isn't meant to be used for your own purposes.
 * If you look at the implementation file you'll see references
 * to options that were never initialized in Gui.cpp.
 * You should really only use it as a reference on how to
 * set up your own stats widget.
 */

#ifndef SLUI_GUISTATS_H
#define SLUI_GUISTATS_H

#include "Widget.h"

namespace SLUI {

struct FPS {

	FPS();
	unsigned int getFPS() const { return m_fps; }
	void update();
	unsigned int m_fps, m_frame;
	sf::Clock m_clock;
};

class GuiStats : public Widget {

	public:
		/** @brief Constructor
		*/
		GuiStats(sf::RenderWindow*);

		/** @brief Updates the stats position with the new camera position
		*/
		void updateCameraPos(float posX, float posY, float posZ);

		/** @brief Updates the sun position
		*/
		void updateSunPos(float altitude, float azimuth, float angle);

		/** @brief Updates the latitude and longitude
		*/
		void updateLatLon(float latitude, float longitude);

		/** @brief Update the date
		* 	1980 <= year <= 2999
		*	1 <= month <= 12
		*	1 <= day <= 31
		*/
		void updateDate(int year, int month, int day);

		/** @brief Update timezone
		*/
		void updateTimezone( float timezone );

		/** @brief Update the time in the format HH:MM:SS
		*/
		void updateTime( std::string time );

		/** @brief Draws the backround and stats elements
		*/
		void draw();

		/** @brief Resize and reposition the stats widget
		*/
		void resizeEvent();

	private:
		sf::String fps_string;
		sf::String position_string;
		sf::String date_string;
		FPS fps;

		sf::String sun_alt_string;
		sf::String sun_azi_string;
	
		sf::String sunangle_string;

		sf::String latitude_string;
		sf::String longitude_string;

		sf::String time_string;
		sf::String timezone_string;

};

}

#endif

