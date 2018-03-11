/* File: GuiStats.cpp
 * Author: Matthew Overby
 */

#include "GuiStats.h"

using namespace SLUI;

FPS::FPS(){

	m_fps = 0;
	m_frame = 0;
}

void FPS::update(){

	if( m_clock.GetElapsedTime() >= 0.5f ){
		m_fps = m_frame*2;
		m_frame = 0;
		m_clock.Reset();
	}
 
	m_frame++;
}

GuiStats::GuiStats(sf::RenderWindow* _App){

	m_App = _App;
	int w = m_App->GetWidth();
	int h = m_App->GetHeight();

	m_width = 180;
	m_height = 170;
	m_posX = w-m_width-5;
	m_posY = 30+5;
	m_background = createBackground(m_posX, m_posY, m_height, m_width);

	fps_string.SetPosition(m_posX+10, m_posY+10);
	fps_string.SetColor(sf::Color(0,0,0));
	fps_string.SetSize(12);
	fps_string.SetText("FPS: ");

	position_string.SetPosition(m_posX+10, m_posY+25);
	position_string.SetColor(sf::Color(0,0,0));
	position_string.SetSize(12);
	position_string.SetText("Camera: ");

	date_string.SetPosition(m_posX+10, m_posY+40);
	date_string.SetColor(sf::Color(0,0,0));
	date_string.SetSize(12);
	date_string.SetText("Date: ");

	time_string.SetPosition(m_posX+10, m_posY+55);
	time_string.SetColor(sf::Color(0,0,0));
	time_string.SetSize(12);
	time_string.SetText("Time: ");

	sun_alt_string.SetPosition(m_posX+10, m_posY+70);
	sun_alt_string.SetColor(sf::Color(0,0,0));
	sun_alt_string.SetSize(12);
	sun_alt_string.SetText("Sun Altitude: ");

	sun_azi_string.SetPosition(m_posX+10, m_posY+85);
	sun_azi_string.SetColor(sf::Color(0,0,0));
	sun_azi_string.SetSize(12);
	sun_azi_string.SetText("Sun Azimuth: ");

	latitude_string.SetPosition(m_posX+10, m_posY+100);
	latitude_string.SetColor(sf::Color(0,0,0));
	latitude_string.SetSize(12);
	latitude_string.SetText("Latitude: ");

	longitude_string.SetPosition(m_posX+10, m_posY+115);
	longitude_string.SetColor(sf::Color(0,0,0));
	longitude_string.SetSize(12);
	longitude_string.SetText("Longitude: ");

	timezone_string.SetPosition(m_posX+10, m_posY+130);
	timezone_string.SetColor(sf::Color(0,0,0));
	timezone_string.SetSize(12);
	timezone_string.SetText("Timezone: ");

	sunangle_string.SetPosition(m_posX+10, m_posY+145);
	sunangle_string.SetColor(sf::Color(0,0,0));
	sunangle_string.SetSize(12);
	sunangle_string.SetText("Sun Angle: ");

}

void GuiStats::updateCameraPos(float posX, float posY, float posZ){

	std::stringstream pos;
	pos << "Camera Pos: ( " << (int)posX << ", " << (int)posY
		<< ", " << (int)posZ << " )";
	position_string.SetText(pos.str());

}

void GuiStats::updateSunPos(float altitude, float azimuth, float angle){

	std::stringstream alt, azi, ang;
	alt << "Sun Altitude: " << altitude;
	sun_alt_string.SetText(alt.str());
	azi << "Sun Azimuth: " << azimuth;
	sun_azi_string.SetText(azi.str());
	ang << "Sun Angle: " << angle;
	sunangle_string.SetText( ang.str() );
}

void GuiStats::updateLatLon(float latitude, float longitude){

	std::stringstream lat, lon;
	lat << "Latitude: " << latitude;
	latitude_string.SetText( lat.str() );
	lon << "Longitude: " << longitude;
	longitude_string.SetText( lon.str() );

}

void GuiStats::updateDate(int year, int month, int day){

	std::stringstream date("");
	date << "Date: " << month << "/" << day << "/" << year;
	date_string.SetText( date.str() );
}

void GuiStats::updateTimezone( float timezone ){

	std::stringstream timezonestr("");
	timezonestr << "Timezone: " << (int)timezone;
	timezone_string.SetText( timezonestr.str() );
}

void GuiStats::updateTime( std::string time ){

	std::string newTime = "Time: " + time;
	time_string.SetText( newTime );
}

void GuiStats::draw(){

	fps.update();
	std::stringstream current_fps;
	current_fps << "FPS: " << fps.getFPS();
	fps_string.SetText(current_fps.str());

	m_App->Draw(m_background);
	m_App->Draw(fps_string);
	m_App->Draw(position_string);
	m_App->Draw(date_string);
	m_App->Draw(time_string);
	m_App->Draw(sun_alt_string);
	m_App->Draw(sun_azi_string);
	m_App->Draw(latitude_string);
	m_App->Draw(longitude_string);
	m_App->Draw(timezone_string);
	m_App->Draw(sunangle_string);

}

void GuiStats::resizeEvent(){

	int w = m_App->GetWidth();
	int h = m_App->GetHeight();

	m_posX = w-m_width-5;
	m_posY = 30+5;
	m_background = createBackground(m_posX, m_posY, m_height, m_width);

	fps_string.SetPosition(m_posX+10, m_posY+10);
	position_string.SetPosition(m_posX+10, m_posY+25);
	date_string.SetPosition(m_posX+10, m_posY+40);
	time_string.SetPosition(m_posX+10, m_posY+55);
	sun_alt_string.SetPosition(m_posX+10, m_posY+70);
	sun_azi_string.SetPosition(m_posX+10, m_posY+85);
	latitude_string.SetPosition(m_posX+10, m_posY+100);
	longitude_string.SetPosition(m_posX+10, m_posY+115);
	timezone_string.SetPosition(m_posX+10, m_posY+130);
	sunangle_string.SetPosition(m_posX+10, m_posY+145);

}



