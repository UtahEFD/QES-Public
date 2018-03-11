/* File: Calendar.cpp
 * Author: Matthew Overby
 */

#include "Calendar.h"

using namespace SLUI;

DateButton::DateButton(){

	m_App = 0;
	active = false;
}

DateButton::DateButton( bool act, std::string day, sf::RenderWindow *app ){

	m_width = 40;
	m_height = 20;
	m_App = app;
	active = act;
	alwaysActive = false;

	int num = 0;
	if( sscanf( day.c_str(), "%u", &num) != EOF ){
		num_date = num;
	}
	else{  num_date = 0;  }

	m_date = sf::String( day );
	m_date.SetColor( sf::Color::Black );
	m_date.SetSize( 16 );
	mouseOver = false;

	m_background = sf::Shape::Rectangle(0, 0, m_width, m_height, sf::Color::White, 1, sf::Color::Black);
	m_highlight = sf::Shape::Rectangle(0, 0, m_width, m_height, sf::Color(255, 153, 051, 200), 1, sf::Color::Black);
}

void DateButton::setPosition( float x, float y ){

	m_posX = x;
	m_posY = y;

	m_date.SetPosition( x+2, y+1 );
	m_background = sf::Shape::Rectangle(m_posX, m_posY, m_posX+m_width, m_posY+m_height, sf::Color::White, 1, sf::Color::Black);
	m_highlight = sf::Shape::Rectangle(m_posX, m_posY, m_posX+m_width, m_posY+m_height, sf::Color(255, 153, 051, 200), 1, sf::Color::Black);
}

void DateButton::draw(){

	if( mouseOver || alwaysActive ) m_App->Draw( m_highlight );
	else m_App->Draw( m_background );
	m_App->Draw( m_date );
}

void DateButton::highlight(){

	if( active ){
		if( isMouseOver() ){
			mouseOver = true;
		}
		else mouseOver = false;
	}
}

bool DateButton::isMouseOver(){

	const sf::Input &input = m_App->GetInput();
	float mouseX = input.GetMouseX();
	float mouseY = input.GetMouseY();

	if((mouseX < m_posX + m_width) && (m_posX < mouseX) && 
	(mouseY < m_posY + m_height) && (m_posY < mouseY)) {
		return true;
	}
	else return false;
}


Calendar::Calendar( EventTracker *evt, sf::RenderWindow *app ){

	month_names[ 1 ] = "January";
	month_names[ 2 ] = "February";
	month_names[ 3 ] = "March";
	month_names[ 4 ] = "April";
	month_names[ 5 ] = "May";
	month_names[ 6 ] = "June";
	month_names[ 7 ] = "July";
	month_names[ 8 ] = "August";
	month_names[ 9 ] = "September";
	month_names[ 10 ] = "October";
	month_names[ 11 ] = "November";
	month_names[ 12 ] = "December";

	m_App = app;
	m_eventTracker = evt;
	m_width = 300;
	m_height = 200;
	m_posX = 0;
	m_posY = 0;
	m_paddingL = 5;
	m_paddingR = 5;
	m_paddingT = 5;
	active = true;
	updated = false;

	time_t rawtime;
	struct tm *timeinfo;

	time ( &rawtime );
	timeinfo = localtime ( &rawtime );
	curr_month = timeinfo->tm_mon + 1;
	curr_year = timeinfo->tm_year + 1900;
	curr_day = timeinfo->tm_mday;

	m_background = createBackground( m_posX, m_posY, m_height, m_width );
	m_inactiveBG = createBackground( m_posX, m_posY, m_height, m_width );
	m_inactiveBG.SetColor( sf::Color( 250, 250, 250 ) );

	// Need to set m_posX/Y, and m_width/height
	setDate( curr_year, curr_month, curr_day );
	generateCalData();
	initCalButtons();
	setPosition( 0, 0 );
}

void Calendar::initCalButtons(){


	//
	//	Month
	//

	sf::String september = sf::String( "September" );
	september.SetSize( 16 );
	monthIconSpacing = september.GetRect().GetWidth();

	if( curr_month > 0 && curr_month < 13 ){
		std::string month_str = month_names[ curr_month ];
		month = sf::String( month_str );
	}
	else {
		month = sf::String( "January" );
	}
	month.SetSize( 16 );
	month.SetPosition( m_posX + m_paddingL + lrButtonWidth + lrButtonSpace, m_posY+m_paddingT );
	month.SetColor( sf::Color::Black );

	int month_decIcon_posX = m_posX+m_paddingL+monthIconSpacing+lrButtonWidth+lrButtonSpace*2;

	month_incIcon = sf::Shape::Rectangle(0, 0, lrButtonWidth, lrButtonHeight, m_bgColor, 1, sf::Color(0,0,0));
	month_incIcon.SetPosition( month_decIcon_posX, m_posY+m_paddingT );

	month_decIcon = sf::Shape::Rectangle(0, 0, lrButtonWidth, lrButtonHeight, m_bgColor, 1, sf::Color(0,0,0));
	month_decIcon.SetPosition( m_posX+m_paddingL, m_posY+m_paddingT );

	month_decLab.AddPoint( 0, 0, sf::Color::Black, sf::Color::Black);
	month_decLab.AddPoint( 0, 8, sf::Color::Black, sf::Color::Black);
	month_decLab.AddPoint( -4, 4, sf::Color::Black, sf::Color::Black);
	month_decLab.SetOutlineWidth(1);
	month_decLab.SetPosition( m_posX+m_paddingL+7, m_posY+m_paddingT+6 );

	month_incLab.AddPoint( 0, 0, sf::Color::Black, sf::Color::Black);
	month_incLab.AddPoint( 0, 8, sf::Color::Black, sf::Color::Black);
	month_incLab.AddPoint( 4, 4, sf::Color::Black, sf::Color::Black);
	month_incLab.SetOutlineWidth(1);
	month_incLab.SetPosition( month_decIcon_posX+3, m_posY+m_paddingT+6 );

	//
	//	Year
	//

	std::stringstream year_ss;
	year_ss << curr_year;
	year = sf::String( year_ss.str() );
	year.SetSize( 16 );
	int year_decIcon_posX = m_posX+m_width-m_paddingR-year.GetRect().GetWidth()-lrButtonSpace*2-lrButtonWidth*2;
	year.SetPosition( year_decIcon_posX + lrButtonWidth + lrButtonSpace, m_posY+m_paddingT );
	year.SetColor( sf::Color::Black );

	year_incIcon = sf::Shape::Rectangle(0, 0, lrButtonWidth, lrButtonHeight, m_bgColor, 1, sf::Color(0,0,0));
	year_incIcon.SetPosition( m_posX+m_width-m_paddingR-lrButtonWidth, m_posY+m_paddingT );

	year_decIcon = sf::Shape::Rectangle(0, 0, lrButtonWidth, lrButtonHeight, m_bgColor, 1, sf::Color(0,0,0));
	year_decIcon.SetPosition( year_decIcon_posX, m_posY+m_paddingT );

	year_decLab.AddPoint( 0, 0, sf::Color::Black, sf::Color::Black);
	year_decLab.AddPoint( 0, 8, sf::Color::Black, sf::Color::Black);
	year_decLab.AddPoint( -4, 4, sf::Color::Black, sf::Color::Black);
	year_decLab.SetOutlineWidth(1);
	year_decLab.SetPosition( year_decIcon_posX+7, m_posY+m_paddingT+6 );

	year_incLab.AddPoint( 0, 0, sf::Color::Black, sf::Color::Black);
	year_incLab.AddPoint( 0, 8, sf::Color::Black, sf::Color::Black);
	year_incLab.AddPoint( 4, 4, sf::Color::Black, sf::Color::Black);
	year_incLab.SetOutlineWidth(1);
	year_incLab.SetPosition(  m_posX+m_width-m_paddingR-lrButtonWidth+3, m_posY+m_paddingT+6 );

}

void Calendar::setDate( int y, int m, int d ){

	if( y < 1980 ) y = 1980;
	if( m > 12 ){
		y++;
		m = 1;
	}
	else if( m < 1 ){
		y--;
		m = 12;
	}
	if( d > 31 || d < 1 ) d = curr_day;

	curr_year = y;
	curr_month = m;
	curr_day = d;

	std::string month_str = month_names[ curr_month ];
	month.SetText( month_str );

	std::stringstream year_ss;
	year_ss << curr_year;
	year.SetText( year_ss.str() );

	generateCalData();
	setPosition( m_posX, m_posY );

	updated = true;
}

CalDate Calendar::getDate(){

	CalDate result;
	result.year = curr_year;
	result.month = curr_month;
	result.day = curr_day;
	return result;
}

void Calendar::draw(){

	m_App->Draw( m_background );
	m_App->Draw( month );
	m_App->Draw( year );

	m_App->Draw( month_incIcon );
	m_App->Draw( month_decIcon );
	m_App->Draw( month_incLab );
	m_App->Draw( month_decLab );

	m_App->Draw( year_incIcon );
	m_App->Draw( year_decIcon );
	m_App->Draw( year_incLab );
	m_App->Draw( year_decLab );

	for( int i = 0; i < 7; i++ ){
		for( int j = 0; j < 7; j++ ){
			if( m_dates[i][j].num_date == curr_day ){
				m_dates[i][j].alwaysActive = true;
			}
			else{
				m_dates[i][j].alwaysActive = false;
			}

			m_dates[i][j].draw();
		} // end loop weeks
	} // end loop days

	if( !active ){
		m_App->Draw( m_inactiveBG );
	}
}

void Calendar::checkEvent(){

	if( m_eventTracker->eventType == EventType::M_Clickleft && active ){

		const sf::Input &input = m_App->GetInput();
		float mouseX = input.GetMouseX();
		float mouseY = input.GetMouseY();

		float month_decPosX = m_posX+m_paddingL;
		float month_decPosY = m_posY+m_paddingT;
		float month_incPosX = month_incIcon.GetPosition().x;
		float month_incPosY = m_posY+m_paddingT;

		float year_decPosX = year_decIcon.GetPosition().x;
		float year_decPosY = m_posY+m_paddingT;
		float year_incPosX = m_posX+m_width-m_paddingR-lrButtonWidth;
		float year_incPosY = m_posY+m_paddingT;

		if((mouseX < month_decPosX + lrButtonWidth) && (month_decPosX < mouseX) && 
		(mouseY < month_decPosY+lrButtonHeight) && ( month_decPosY < mouseY)) {
			// Over decrement button
			setDate( curr_year, curr_month-1, curr_day );
		}
		else if((mouseX < month_incPosX + lrButtonWidth) && (month_incPosX < mouseX) && 
		(mouseY < month_incPosY+lrButtonHeight) && ( month_incPosY < mouseY)) {
			// Over increment button
			setDate( curr_year, curr_month+1, curr_day );
		}
		if((mouseX < year_decPosX + lrButtonWidth) && (year_decPosX < mouseX) && 
		(mouseY < year_decPosY+lrButtonHeight) && ( year_decPosY < mouseY)) {
			// Over decrement button
			setDate( curr_year-1, curr_month, curr_day );
		}
		else if((mouseX < year_incPosX + lrButtonWidth) && (year_incPosX < mouseX) && 
		(mouseY < year_incPosY+lrButtonHeight) && ( year_incPosY < mouseY)) {
			// Over increment button
			setDate( curr_year+1, curr_month, curr_day );
		}
		else{
			for( int i = 0; i < 7; i++ ){
				for( int j = 0; j < 7; j++ ){
					if( m_dates[i][j].isMouseOver() ){
						setDate( curr_year, curr_month, m_dates[i][j].num_date );
					}
				} // end loop weeks
			} // end loop days
		}


	} // end if left mouse clicked
}

void Calendar::highlight(){

	if( active ){
		for( int i = 0; i < 7; i++ ){
			for( int j = 0; j < 7; j++ ){
				m_dates[i][j].highlight();
			}
		}
	}
}

void Calendar::setPosition( float x, float y ){

	m_posX = x;
	m_posY = y;
	m_background.SetPosition( m_posX, m_posY );
	m_inactiveBG.SetPosition( m_posX, m_posY );

	int year_decIcon_posX = m_posX+m_width-m_paddingR-year.GetRect().GetWidth()-lrButtonSpace*2-lrButtonWidth*2;
	year.SetPosition( year_decIcon_posX + lrButtonWidth + lrButtonSpace, m_posY+m_paddingT );

	month.SetPosition( m_posX + m_paddingL + lrButtonWidth + 5, m_posY+m_paddingT );

	int month_decIcon_posX = m_posX+m_paddingL+monthIconSpacing+lrButtonWidth+lrButtonSpace*2;
	month_decIcon.SetPosition( m_posX+m_paddingL, m_posY+m_paddingT );
	month_incIcon.SetPosition( month_decIcon_posX, m_posY+m_paddingT );

	month_decLab.SetPosition( m_posX+m_paddingL+7, m_posY+m_paddingT+6 );
	month_incLab.SetPosition( month_decIcon_posX+3, m_posY+m_paddingT+6 );

	year_incIcon.SetPosition( m_posX+m_width-m_paddingR-lrButtonWidth, m_posY+m_paddingT );
	year_decIcon.SetPosition( year_decIcon_posX, m_posY+m_paddingT );

	year_decLab.SetPosition( year_decIcon_posX+7, m_posY+m_paddingT+6 );
	year_incLab.SetPosition(  m_posX+m_width-m_paddingR-lrButtonWidth+3, m_posY+m_paddingT+6 );


	for( int x = 0; x < 7; x++ ){
		for( int y = 0; y < 7; y++ ){
			int posX = (m_posX+m_paddingL+5) + x*m_dates[x][y].m_width;
			int posY = (m_posY+m_paddingT+35) + y*m_dates[x][y].m_height;
			m_dates[x][y].setPosition( posX, posY );
		} // end y
	} // end x

}

void Calendar::setToDefault(){

	time_t rawtime;
	struct tm *timeinfo;

	time ( &rawtime );
	timeinfo = localtime ( &rawtime );
	curr_month = timeinfo->tm_mon + 1;
	curr_year = timeinfo->tm_year + 1900;
	curr_day = timeinfo->tm_mday;

	setDate( curr_year, curr_month, curr_day );
}


// Found on: http://c2.com/cgi/wiki?PerpetualCalendarAlgorithm
// Using calculations: http://www.jimloy.com/math/day-week.htm
// year (4-digit), month (1-12), day (1-31)
int Calendar::getDayOfWeek(int year, int month, int day) {

	if(month > 2) {
		month -= 3;
	}
	else {
		month += 9;
		year -= 1;
	}

	int century = year / 100;
	int year_last2 = year % 100;

	int date = (
		( (146097 * century) >> 2) +
		( (1461 * year_last2) >> 2) +
		( ( (153 * month) + 2) / 5) +
		day + 1721119
	);

	// 0 = Sun, 1 = Mon, ... 
	return ((date + 1) % 7);
}

void Calendar::generateCalData(){

	// Unlike curr_year/month/date, days_of_week and table are 0 indexed

	int days[] = { 0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31 };
	std::string days_of_week[] = { "Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat" };
	int start_date = getDayOfWeek( curr_year, curr_month, 1 );

	int date_lab = 1;
	for( int week = 0; week < 7; week++ ){
		for( int day = 0; day < 7; day++ ){

			if( week == 0 ){
				m_dates[day][week] = DateButton( false, days_of_week[day], m_App );
			}
			else if( day < start_date && date_lab == 1 ) {
				m_dates[day][week] = DateButton( false, "", m_App );
			}
			else if( date_lab <= days[curr_month] ){
				std::stringstream date_ss;
				date_ss << date_lab;
				m_dates[day][week] = DateButton( true, date_ss.str(), m_App );
				date_lab++;
			}
			else{
				m_dates[day][week] = DateButton( false, "", m_App );
			}

		} // end loop 7 days
	} // end loop 6 weeks
}


