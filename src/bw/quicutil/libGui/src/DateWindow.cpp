/* File: DateWindow.cpp
 * Author: Matthew Overby
 */

#include "DateWindow.h"

using namespace SLUI;

DateWindow::DateWindow(int newId, WindowController *winC, OptionTracker *opT, EventTracker *evT, sf::RenderWindow *app){

	int w = app->GetWidth();
	int h = app->GetHeight();
	id = newId;
	m_winController = winC;
	m_eventTracker = evT;
	m_optTracker = opT;
	m_App = app;
	m_width = w-Margin;
	m_height = h-Margin;
	m_posX = Margin/2;
	m_posY = Margin/2;
	m_paddingL = 50;
	m_paddingR = 50;
	m_paddingT = 20;
	m_paddingB = 20;

	m_background = createBackground(m_posX, m_posY, m_height, m_width);

	closeButton = new StandardButton("Close", m_eventTracker, m_App);
	closeButton->setPosition( m_posX+m_width-Button::Width-m_paddingR, m_posY+m_height-Button::Height-m_paddingB );

	applyButton = new StandardButton("Apply Settings", m_eventTracker, m_App);
	applyButton->setPosition(m_posX+m_paddingL, m_posY+m_height-Button::Height-m_paddingB);

	useToday = new RadioButton("Use Current Date", true, m_eventTracker, m_App);
	useToday->setPosition( m_posX+m_width-Button::Width-m_paddingR, m_posY+m_paddingT*2 );

	hourButton = new ValueButton("Hour", 12.f, m_eventTracker, m_App);
	hourButton->setPosition( m_posX+m_width-Button::Width-m_paddingR, m_posY+m_paddingT*2+Button::Height*2 + 10 );
	hourButton->setMinMax( 0.f, 23.f );

	minuteButton = new ValueButton("Minute", 0.f, m_eventTracker, m_App);
	minuteButton->setPosition( m_posX+m_width-Button::Width-m_paddingR, m_posY+m_paddingT*2+Button::Height*3 + 20 );
	minuteButton->setMinMax( 0.f, 59.f );

	latButton = new ValueButton("Latitude", m_optTracker->getValue( "latitude" ), m_eventTracker, m_App);
	latButton->setPosition( m_posX+m_width-Button::Width-m_paddingR, m_posY+m_paddingT*2+Button::Height*4 + 30 );
	latButton->setMinMax( -90.f, 90.f );

	longButton = new ValueButton("Longitude", m_optTracker->getValue( "longitude" ), m_eventTracker, m_App);
	longButton->setPosition( m_posX+m_width-Button::Width-m_paddingR, m_posY+m_paddingT*2+Button::Height*5 + 40 );
	longButton->setMinMax( -180.f, 180.f );

	m_calendar = new Calendar( m_eventTracker, m_App );
	m_calendar->setPosition( m_posX+m_paddingL, m_posY+m_paddingT*2 );
	m_calendar->active = false;

	// Default to Today
	m_optTracker->addOption( "datewindow.disablecal", new BoolOption( true ) );
	CalDate newDate = m_calendar->getDate();
	m_optTracker->setValue( "sun.year", newDate.year );
	m_optTracker->setValue( "sun.month", newDate.month );
	m_optTracker->setValue( "sun.day", newDate.day );
}

DateWindow::~DateWindow(){

	delete closeButton;
	delete applyButton;
	delete useToday;
	delete m_calendar;
	delete hourButton;
	delete minuteButton;
	delete latButton;
	delete longButton;
}

void DateWindow::draw(){

	m_App->Draw( m_background );
	closeButton->draw();
	applyButton->draw();
	useToday->draw();
	m_calendar->draw();
	hourButton->draw();
	minuteButton->draw();
	latButton->draw();
	longButton->draw();
}

void DateWindow::checkEvent(){

	const sf::Input *input = &m_App->GetInput();
	closeButton->checkEvent();
	applyButton->checkEvent();
	useToday->checkEvent();
	m_calendar->checkEvent();
	hourButton->checkEvent();
	minuteButton->checkEvent();
	latButton->checkEvent();
	longButton->checkEvent();

	if(minuteButton->updated){
		minuteButton->updated = false;
		m_optTracker->setValue( "sun.minute", minuteButton->getValue() );
	}

	if(hourButton->updated){
		hourButton->updated = false;
		m_optTracker->setValue( "sun.hour", hourButton->getValue() );
	}

	if(latButton->updated){
		latButton->updated = false;
		m_optTracker->setValue( "latitude", latButton->getValue() );
	}

	if(longButton->updated){
		longButton->updated = false;
		m_optTracker->setValue( "longitude", longButton->getValue() );
	}
	
	if(closeButton->updated){
		closeButton->updated = false;
		close();
	}

	if(applyButton->updated){
		applyButton->updated = false;
		m_eventTracker->showMessage("Settings Updated", 2);
		m_eventTracker->callFunction( "updateconfig" );
		m_eventTracker->callFunction( "retrace" );
	}

	if(useToday->updated){
		useToday->updated = false;
		m_optTracker->toggle( "datewindow.disablecal" );
		if( m_optTracker->getActive( "datewindow.disablecal" ) ){
			m_calendar->active = false;
			m_calendar->setToDefault();
		}
		else{
			m_calendar->active = true;
		}
		m_eventTracker->callFunction( "updateconfig" );
	}

	if(m_calendar->updated){
		m_calendar->updated = false;
		CalDate newDate = m_calendar->getDate();
		m_optTracker->setValue( "sun.year", newDate.year );
		m_optTracker->setValue( "sun.month", newDate.month );
		m_optTracker->setValue( "sun.day", newDate.day );
		m_eventTracker->callFunction( "updateconfig" );
	}
}

void DateWindow::close(){

	if( m_winController ) m_winController->close( id );
	hourButton->active = false;
	minuteButton->active = false;
	latButton->active = false;
	longButton->active = false;
}

void DateWindow::open(){

	if( m_winController ) m_winController->open( id );
}

void DateWindow::highlight(){

	closeButton->highlight();
	applyButton->highlight();
	useToday->highlight();
	m_calendar->highlight();
	hourButton->highlight();
	minuteButton->highlight();
	latButton->highlight();
	longButton->highlight();
}

void DateWindow::update(){

	hourButton->setValue( m_optTracker->getValue( "sun.hour" ) );
	hourButton->updated = false;

	minuteButton->setValue( m_optTracker->getValue( "sun.minute" ) );
	minuteButton->updated = false;

	latButton->setValue( m_optTracker->getValue( "latitude" ) );
	latButton->updated = false;

	longButton->setValue( m_optTracker->getValue( "longitude" ) );
	longButton->updated = false;

	if( m_optTracker->getActive( "datewindow.disablecal" ) ){

		CalDate newDate = m_calendar->getDate();
		m_optTracker->setValue( "sun.year", newDate.year );
		m_optTracker->setValue( "sun.month", newDate.month );
		m_optTracker->setValue( "sun.day", newDate.day );

		useToday->setValue( true );
		useToday->updated = false;
		m_calendar->active = false;
		m_calendar->setToDefault();
		m_calendar->updated = false;


	}
	else{
		useToday->setValue( false );
		useToday->updated = false;
		m_calendar->active = true;

		m_calendar->setDate(
			m_optTracker->getValue( "sun.year" ),
			m_optTracker->getValue( "sun.month" ),
			m_optTracker->getValue( "sun.day" )
		);
		m_calendar->updated = false;
	}

}

void DateWindow::resizeEvent(){

	int w = m_App->GetWidth();
	int h = m_App->GetHeight();

	m_width = w-Margin;
	m_height = h-Margin;

	m_background = createBackground(m_posX, m_posY, m_height, m_width);

	closeButton->setPosition( m_posX+m_width-Button::Width-m_paddingR, m_posY+m_height-Button::Height-m_paddingB );
	applyButton->setPosition(m_posX+m_paddingL, m_posY+m_height-Button::Height-m_paddingB);
	useToday->setPosition( m_posX+m_width-Button::Width-m_paddingR, m_posY+m_paddingT*2 );
	m_calendar->setPosition( m_posX+m_paddingL, m_posY+m_paddingT*2 );
	hourButton->setPosition( m_posX+m_width-Button::Width-m_paddingR, m_posY+m_paddingT*2+Button::Height*2 + 10 );
	minuteButton->setPosition( m_posX+m_width-Button::Width-m_paddingR, m_posY+m_paddingT*2+Button::Height*3 + 20 );
	latButton->setPosition( m_posX+m_width-Button::Width-m_paddingR, m_posY+m_paddingT*2+Button::Height*4 + 30 );
	longButton->setPosition( m_posX+m_width-Button::Width-m_paddingR, m_posY+m_paddingT*2+Button::Height*5 + 40 );
}



