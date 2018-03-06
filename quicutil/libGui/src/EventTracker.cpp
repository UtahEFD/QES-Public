/* File: EventTracker.cpp
 * Author: Matthew Overby
 */

#include "EventTracker.h"

using namespace SLUI;

EventTracker::EventTracker(sf::RenderWindow* app){

	lastEvent = 0;
	eventType = -1;
	oldPos = sf::Vector2f(0, 0);
	newPos = sf::Vector2f(0, 0);
	m_App = app;
	showWarnings = true;
}

EventTracker::~EventTracker(){}

void EventTracker::handleEvent(sf::Event *event){

	functionCount.clear();
	lastEvent = event;
	eventType = -1;

	if( event->Type == sf::Event::Resized ) eventType = EventType::W_Resize;
	else if( event->Type == sf::Event::Closed ) eventType = EventType::W_Close;
	else if( event->Type == sf::Event::KeyPressed ){
		eventType = EventType::K_Press;
		lastKeyCode = event->Key.Code;
	}
	else if( event->Type == sf::Event::KeyReleased ){
		eventType = EventType::K_Released;
		lastKeyCode = event->Key.Code;
	}
	else if( lastEvent->Type == sf::Event::MouseWheelMoved && lastEvent->MouseWheel.Delta > 0 ){
		eventType = EventType::M_Scrollup;
	}
	else if( lastEvent->Type == sf::Event::MouseWheelMoved && lastEvent->MouseWheel.Delta < 0 ){
		eventType = EventType::M_Scrolldown;
	}
	else if( lastEvent->Type == sf::Event::MouseButtonPressed && lastEvent->MouseButton.Button == sf::Mouse::Left ){
		eventType = EventType::M_Clickleft;
	}
	else if( lastEvent->Type == sf::Event::MouseButtonPressed && lastEvent->MouseButton.Button == sf::Mouse::Right ){
		eventType = EventType::M_Clickright;
	}
	else if( mouseLeftDragged() ) eventType = EventType::M_Dragleft;
	else if( mouseRightDragged() ) eventType = EventType::M_Dragright;
	else if( lastEvent->Type == sf::Event::MouseMoved ) eventType = EventType::M_Moved;

}

void EventTracker::addFunction( std::string label,  boost::function<void()> f ){

	// Check to see if the label already exists
	std::map< std::string, boost::function<void()> >::iterator it;
	it = m_eventFuncs.find( label );

	// If it's not there, add the new function to the table
	if( it == m_eventFuncs.end() ){
		m_eventFuncs.insert( std::pair< std::string, boost::function<void()> >( label, f ) );
	}
	// Otherwise, let the user know a function is being overwritten
	else {
		std::cout << "Warning: Overwriting function " << label << " in m_eventFuncs" << std::endl;
		m_eventFuncs.erase( it );
		m_eventFuncs.insert( std::pair< std::string, boost::function<void()> >( label, f ) );
	}
}

void EventTracker::callFunction( std::string label ){

	if( functionCount[ label ] > 10 && showWarnings ){
		std::cout << "\n\tWarning: EventTracker detected a possible loop while "
		<< "calling function \"" << label << "\"\n\tQuitting now" << std::endl;
		return;
	}

	// See if function with that label exists.  If it does, call it
	std::map< std::string, boost::function<void()> >::iterator it;
	it = m_eventFuncs.find( label );
	if( it != m_eventFuncs.end() ){
		functionCount[ label ]++;
		boost::function<void()> f = it->second;
		f();
	} // end function found
	else {
		std::cout << "\n**Error:  Could not find function " << label << std::endl;
	}
}

void EventTracker::showMessage(std::string str, int t){

	message = str;
	messageTime = t;
}

bool EventTracker::mouseDragged() {

	const sf::Input &input = m_App->GetInput();
	if ( input.IsMouseButtonDown(sf::Mouse::Left) && lastEvent->Type == sf::Event::MouseMoved ){
		newPos = sf::Vector2f(input.GetMouseX(), input.GetMouseY());
		return true;
	}
	return false;
}

bool EventTracker::mouseLeftDragged() {

	const sf::Input &input = m_App->GetInput();
	if ( input.IsMouseButtonDown(sf::Mouse::Left) && lastEvent->Type == sf::Event::MouseMoved ){
		newPos = sf::Vector2f(input.GetMouseX(), input.GetMouseY());
		return true;
	}
	return false;
}

bool EventTracker::mouseRightDragged() {

	const sf::Input &input = m_App->GetInput();
	if ( input.IsMouseButtonDown(sf::Mouse::Right) && lastEvent->Type == sf::Event::MouseMoved ){
		newPos = sf::Vector2f(input.GetMouseX(), input.GetMouseY());
		return true;
	}
	return false;
}




