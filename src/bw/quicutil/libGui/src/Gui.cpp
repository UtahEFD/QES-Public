/* File: Gui.cpp
 * Author: Matthew Overby
 */

#include "Gui.h"

using namespace SLUI;


Gui::Gui() : m_height(500), m_width(500), m_camera(0) {

	initializeMemory();
	initializeMenu();
}

Gui::Gui(int h, int w) : m_height(h), m_width(w) {

	initializeMemory();
	initializeMenu();
}

Gui::~Gui() {

	delete loadScreen;
	delete m_legend;
	delete m_App;
	delete m_optTracker;
	delete m_graphTracker;
	delete m_winManager;
	delete m_winController;
	delete m_config;
	delete m_stats;
	delete m_eventTracker;
	delete m_menuTracker;
	if(m_camera) delete m_camera;
}


void Gui::initializeMemory(){

	// Create a new render window
	m_App = new sf::RenderWindow();
	m_App->Create(sf::VideoMode(m_width, m_height), "siveLAB Gui");
	m_App->UseVerticalSync(true);
	m_App->PreserveOpenGLStates(true);
	m_App->SetActive(true);

	/*	I realize there is a terrible amount of coupling for these trackers.
	*	Appologize in advance for whatever headaches it might bring you.
	*/

	// Create the gui objects
	m_eventTracker = new EventTracker(m_App);
	m_optTracker = new OptionTracker();
	m_winController = new WindowController();
	m_winManager = new WindowManager(m_winController, m_eventTracker, m_optTracker, m_App);
	loadScreen = new LoadScreen(m_App);
	m_graphTracker = new GraphTracker(m_optTracker, m_eventTracker);
	m_config = new Config(m_optTracker);
	m_stats = new GuiStats(m_App);
	m_menuTracker = new MenuTracker(m_winManager, m_eventTracker, m_optTracker, m_App);
	m_cellTracker = new CellTracker(m_eventTracker, m_App);
	m_legend = new Legend(m_eventTracker, m_optTracker, m_App);

	// Create the menu
	m_menuTracker->createMenu("Settings");
	m_menuTracker->addItem(CONTROLS, "Controls", "Settings");
	m_menuTracker->addItem(VISUALS, "Visuals", "Settings");

	m_menuTracker->createMenu("Help");
	m_menuTracker->addItem(HELPCONTROLS, "Controls", "Help");
	m_menuTracker->addItem(HELPCONSOLE, "Console", "Help");

	// Create the Windows
	m_winManager->addWindow(new TextWindow(HELPCONTROLS, m_winController, m_optTracker, m_eventTracker, m_App));
	m_winManager->addWindow(new TextWindow(HELPCONSOLE, m_winController, m_optTracker, m_eventTracker, m_App));
	m_winManager->loadText(HELPCONTROLS, "controls.txt");
	m_winManager->loadText(HELPCONSOLE, "console.txt");

	m_winManager->addWindow(new PrefWindow(CONTROLS, m_winController, m_optTracker, m_eventTracker, m_App));
	m_winManager->addWindow(new PrefWindow(VISUALS, m_winController, m_optTracker, m_eventTracker, m_App));

	moveAccel = 0;
}

void Gui::initializeMenu(){
	
	// Options values
	worldClick = false;
	graphing = false;

	m_eventTracker->addFunction( "clearselected", boost::bind( &CellTracker::clearSelected, m_cellTracker ) );
	m_eventTracker->addFunction( "updateconfig", boost::bind( &Config::updateConfig, m_config ) );
	m_eventTracker->addFunction( "updatemenus", boost::bind( &WindowManager::update, m_winManager ) );

	addKeyOption("forward", "Forward", "w", CONTROLS);
	addKeyOption("backward", "Backward", "s", CONTROLS);
	addKeyOption("left", "Left", "a", CONTROLS);
	addKeyOption("right", "Right", "d", CONTROLS);

	addBoolOption("minvert", "Invert Mouse",  false, CONTROLS);
	addBoolOption("wireframe", "Wireframe", false, VISUALS);
	addBoolOption("showstats", "Show Stats",  true, VISUALS);
	addBoolOption("moveaccel", "Move Acceleration", false, CONTROLS);

	std::vector< std::string > listOptions;
	listOptions.push_back( "Free Move" );
	listOptions.push_back( "Model View" );
	listOptions.push_back( "Top-Down" );
	listOptions.push_back( "North Face" );
	listOptions.push_back( "South Face" );
	listOptions.push_back( "East Face" );
	listOptions.push_back( "West Face" );
	addListOption("camera.view", "Camera View", listOptions, MENU);

	addValueOption("mousespeed", "Mouse Speed", 0.5, CONTROLS);
	addValueOption("movespeed", "Move Speed", 0.3, CONTROLS);
	m_optTracker->setMinMax("movespeed", 0.f, 2.f );

	m_optTracker->addBoolOption("hidevisuals", false);

	loadConfig();

	m_eventTracker->showMessage("", 0);

}

void Gui::enableWorldClick(bool val){

	worldClick = val;

	if(worldClick) {

		m_winManager->addWindow(new ClickcellWindow(CLICKCELLWINDOW, m_winController,
			m_cellTracker, m_optTracker, m_eventTracker, m_App));

		m_menuTracker->createMenu("Cell Info");
		m_menuTracker->addItem(CLICKCELLWINDOW, "Cell Window", "Cell Info");
	}
}

void Gui::enableGraphing(bool val){

	graphing = val;

	if(graphing){

		m_winManager->addWindow(new GraphWindow(GRAPHWINDOW, m_winController,
			m_graphTracker, m_optTracker, m_eventTracker, m_App));

		m_winManager->addWindow(new PrefWindow(GRAPHING, m_winController,
			m_optTracker, m_eventTracker, m_App));

		m_winManager->addWindow(new TextWindow(HELPGRAPH, m_winController,
			m_optTracker, m_eventTracker, m_App));

		m_menuTracker->createMenu("Graphing");
		m_winManager->loadText(HELPGRAPH, "graphing.txt");

		m_menuTracker->addItem(GRAPHWINDOW, "Display", "Graphing");
		m_menuTracker->addItem(GRAPHING, "Settings", "Graphing");
		m_menuTracker->addItem(HELPGRAPH, "Graphing", "Help");

		addValueOption( "xinterval", "X Interval", 10, 1, 99, GRAPHING );
		addValueOption( "yinterval", "Y Interval", 10, 1, 99, GRAPHING );
		m_winManager->update();

	}
}

void Gui::setHideMenu(bool val){

	m_optTracker->setActive("hidevisuals", val);
}

void Gui::setFarPlane(float far){

	m_camera->farPlane = far;
}

void Gui::displayMenu(){

	if(!m_optTracker->getActive("hidevisuals")){

		preSFML();
			if(m_optTracker->getActive("showLegend")){ m_legend->draw(); }
			if(m_optTracker->getActive("showstats")){ m_stats->draw(); }
			m_winManager->drawWindows();
			m_menuTracker->draw();
		postSFML();

	}

	m_App->Display();
}

void Gui::loadConfig(){

	if( m_config->loadConfig() ){
		m_winManager->update();
	}
}

void Gui::createCamera(vector3D pos, vector3D lap, vector3D up) {

	m_camera = new Camera(pos, lap, up, m_optTracker, m_eventTracker, m_App);
}

void Gui::createRW(std::string windowName){

	m_App->Create(sf::VideoMode(m_width, m_height), windowName);
}

void Gui::preEventLoop(){

	const sf::Input &Input = m_App->GetInput();
	m_eventTracker->oldPos = sf::Vector2f(Input.GetMouseX(), Input.GetMouseY());
}

void Gui::addBoolOption(std::string command, std::string label, bool active, unsigned int window){

	PrefWindow* winPointer = (PrefWindow*)m_winManager->getWindow(window);
	if(winPointer) winPointer->addRadioButton(command, label);
	m_optTracker->addBoolOption(command, active);
}

void Gui::addValueOption(std::string command, std::string label, float initVal, unsigned int window) {

	PrefWindow* winPointer = (PrefWindow*)m_winManager->getWindow(window);
	if(winPointer) winPointer->addValueButton(command, label, initVal);
	m_optTracker->addValueOption(command, initVal);
}

void Gui::addValueOption(std::string command, std::string label, float initVal, float minVal, float maxVal, unsigned int window) {

	PrefWindow* winPointer = (PrefWindow*)m_winManager->getWindow(window);
	if(winPointer){
		winPointer->addValueButton(command, label, initVal);
		winPointer->setButtonBounds(command, minVal, maxVal);
	}
	m_optTracker->addValueOption(command, initVal);
}

void Gui::addKeyOption(std::string command, std::string label, std::string desiredKey, unsigned int window){

	PrefWindow* winPointer = (PrefWindow*)m_winManager->getWindow(window);
	if(winPointer) winPointer->addKeyButton(command, label, desiredKey);
	m_optTracker->addKeyOption(command, desiredKey);
}

void Gui::addListOption(std::string command, std::string label, std::vector<std::string> opts, unsigned int window){

	if( window == MENU ){
		m_menuTracker->createMenu( label, command, opts );
		m_optTracker->addListOption(command, opts);
	}
	else{
		PrefWindow* winPointer = (PrefWindow*)m_winManager->getWindow(window);
		if(winPointer) winPointer->addListButton(command, label, opts);
		m_optTracker->addListOption(command, opts);
	}
}	

void Gui::clearGL(){

	m_App->Clear();

	glEnable(GL_DEPTH_TEST);
	glDepthMask(GL_TRUE);
	glDepthFunc(GL_LEQUAL);
	glHint(GL_PERSPECTIVE_CORRECTION_HINT, GL_NICEST);
		
		glClearColor(bgColor[0], bgColor[1], bgColor[2], bgColor[3]);
	glClearDepth(1.0f);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
}

void Gui::preSFML(){

	if(m_optTracker->getActive("wireframe"))
		glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
}

void Gui::postSFML(){

	if(m_optTracker->getActive("wireframe"))
		glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
	else
		glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);

}

void Gui::setActive(){

	// TODO: A better way to call check event on graphTracker
	if(graphing) m_graphTracker->checkEvent(m_App);
	m_camera->setGLView();
	clearGL();
}

void Gui::postEventLoop(){

	std::string cameraView = m_optTracker->getListValue( "camera.view" );
	if(!m_winManager->isOpen() && cameraView.compare("Free Move") == 0){
		//std::cout << elapsedTime << "\n";
		const sf::Input& Input = m_App->GetInput();
		float elapsedTime = m_App->GetFrameTime();
		float w_amount = 0.0f;
		float u_amount = 0.0f;
		float m_speed = m_optTracker->getValue("movespeed");

		if( !m_optTracker->getActive("moveaccel") ){ moveAccel = 0; }
		else if( moveAccel > 2 ) moveAccel = 2;

		if (Input.IsKeyDown(m_optTracker->getKey("forward"))){
			w_amount -= ( m_speed + moveAccel ) * (100*elapsedTime) ;
			m_camera->move(0, 0, w_amount);
			moveAccel += (2 * elapsedTime);
		}
		if (Input.IsKeyDown(m_optTracker->getKey("backward"))){
			w_amount += m_speed * (100*elapsedTime);
			m_camera->move(0, 0, w_amount);
		}
		if (Input.IsKeyDown(m_optTracker->getKey("left"))){
			u_amount -= m_speed * (100*elapsedTime);
			m_camera->move(u_amount, 0, 0);
		}
		if (Input.IsKeyDown(m_optTracker->getKey("right"))){
			u_amount += m_speed * (100*elapsedTime);
			m_camera->move(u_amount, 0, 0);
		}
	}

	m_stats->updateCameraPos(m_camera->position.x, m_camera->position.y, m_camera->position.z);
}

void Gui::menuEvents(sf::Event Event){

	const sf::Input &Input = m_App->GetInput();
	m_eventTracker->handleEvent(&Event);
	m_winManager->checkEvent();
	m_menuTracker->checkEvent();
	m_legend->checkEvent();

	// Ctrl-C to exit program
	if (Event.Type == sf::Event::KeyPressed && Event.Key.Code == sf::Key::C
		&& Input.IsKeyDown(sf::Key::LControl)){
		m_config->updateConfig();
		m_App->Close();
	}


	/**********
	*	Handle the event
	***********/


	switch(m_eventTracker->eventType){

		/**********
		*	Window Closed
		***********/

		case EventType::W_Close:{
			m_App->Close();
		} break;

		/**********
		*	Mouse Left Click
		***********/

		case EventType::M_Clickleft:{

			if(worldClick){
				if(Input.IsKeyDown(sf::Key::LShift)){
					m_cellTracker->checkEvent();
					m_winManager->update();
				}
			}

		} break;

		/**********
		*	Key Pressed
		***********/

		case EventType::K_Press:{

			switch(Event.Key.Code){

				case sf::Key::Escape:{
					int topOpen = m_winManager->getTop();
					if(topOpen > 0){
						m_winManager->close(topOpen);
						m_config->updateConfig();
					}
					else{
						m_config->updateConfig();
						m_App->Close();
					}
				} break;

				case sf::Key::Tab:{
					if(!m_winManager->isOpen(CONSOLE)){
						m_winManager->open(CONSOLE);
					}
					else{
						m_winManager->close(CONSOLE);
					}
				} break;

				case sf::Key::F12:{
					m_optTracker->toggle("hidevisuals");
				} break;

				case 0:{
				} break;

			} // end switch Key::Code

		} break;

		/**********
		*	Key Released
		***********/

		case EventType::K_Released:{

			// This is to handle Move Acceleration
			// (holding down a key will increase your movespeed)
			if( Event.Key.Code == m_optTracker->getKey("forward") ){
				moveAccel = 0;
			}

		} break;

		/**********
		*	Window Resized
		***********/

		case EventType::W_Resize:{

	      		glViewport(0, 0, Event.Size.Width, Event.Size.Height);
	      		m_width = Event.Size.Width;
	      		m_height = Event.Size.Height;

			// Reset menu
			static sf::View view; 
			view.SetFromRect(sf::FloatRect(0, 0, m_width, m_height)); 
			m_App->SetView(view);

			loadScreen->resizeEvent();
			m_stats->resizeEvent();

			m_camera->setGLView();

		} break;

		/**********
		*	Not Found
		***********/

		case -1:{
		} break;

	}


	/**********
	*	Move the Camera
	***********/


	if( m_eventTracker->oldPos.y > 30 && !m_menuTracker->isOpen() && !m_winManager->isOpen()){
		m_camera->checkEvent();
	}

}



