/*	
*	Class Implementation
*/


#include "SampleBasic.h"

using namespace SLUI;




//  Initalize your render window and camera in the constructor.
SampleBasic::SampleBasic(int h, int w) : Gui(h, w){

  	// Create the render window
	createRW("Sample Gui");

	// Create the camera
	vector3D initialPosition = vector3D(20.0, 20.0, 20.0);
	vector3D initialLookAtPoint = vector3D(0.0, 0.0, 0.0);
	vector3D up = vector3D(0.0, 0.0, 1.0);
	createCamera(initialPosition, initialLookAtPoint, up);
}




// Sample function used in control loop, see below
void SampleBasic::sampleFunc() {

	std::cout << "Toggling samplebool!" << std::endl;
	m_optTracker->toggle("samplebool");

	// Update the menu windows to reflect the recent change to samplebool
	m_winManager->update();
}




// Pure virtual method int Gui::display() must be declared
int SampleBasic::display(){


	/*	CREATING OPTIONS
	*
	*	Shown below are how to create the three types of
	* 	options.  The general intialization is
	*	add____Option(command, label, initialValue, MENU).
	*
	*	The command string is the map table's key to this
	*	option, and should only include numbers, letters,
	*	and periods.  Letters MUST be lowercase.
	*
	*	The label string is what is viewable in the preferences
	*	and graph menu.  It can contain any characters.
	*
	*	The initial value varies depending on the type of option.
	*
	*	The MENU is an int value of which menu you want the option
	*	to appear in.  You can choose any PrefWindow object, such
	*	as CONTROLS or VISUALS.
	*
	*	If you want to add an option but don't want it displayed
	*	in a PrefWindow, set the last parameter to 0.
	*/


	/*	BOOL OPTION	
	*	
	*	This creates a new bool option.  You can change its
	*	value by clicking the button in the preferences menu,
	*	or by typing "set samplebool <on/off/true/false>"
	*	in the console.  When the value of this option is
	*	changed, a flag is set to true and can be checked 
	*	using "m_optTracker->stateChanged("sampleBool")".
	*	This will return the value of the flag and change the 
	*	flag back to false.  An example is shown in the render 
	*	loop below.
	*
	*	To access a bool option's value, use
	*	"<bool> = m_optTracker->getActive("sampleBool")". To 
	*	toggle the bool option, use "m_optTracker->toggle("sampleBool")". 
	*/
	bool initialB = true;
	addBoolOption("samplebool", "Sample Bool", initialB, CONTROLS);

	addBoolOption("showLegend", "Show Legend", true, VISUALS);
	addBoolOption("colormode", "Color Mode", true, VISUALS);


	/*	VALUE OPTION	
	*	
	*	Value options are declared and operate much like a
	*	bool option.  Their value is represented as a float value and can be
	*	set by typing "set sampleVal <float>" in the console.  Also,
	*	you can set it in code by using
	*	"m_optTracker->setValue("sampleVal", <float>)".
	*
	*	To get the float value stored in this option, use the code
	*	"<float> = m_optTracker->getValue("sampleVal")".
	*
	*	You also have the option of setting upper and lower limits.
	*	This will only reflect the button, and it can still be manually
	*	set to a value higher/lower than the bounds.
	*/
	float min = -1.f;
	float max = 1.f;
	float initialF = 0.f;
	addValueOption("sampleval", "Sample Val", initialF, min, max, CONTROLS);


	/*	KEY OPTION	
	*	
	*	Key options create a new key macro useful for binding keys to specific
	*	tasks.  To bind a new key, type "bind sampleKey <char>" in the console.
	*	A new key can also be set in the preferences or graph menu with a
	*	button that is automatically generated.  For a list of valid keys,
	*	see Keys.cpp.
	*
	*	To access the key use "m_optTracker->getString("sampleKey")" which
	*	is a std::string value.  To get the sf::Key::Code value, use
	*	"m_optTracker->getKey("sampleKey")".  Keys are especially useful
	*	when creating your own gui functions shown in the control loop below.
	*/
	addKeyOption("samplekey", "Sample Key", "1", CONTROLS);


	/*	LIST OPTION
	*
	*	List options have a pre-defined set of values that they may be assigned
	*	to.  These values are represented by an std::string which is also
	*	their label in the button.  It can be assigned by typing
	*	"set samplelist <string>" in the console.  Also, you can set it in code
	*	by using "m_optTracker->setListValue("samplelist", <string>)".
	*
	*	To get the string value stored in this optino, use the code
	*	"std::string = m_optTracker->getListValue("samplelist")".
	*/
	std::vector< std::string > listOptions;
	listOptions.push_back( "Selection 1" );
	listOptions.push_back( "Selection 2" );
	addListOption("samplelist", "Sample List", listOptions, CONTROLS);


	/*	LOAD CONFIG	
	*
	*	SLUI has a built in configuration file generator.  When you
	*	create options like the ones shown above, the gui will store their
	*	values in the file.  Then, when you load the program again, the stored
	*	values will overwrite the default ones.  If you want to go back to the
	*	default values, delete the "config" file in your base directory.
	*
	*	IMPORTANT:  loadConfig() must be called after add___Option calls if
	*	you want to load these values.  If you have multiple executables in 
	*	the same directory, the config files of different programs will 
	*	overwrite each other.
	*/
	loadConfig();


	/*	RENDER LOOP	
	*	
	*	Use the sample outline below to render your scene
	*	using SLUI.  Functions and their locations
	*	shown below should be left intact.
	*/

   	// Start render loop
    	while (m_App->IsOpened())
    	{
		// Process events
		sf::Event Event;

		// Start control loop
		preEventLoop();
		while (m_App->GetEvent(Event))
		{


			/*	ADDING KEY FUNCTIONS TO GUI
			*	
			*	If you want a function to run when the app is open, pass 
			*	a function pointer to the gui using this command.  Below
			*	is a way to incorporate key bindings, but it can
			*	be replaced by an std::string, i.e. "1".  For example:
			*	addKeyFunction(Event, "1", &Sample::sampleFunc).
			*	The function must return void and have no paramaters.
			*/
			addKeyFunction(Event, m_optTracker->getString("samplekey"), &SampleBasic::sampleFunc);


			// This function controls all menus and controls
			menuEvents(Event);
        	}
		postEventLoop();

		// Activate the render window
		setActive();

		/*	
		*	Draw your scene and handle state changes here!
		*/
		drawCube();
		if( m_optTracker->stateChanged("samplebool") ) {
			m_eventTracker->showMessage("samplebool has been toggled!", 2);
		}



		// Draw the scene and menu to the render window
		displayMenu();
	}


	return EXIT_SUCCESS;
}




// Sample Color Cube
void SampleBasic::drawCube() {

	glPushMatrix();

	glBegin(GL_QUADS);

		glNormal3f(0,0,1);
		glColor3f(1,1,1);
		glVertex3f(1,1,1);
		glColor3f(1,1,0);
		glVertex3f(-1,1,1);
		glColor3f(1,0,0);
		glVertex3f(-1,-1,1);
		glColor3f(1,0,1);
		glVertex3f(1,-1,1);

		glNormal3f(1,0,0);
		glColor3f(1,1,1);
		glVertex3f(1,1,1);
		glColor3f(1,0,1);
		glVertex3f(1,-1,1);
		glColor3f(0,0,1);
		glVertex3f(1,-1,-1);
		glColor3f(0,1,1);
		glVertex3f(1,1,-1);

		glNormal3f(0,1,0);
		glColor3f(1,1,1);
		glVertex3f(1,1,1);
		glColor3f(0,1,1);
		glVertex3f(1,1,-1);
		glColor3f(0,1,0);
		glVertex3f(-1,1,-1);
		glColor3f(1,1,0);
		glVertex3f(-1,1,1);

		glNormal3f(-1,0,0);
		glColor3f(1,1,0);
		glVertex3f(-1,1,1);
		glColor3f(0,1,0);
		glVertex3f(-1,1,-1);
		glColor3f(0,0,0);
		glVertex3f(-1,-1,-1);
		glColor3f(1,0,0);
		glVertex3f(-1,-1,1);

		glNormal3f(0,-1,0);
		glColor3f(0,0,0);
		glVertex3f(-1,-1,-1);
		glColor3f(0,0,1);
		glVertex3f(1,-1,-1);
		glColor3f(1,0,1);
		glVertex3f(1,-1,1);
		glColor3f(1,0,0);
		glVertex3f(-1,-1,1);

		glNormal3f(0,0,-1);
		glColor3f(0,0,1);
		glVertex3f(1,-1,-1);
		glColor3f(0,0,0);
		glVertex3f(-1,-1,-1);
		glColor3f(0,1,0);
		glVertex3f(-1,1,-1);
		glColor3f(0,1,1);
		glVertex3f(1,1,-1);

	glEnd();

	glPopMatrix();
}


