/*	
*	Class Implementation
*
*	Before viewing this tutorial, it is recommended you understand
*	the basics explained in SampleBasic.
*/


#include "SampleClickworld.h"
#include <math.h>

using namespace SLUI;
using namespace std;




/*	WHAT IS CLICKWORLD?
*
*	slui allows you to generate a list of cells that are renderable
*	and clickable.  Activating world click allows you to attach certain information 
*	to each cell.  When that cell is selected (by pressing shift + left click) this 
*	information will be loaded into the cell info menu.
*
*	It also allows you to do something with that particular cell's information.
*	A cells's function button (show in cell info window) executes the virtual 
*	function "void cellFunction(int id)".
*
*	Although this example renders Cubes, it is not necessary.  You can make a
*	cellList with the same size/coordinates as your own models or objects using the
*	specific addPatch command (view ClickcellTracker.cpp for information on patches).
*	
*	I'd advise against extensive amount of renderable cells.  They are rendered
*	using glQuads so it will slow a large project down.
*/




//  Initalize your render window and camera in the constructor.
SampleClickworld::SampleClickworld(int h, int w) : Gui(h, w){

  	// Create the render window
	createRW("Sample Clickworld");

	// Create the camera
	vector3D initialPosition = vector3D(0.0, 20.0, 20.0);
	vector3D initialLookAtPoint = vector3D(0.0, 0.0, 0.0);
	vector3D up = vector3D(0.0, 0.0, 1.0);
	createCamera(initialPosition, initialLookAtPoint, up);


	/*	ACTIVATE CLICKWORLD
	*
	*	If you want to be able to click cells, set worldClick to true.
	*	The default value of worldClick is set to false because allowing a
	*	clickable world will lower your frames per second.
	*/
	enableWorldClick(true);

}




// Generate our cubeList
void SampleClickworld::generateCubes(){


	/*	CREATING A CUBE
	*
	*	By using the addCube function of the cellTracker, we'll create
	*	a list of renderable cubes.
	*/


	// For this example, we'll create a plane of cubes
	int xdim = 16;
	int ydim = 16;
	int zdim = 8;
	float cubeWidth = 1.f;
	float cubeHeight = 1.f;
	float zpos = 0.f;
	int id = 0;
	for(float x = 0; x < xdim; x+=cubeWidth){
		for(float y = 0; y < ydim; y+=cubeWidth){
			zpos = 0.f;
			cubeHeight = 1.f+log(1);
			float lastcubeheight = cubeHeight;
			for(int z = 1; z <= zdim; z++){
				// Set the variables used in the cube constructor
				vector4D color = vector4D(0.6f, 0.6f, 0.6f, 1.f);
				vector3D min = vector3D(x, y, zpos);
				vector3D max = vector3D(x+cubeWidth, y+cubeWidth, zpos+cubeHeight);
				float xc = (max.x+min.x)/2.f;
				float yc = (max.y+min.y)/2.f;
				float zc = (max.z+min.z)/2.f;

				id++;

				// Change the color of the cube
				float checkme = (xc*(1.f/cubeWidth) + yc*(1.f/cubeWidth) + z);
				if( (int)(checkme) % 2 == 0 ){
					color = vector4D(0.f, 0.f, 1.f, 1.f);
				} cout << checkme << endl;

				//color.x = ( (float)z * 1.f/(float)zdim );

				// Create the cube
				RenderableCell *box = new BoxCell( min, max );
				box->setColor( color );
				m_cellTracker->addRenderableCell( id, box );
				//m_cellTracker->addCube(id, cubeWidth, cubeHeight, center, color);


				/*	ADDING STATS TO A CUBE
				*
				*	You can add information that is stored with each individual 
				*	cube.  These stats are viewable in the cell info menu.
				*/
				std::stringstream stat1;
				stat1 << "Position:  ( " << x << ", " << y << ", 0 )";
				m_cellTracker->addStat( id, stat1.str() );

				zpos += cubeHeight;
				lastcubeheight = cubeHeight;
				cubeHeight = 1.f+log((float)z);

			} // end z loop
		} // end y loop
	} // end x loop

}




// Implement the virtual function Gui::cellFunction(std::vector<unsigned int> selected)
void SampleClickworld::cellFunction(std::vector<unsigned int> selected){


	/*	THE CELL FUNCTION
	*
	*	This is the function that is executed when the cell window Cell Function
	*	button is pressed.  It is the same for all cells, with the selected param
	*	being a copy of the stack of ids that are currently selected.  The most
	*	recently selected is on top.
	*
	*	In this example, we'll remove the selected cells completely.
	*/

	while( !selected.empty() ){

		int id = selected.back();
		selected.pop_back();

		// Display a message to the console
		std::stringstream message;
		message << "Removing cube " << id;
		m_eventTracker->showMessage(message.str(), 4);

		// Remove the cell from the list
		m_cellTracker->removeCell(id);
		m_cellTracker->removeSelected(id);
	}


	/*	MANUALLY SELECTING A CELL
	*
	*	You can manually set which cell is selected by calling
	*	the setSelected( id ) function of m_cellTracker.
	*/
}




// Pure virtual function int Gui::display() must be declared
int SampleClickworld::display(){

	// Load config file
	loadConfig();

	// Generate our cubeList
	generateCubes();

	m_eventTracker->showMessage("Press Shift + Left Click to select a cell", 100);

   	// Start render loop
    	while (m_App->IsOpened())
    	{
		// Process events
		sf::Event Event;

		// Start control loop
		preEventLoop();
		while (m_App->GetEvent(Event))
		{
			// This function controls all menus and controls
			menuEvents(Event);
        	}
		postEventLoop();

		// Activate the render window
		setActive();

		// Draw all of the cubes in the cellList
		m_cellTracker->renderCells();

		// Draw the scene and menu to the render window
		displayMenu();
	}


	return EXIT_SUCCESS;
}


