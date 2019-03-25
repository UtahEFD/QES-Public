/*	
*	Class Implementation
*
*	Before viewing this tutorial, it is recommended you understand
*	the basics explained in SampleBasic.
*/


#include "SampleGraph.h"

using namespace SLUI;




//  Initalize your render window and camera in the constructor.
SampleGraph::SampleGraph(int h, int w) : Gui(h, w){

  	// Create the render window
	createRW("Sample Graph");

	// Create the camera
	vector3D initialPosition = vector3D(20.0, 20.0, 20.0);
	vector3D initialLookAtPoint = vector3D(0.0, 0.0, 0.0);
	vector3D up = vector3D(0.0, 0.0, 1.0);
	createCamera(initialPosition, initialLookAtPoint, up);

	addKeyOption("graph.1", "Graph 1", "1", CONTROLS);
	addKeyOption("graph.2", "Graph 2", "2", CONTROLS);
	addKeyOption("graph.3", "Graph 3", "3", CONTROLS);

	/*	ENABLING GRAPH FEATURES
	*
	*	Before you can go about and use all of the graphing features of
	*	SLUI, you need to activate them.  They are off by default
	*	because using them will lower your frame rate.  This function
	*	creates all of the graphing windows and options.
	*/
	enableGraphing(true);
}




// Sample Graph 1
void SampleGraph::createGraph1() {


	/*	ADDING A PLOT TO THE GRAPH
	*
	*	Plots are kept track of in the m_graphTracker object.
	*	You can have as many plots as you want for a graph, and there
	*	are many ways to add them.  Take a look at GraphTracker.h
	*	to see them all.
	*/


	// Clear the current plot list
	m_graphTracker->clearPlots();

	
	// Using a vector array of Y values (each X value will then default to x+1)
	std::vector<float> arrayX, arrayY;
	for(int i=0; i<50; i++){
		arrayX.push_back(i);
		arrayY.push_back(i);
	}
	//m_graphTracker->addPlot(arrayX, arrayY);
	m_graphTracker->addPlot("testplot", arrayX, arrayY);
	m_graphTracker->setXMinMax(0.f, 20.f);
	m_graphTracker->setYMinMax(0.f, 50.f);



	// Using two float arrays and including a plot name
	int plotsize = 50;
	float xVals[plotsize];
	float yVals[plotsize];
	for(int i=0; i<plotsize; i++){ 
		xVals[i] = i;
		yVals[i] = sin(i)+25;
	}
	//m_graphTracker->addPlot(xVals, yVals, plotsize, "Sin Plot");


	/*	GRAPH OPTIONS
	*
	*	To change graph settings, go to Settings->Graphing.
	*	Or, you can set the default values in code using the commands
	*	below.  These options are written to the config file and
	*	reloaded with a loadConfig call.
	*
	*	If these settings are changed and you want to view their result,
	*	click the "Refresh" button in the graph window.
	*/


	/*	X/Y RANGE
	*
	*	The X and Y max values of your graph.  A graph will always start at 0.
	*/
	m_optTracker->setValue("xrange", 20);
	m_optTracker->setValue("yrange", 50);


	/*	X/Y STEPS (INTERVALS)
	*
	*	The number times you want to increment your plot.  If a fraction of
	*	a number is used, the graph will round to the nearest whole number.
	*/
	m_optTracker->setValue("xsteps", 10);
	m_optTracker->setValue("ysteps", 5);


	/*	SET LABELS
	*
	*	These labels cannot be changed within the user interface.
	*	However, you can change them in code with the setLabels
	*	command.
	*/
	m_graphTracker->setLabels("X Axis", "Y Axis");


	// Display a message
	m_eventTracker->showMessage("Generated Graph 1", 3);
}




// Sample Graph 2
void SampleGraph::createGraph2() {


	/*	MULTIPLE GRAPHS EXAMPLE
	*
	*	The best way to enable your program to have multiple graphs
	*	is having functions that are bound to keys using addKeyFunction.
	*
	*	Remember that GraphTrackers and GraphWindows are different objects.
	*	A GraphWindow needs a GraphTracker to pull plots and other info from,
	*	but you can have as many of each as you'd like.  To add an additional
	*	GraphWindow, see WindowManager::addWindow.
	*/


	// Clear the current plot list
	m_graphTracker->clearPlots();


	// Add a plot
	int plotsize = 200;
	float yVals[plotsize];
	float xVals[plotsize];
	for(int x=0; x<plotsize; x++){
		yVals[x] = cos(x)+4;
		xVals[x] = 0.5f+x;
	}
	//m_graphTracker->addPlot(xVals, yVals, plotsize, "Cos Plot");

	// Change graph options
	m_graphTracker->setLabels("Stuff", "Things");
	m_optTracker->setValue("xrange", 100);
	m_optTracker->setValue("yrange", 10);


	// Display a message
	m_eventTracker->showMessage("Generated Graph 2", 3);
}




// Sample Graph 3
void SampleGraph::createGraph3() {

	// Clear the current plot list
	m_graphTracker->clearPlots();


	// Add a plot
	int plotsize = 24;
	std::vector<float> xVals;
	std::vector<float> yVals;
	float value = 0.f;
	for(int x=0; x<plotsize; x++){
		xVals.push_back( value );
		yVals.push_back( 0.2f );
		value += 1.f;
	}

	m_graphTracker->addPlot("Negative Plot", xVals, yVals);

	// Change graph options
	m_graphTracker->setLabels("Foo", "Bar");
	m_graphTracker->setXMinMax(0.f, 24.f);
	m_graphTracker->setYMinMax(-1.f, 1.f);


	// Display a message
	m_eventTracker->showMessage("Generated Graph 3", 3);
}




// Pure virtual method int Gui::display() must be declared
int SampleGraph::display(){


	/*	SET SAVE LOCATION
	*
	*	The graph window allows you to save a graph to a png file.
	*	The default directory is root, but you can change it with
	*	the setSaveLocation command.  The filename will be a 
	*	current timestamp.
	*/
	m_graphTracker->setSaveLocation("samples/");


	// Load config file
	loadConfig();

	// Start off the example with graph 1
	createGraph1();
	m_eventTracker->showMessage("Go to Graphing->Display to view the graph", 100);

   	// Start render loop
    	while (m_App->IsOpened())
    	{
		// Process events
		sf::Event Event;

		// Start control loop
		preEventLoop();
		while (m_App->GetEvent(Event))
		{
			// Our two graph generating functions
			addKeyFunction(Event, m_optTracker->getString("graph.1"), &SampleGraph::createGraph1);
			addKeyFunction(Event, m_optTracker->getString("graph.2"), &SampleGraph::createGraph2);
			addKeyFunction(Event, m_optTracker->getString("graph.3"), &SampleGraph::createGraph3);

			// This function controls all menus and controls
			menuEvents(Event);
        	}
		postEventLoop();

		// Activate the render window
		setActive();

		// Draw the scene and menu to the render window
		displayMenu();
	}


	return EXIT_SUCCESS;
}


