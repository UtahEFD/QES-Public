
#include "QPSource.h"
void qpSource::build_map()
{



	std::map<std::string,boost::any>::iterator it;   
	void* temp_ptr;
	var_addressMap["numberOfSources"]=&numberOfSources;
	var_addressMap["numberOfSourceNodes"]=&numberOfSourceNodes;
	var_addressMap["sources[]"]=sizeof(SourceInfo);
	var_addressMap["sources.name"]=&sources[0].name;
	var_addressMap["sources.strengthUnits"]=&sources[0].strengthUnits;
	var_addressMap["sources.strength"]=&sources[0].strength;
	var_addressMap["sources.density"]=&sources[0].density;
	var_addressMap["sources.release"]=&sources[0].release;
	var_addressMap["sources.startTime"]=&sources[0].startTime;
	var_addressMap["sources.duration"]=&sources[0].duration;
	var_addressMap["sources.geometry"]=&sources[0].geometry;
	var_addressMap["sources.radius"]=&sources[0].radius;
       
	std::string variable_name= "sources[";
	std::string newvar;
	std::cout<<"the sources capacitu "<<sources.capacity()<<std::endl;
	std::cout<<"the size"<<sources.size()<<std::endl;
      	for(unsigned int i=0;i<sources.capacity();i++)
	{
	newvar=variable_name+boost::lexical_cast<std::string>(i)+"]";
	newvar+=".points";			
			
	
	///so the problem is the base address is a vector. we wont be able to cast it or change it 
	temp_ptr=&sources[i].points[0];
	var_addressMap[newvar]=temp_ptr;
	//std::cerr<<"the i"<<i<<"the address "<<temp_ptr<<std::endl;
	//std::cout<<"\n"<<var_addressMap[newvar].type().name()<<std::endl;
	//	exit(1);
	}
		

	var_addressMap["sources.points[]"]=sizeof(DataPoint);  //size of points
	var_addressMap["sources.points.x"]=&sources[0].points[0].x;
	var_addressMap["sources.points.y"]=&sources[0].points[0].y;
	var_addressMap["sources.points.z"]=&sources[0].points[0].z;     //this is need to get the type info out of it 

	//variable_name =variable_name + "]";
	/*for(it=var_addressMap.begin();it!=var_addressMap.end();it++)
	{
	  std::cout<<(*it).first<<":"<<std::endl;
	}
	*/
	//std::cout<<"THe variable_name "<<variable_name <<std::endl;
//	exit(1);
	///this is where i would like to build the map
	//	std::string variable_name = "sources[";
	//////	for(unsigned int i=0;i<sources.capacity();i++)
	// {
	//   varaible_name+=(string)i .
	// }
	  ///vector.capacity() give the size that is required . 

	//var_addressMap["sources."]=&sources[0].;        // std::vector<DataPoint> points;


	//var_addressMap[""]=&;


}
qpSource& qpSource::operator=(const qpSource& other)
  {

    //      std::cerr<<"operator ---------qpSource---------"<<std::endl;
     if (this == &other)
        return *this;

	
   numberOfSources=other.numberOfSources;
   numberOfSourceNodes=other.numberOfSourceNodes;


   sources.reserve( other.sources.capacity());
   copy(other.sources.begin(),other.sources.end(), std::back_inserter(sources));

std::cerr<<"TO TEST QPSOURCE CONPY CONSTRUCTOR "<<std::endl;
   return * this;
  }



bool qpSource::readQUICFile(const std::string &filename)
{
  if (beVerbose)
  {
    std::cout << "\tParsing: QP_source.inp: " << filename << std::endl;;
  }
  
  std::ifstream sourceFile(filename.c_str(), std::ifstream::in);
  if(!sourceFile.is_open())
    {
      std::cerr << "gpuPlume could not open :: " << filename << "." << std::endl;
      exit(EXIT_FAILURE);
    }
		
  std::string line;
  std::stringstream ss(line, std::stringstream::in | std::stringstream::out);

  // first thing in these files is now a comment with version information
  getline(sourceFile, line);

  std::string quicComment;
  ss.str(line);
  ss >> quicComment >> quicVersion;

  if (quicVersion != "5.72" && quicVersion != "5.6" && quicVersion != "5.92" && quicVersion != "6.01") {
    std::cerr << "Error!  Exiting!  Only QUIC version 5.6, 5.72, 5.92, and 6.01 currently supported!  Version in current file set: " << quicVersion << std::endl;
    exit(EXIT_FAILURE);
  }
  ss.clear();

  // Number of sources
  getline(sourceFile, line);
  ss.str(line);
  ss >> numberOfSources;
  ss.clear();
		
  // Number of source nodes
  getline(sourceFile, line);
  ss.str(line);
  ss >> numberOfSourceNodes;
  ss.clear();

  //
  // Allocate space for the sources
  //
  sources.resize(numberOfSources);

  // read over the remainder of the source file and pull out the respective parts
  for(unsigned int i = 0; i < sources.size(); i++)
  {
    // First line in the source info is a comment like this: !Start of source number 1
    getline(sourceFile, line);

    getline(sourceFile, line);
    ss.str(line);
    ss >> sources[i].name;
    ss.clear();

    // source strength units
  //  int sunit = -1;
    getline(sourceFile, line);
    ss.str(line);
   // ss >> sunit;
	ss>>sources[i].strengthUnits;   
 ss.clear();

assert(sources[i].strengthUnits>0 && sources[i].strengthUnits <5);
/*
      if (sunit == 1)
	sources[i].strengthUnits = qpSource::G;
      else if (sunit == 2)
	sources[i].strengthUnits = qpSource::G_PER_S;
      else if (sunit == 3)
	sources[i].strengthUnits = qpSource::L;
      else if (sunit == 4)
	sources[i].strengthUnits = qpSource::L_PER_S;
      else 
	{
	  std::cerr << "quicLoader: unknown strength unit type in source!" << std::endl;
	  exit(EXIT_FAILURE);
	}
*/
      // source strength 
      getline(sourceFile, line);
      ss.str(line);
      ss >> sources[i].strength;
      ss.clear();

      // source density
      getline(sourceFile, line);
      ss.str(line);
      ss >> sources[i].density;
      ss.clear();

      // release type
     // int rType;
      getline(sourceFile, line);
      ss.str(line);
     // ss >> rType;
      ss>>sources[i].release;
      ss.clear();
/*
      if (rType == 1)
	sources[i].release = qpSource::INSTANTANEOUS;
      else if (rType == 2)
	sources[i].release = qpSource::CONTINUOUS;
      else if (rType == 3)
	sources[i].release = qpSource::DISCRETE_CONTINUOUS;
      else
	{
	  std::cerr << "quicLoader: unknown release type for source!" << std::endl;
	  exit(EXIT_FAILURE);
	}
*/
assert(sources[i].release>0 && sources[i].release <4);
      // source start time
      getline(sourceFile, line);
      ss.str(line);
      ss >> sources[i].startTime;
      ss.clear();

      // source duration
      getline(sourceFile, line);
      ss.str(line);
      ss >> sources[i].duration;
      ss.clear();

      // source geometry
   //   int geomType;
      getline(sourceFile, line);
      ss.str(line);
     // ss >> geomType;
	ss >> sources[i].geometry ;
      ss.clear();
assert(sources[i].geometry>0 && sources[i].geometry <9);
      // Source geometry (1 = spherical shell, 2 = line, 3 = cylinder,
      // 4 = Explosive,5 = Area, 6 = Moving Point, 7 = spherical
      // volume, 8 = Submunitions)
      switch(sources[i].geometry)
	{
	  case 1:  // spherical shell
	  case 7:  // spherical volume

	   // sources[i].geometry = qpSource::SPHERICAL_SHELL;
	    sources[i].points.resize(1);

	    // x coord of sphere
	    getline(sourceFile, line);  
	    ss.str(line);
	    ss >> sources[i].points[0].x;
	    ss.clear();

	    // y coord of sphere
	    getline(sourceFile, line);  
	    ss.str(line);
	    ss >> sources[i].points[0].y;
	    ss.clear();

	    // z coord of sphere
	    getline(sourceFile, line);  
	    ss.str(line);
	    ss >> sources[i].points[0].z;
	    ss.clear();

	    // radius
	    getline(sourceFile, line);  
	    ss.str(line);
	    ss >> sources[i].radius;
	    ss.clear();

	    // Adding sphere source
	    if (beVerbose)
	    {
	      std::cout << "\t\tSpherical Shell Source: " 
		        << sources[i].points[0].x << ',' 
		        << sources[i].points[0].y << ',' 
		        << sources[i].points[0].z << std::endl;
		  }
	    break;
	    
	  case 2: // line
	  //  sources[i].geometry = qpSource::LINE;
	    
	    // !Numnber of data points
	    int numPts;
	    getline(sourceFile, line);
	    ss.str(line);
	    ss >> numPts;
	    ss.clear();

	    sources[i].points.resize(numPts);

	    // skip over this...
	    // !x (m)   y (m)   z (m)
	    getline(sourceFile, line);
	    
	    int nPts;
	    for (nPts = 0; nPts < numPts; nPts++)
	      {
		getline(sourceFile, line);  
		ss.str(line);
		ss >> sources[i].points[nPts].x >> sources[i].points[nPts].y >> sources[i].points[nPts].z;
		ss.clear();
	      }

	    sources[i].radius = 0.0;  // doesn't make sense for line

	    // Adding line source
	    if (beVerbose)
	    {
	      std::cout << "\t\tLine Source: ";
	      for (nPts = 0; nPts < numPts; nPts++)
	        {
		  if (nPts > 0) std::cout << " <----> ";
		  std::cout << sources[i].points[nPts].x << ',' 
			    << sources[i].points[nPts].y << ',' 
			    << sources[i].points[nPts].z;
	        }
	      std::cout << std::endl;
	    }
	    break;

	  case 3: // cylinder
	    // 
	    // !!!!!!!!!!!!!!!!!!!!!!!!!!!!!
	    // don't support cylinder yet, so stick a sphere there...
	    // !!!!!!!!!!!!!!!!!!!!!!!!!!!!!
	    // 

	   // sources[i].geometry = qpSource::SPHERICAL_SHELL;

	    sources[i].points.resize(1);

	    // !x coord of center of cylinder base (m)
	    getline(sourceFile, line);
	    ss.str(line);
	    ss >> sources[i].points[0].x;
	    ss.clear();

	    // !y coord of center of cylinder base (m)
	    getline(sourceFile, line);
	    ss.str(line);
	    ss >> sources[i].points[0].y;
	    ss.clear();

	    // !z coord of cylinder base (m)
	    getline(sourceFile, line);
	    ss.str(line);
	    ss >> sources[i].points[0].z;
	    ss.clear();

	    // !radius of cylinder (m)
	    getline(sourceFile, line);
	    ss.str(line);
	    ss >> sources[i].radius;
	    ss.clear();

	    // !height of cylinder (m)
	    getline(sourceFile, line);

      if (beVerbose)
      {
  	    std::cout << "\t\tCylinder Source: not added as cylinder... but as sphere." << std::endl;
	    }
	    break;

	  case 5: // area
	    //
	    // don't suppot area yet, so stick a sphere there at a reasonable location...
	    //

	    float a_xfo, a_yfo, a_zfo, a_w, a_h, a_l, a_rot;

	    // !Area source xfo (m)
	    getline(sourceFile, line);
	    ss.str(line);
	    ss >> a_xfo;
	    ss.clear();

	    // !Area source yfo (m)
	    getline(sourceFile, line);
	    ss.str(line);
	    ss >> a_yfo;
	    ss.clear();

	    // !Area source zfo (m)
	    getline(sourceFile, line);
	    ss.str(line);
	    ss >> a_zfo;
	    ss.clear();

	    // !Area source length (m)
	    getline(sourceFile, line);
	    ss.str(line);
	    ss >> a_l;
	    ss.clear();

	    // !Area source width (m)
	    getline(sourceFile, line);
	    ss.str(line);
	    ss >> a_w;
	    ss.clear();

	    // !Area source height (m)
	    getline(sourceFile, line);
	    ss.str(line);
	    ss >> a_h;
	    ss.clear();

	    // !Area source rotation angle (o)
	    getline(sourceFile, line);
	    ss.str(line);
	    ss >> a_rot;
	    ss.clear();

	    //
	    // don't suppot area yet, so stick a sphere there at a reasonable location...
	    //
	 //   sources[i].geometry = qpSource::SPHERICAL_SHELL;
	    
	    sources[i].points.resize(1);
	    sources[i].points[0].x = a_xfo;
	    sources[i].points[0].y = a_yfo;
	    sources[i].points[0].z = a_zfo;
	    sources[i].radius = a_h;

      if (beVerbose)
      {
	      std::cout << "\t\tArea Source: not added directly, but represented as sphere." << std::endl;
	    }
	    break;

	    // case 4: // explosive
	    // case 6: // moving point
	    // case 8: // submunitions
	  default:
      std::cerr << "\t\tEmitter Type " << sources[i].geometry << " not yet supported." << std::endl;
	    exit(EXIT_FAILURE);
	    break;
	}

      // After this, we again have a comment
      getline(sourceFile, line);        
    }

  sourceFile.close();
  return true;
}


bool qpSource::writeQUICFile(const std::string &filename)
{
  std::cerr<<"within write"<<std::endl;
  std::ofstream qpfile;
  qpfile.open(filename.c_str());

  if (qpfile.is_open())
    {
      qpfile << "!QUIC 5.72" << std::endl;

      qpfile << numberOfSources << "\t\t\t!Number of sources" << std::endl;
      qpfile << numberOfSourceNodes << "\t\t\t!Number of source nodes" << std::endl;

      for (unsigned int i=0; i<sources.size(); i++)
	{
	  // quic uses fortran indexing, so i+1 when writing
	  qpfile << "!Start of source number " << i+1 << std::endl;

	  qpfile << sources[i].name <<"\t"<<i+1<< "\t\t\t!source name" << std::endl;
	  qpfile << sources[i].strengthUnits << "\t\t\t!Source strength units (1 = g, 2 = g/s, 3 = L,4 = L/s)" << std::endl;
	  qpfile << sources[i].strength << "\t\t\t!Source Strength" << std::endl;
	  qpfile << sources[i].density << "\t\t\t!Source Density (kg/m^3) [Only used for Volume based source strengths]" << std::endl;
	  qpfile << sources[i].release << "\t\t\t!Release Type: =1 for instantaneous;=2 for continuous; =3 for discrete continous" << std::endl;
	  qpfile << sources[i].startTime << "\t\t\t!Source start time (s)" << std::endl;
	  qpfile << sources[i].duration << "\t\t\t!Source duration (s)" << std::endl;
	  qpfile << sources[i].geometry << "\t\t\t!Source geometry (1 = spherical shell, 2 = line, 3 = cylinder, 4 = Explosive,5 = Area, 6 = Moving Point, 7 = spherical volume, 8 = Submunitions)" << std::endl;
	  
	  if (sources[i].geometry == 2)
	    {
	      qpfile << sources[i].points.size() << "\t\t\t!Number of data points" << std::endl;
	      qpfile << "!x (m)   y (m)   z (m)" << std::endl;
	      
	      qpfile << sources[i].points[0].x << " " << sources[i].points[0].y << " " << sources[i].points[0].z << std::endl;
	      qpfile << sources[i].points[1].x << " " << sources[i].points[1].y << " " << sources[i].points[1].z << std::endl;

	      qpfile << "!End of source number " << i+1 << std::endl;

	    }
	  else 
	    {

	      // only will work for sphere!!!

	      int nPts = 0;
	      qpfile << sources[i].points[nPts].x << "\t\t\t!x coord of center of sphere (m)" << std::endl;
	      qpfile << sources[i].points[nPts].y << "\t\t\t!y coord of center of sphere (m)" << std::endl;
	      qpfile << sources[i].points[nPts].z << "\t\t\t!z coord of center of sphere (m)" << std::endl;
	      qpfile << sources[i].points[nPts].z << "\t\t\t!z coord of center of sphere (m)" << std::endl;
	  
	      qpfile << sources[i].radius << "\t\t\t!radius of sphere (m)" << std::endl;

	      qpfile << "!End of source number " << i+1 << std::endl;
	    }
	}

      return true;
    }

  return true;
}

