#include "QUScreenout.h"


bool quScreenout::writeQUICFile(const std::string &filename)
{
/*
		std::ofstream file(filename.c_str(), std::ifstream::out);
		if(!file.is_open())
		{
			std::cerr << "urbParser could not open :: " << filename << "." << std::endl;
			return;
		}

	file << "Lx = " << Lx << "Ly = " << Ly << "Lz = "<< Lz << std::endl;
	file << "subdomain southwest corner x coordinate " << x_subdomain_start << std::endl;
	file << "subdomain southwest corner y coordinate " << y_subdomain_start << std::endl;
	file << "subdomain northeast corner x coordinate " << x_subdomain_end << std::endl;
	file << "subdomain northeast corner y coordinate " << y_subdomain_end << std::endl;
	file << "dx = " << dx << std::endl;
	file << "dy = " << dy << std::endl;
	file << "dz = " << dz << std::endl;

	file << std::endl;
  
	for(unsigned int i = 0; i < um->buildings.size(); i++)
	{
		building* b = um->buildings[i];
		
		// There is a print building funtion. Should use it...
		file << "building #  " << i << "  building type  " << b->type << std::endl;
		file << "Height\tWidth\tLength\txfo\tyfo\tzfo\tgamma\tAttenCoef" << std::endl;
		file 	<< b->hght	<< b->wdth 	<< b->lgth
					<< b->xfo		<< b->yfo		<< b->zfo
					<< b->gamma	<< b->attenuation	<< std::endl;
	}
			
		file.close();
*/

}
