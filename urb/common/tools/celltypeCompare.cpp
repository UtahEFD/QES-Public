#include <iostream>
#include <fstream>
#include <cstdlib>

int main(int argc, char* argv[]) 
{
	std::cout << std::endl << "<===> Compare Formatted QUICurb Celltype Data <===>" << std::endl;
	//Arguments passed should be: filename, filename

	char* fileName1;
	char* fileName2;

	if(argc == 3) 
	{
		fileName1 = argv[1];
		fileName2 = argv[2];
	}
	else 
	{
		std::cout << "Please provide the name of two files to be compared." << std::endl << std::endl;
		exit(1);
	}

	std::ifstream file1(fileName1, std::ifstream::in);
	std::ifstream file2(fileName2, std::ifstream::in);

	if(!file1.is_open()) {std::cout << "Unable to open " << fileName1 << "." << std::endl; exit(1);}
	if(!file2.is_open()) {std::cout << "Unable to open " << fileName2 << "." << std::endl; exit(1);}

	float f11, f12, f13; int i1;
	float f21, f22, f23; int i2;

	int ln_dffncs = 0;
	int ln_cnt = 0;
	int frst_dffnc = 0;

	while((file1 >> f11 >> f12 >> f13 >> i1) && (file2 >> f21 >> f22 >> f23 >> i2))
	{
		ln_cnt++;
		if
		(
			f11 == f21 && 
			f12 == f22 &&
			f13 == f23 && 
			i1  == i2
		) {} // Do nothing
		else
		{
			ln_dffncs++;
			if(frst_dffnc == 0) {frst_dffnc = ln_cnt;}
		}
	}
	if(ln_dffncs == 0)
	{
		std::cout << "These files are the same with " << ln_cnt << " lines." << std::endl;
	}
	else
	{
		std::cout << "These files differ first at line " << frst_dffnc << "." << std::endl;
		std::cout << "They differ by a total of " << ln_dffncs << " lines." << std::endl;
	}
	file1.close();
	file2.close();

	std::cout << std::endl;

	return 0;
}

