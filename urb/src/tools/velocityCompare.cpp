#include <iostream>
#include <fstream>
#include <cmath>
#include <cstdlib>

float rel_diff(float, float);

int main(int argc, char* argv[]) 
{
	std::cout << std::endl << "<===> Compare Formatted QUICurb Velocity Data <===>" << std::endl;
	//Arguments passed should be: filename, filename

	char* fileName1;
	char* fileName2;
	float tolerance;

	if(argc == 3) 
	{
		fileName1 = argv[1];
		fileName2 = argv[2];
		tolerance = 0.1;
	}
	else if(argc == 4)
	{
		fileName1 = argv[1];
		fileName2 = argv[2];
		tolerance = (float) atof(argv[3]);
	}
	else 
	{
		std::cout << "Please provide the name of two files to be compared (and tolerance for relative difference)." << std::endl << std::endl;
		exit(1);
	}

	std::cout << tolerance << std::endl;

	std::ifstream file1(fileName1, std::ifstream::in);
	std::ifstream file2(fileName2, std::ifstream::in);

	if(!file1.is_open()) {std::cout << "Unable to open " << fileName1 << "." << std::endl; exit(1);}
	if(!file2.is_open()) {std::cout << "Unable to open " << fileName2 << "." << std::endl; exit(1);}

	float f11, f12, f13, f14, f15, f16;
	float f21, f22, f23, f24, f25, f26;

	int ln_dffncs = 0;
	int ln_cnt = 5;
	int frst_dffnc = 0;

	std::string dummy;

	for(int i = 0; i < 5; i++) {getline(file1, dummy); getline(file2, dummy);}

	while
	(
		(file1 >> f11 >> f12 >> f13 >> f14 >> f15 >> f16) && 
		(file2 >> f21 >> f22 >> f23 >> f24 >> f25 >> f26)
	)
	{
		ln_cnt++;
		if
		(
			f11 == f21 && 
			f12 == f22 &&
			f13 == f23 && 
			rel_diff(f14, f24) < tolerance &&
			rel_diff(f15, f25) < tolerance &&
			rel_diff(f16, f26) < tolerance
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

float rel_diff(float a, float b) 
{
	if(b == 0)	{return fabs(a - b);}
	else		{return fabs((a - b) / b);}
}
