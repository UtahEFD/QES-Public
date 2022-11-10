#include "WRFInput.h"

int main(int argc, char *argv[])
{
    NcFile testInputFile( "/scratch/mslp.nc", NcFile::read );
    NcVar pr = testInputFile.getVar("pr");

    std::vector< size_t > starts = { 0, 0, 0 };
    std::vector< size_t > counts = { 2, 6, 4 }; 
    int subsetDim = 1;
    for (auto i=0; i<counts.size(); i++)  {
        subsetDim *= (counts[i] - starts[i]);
    }
    
    double* prData = new double[ subsetDim ];
    pr.getVar( starts, counts, prData );

    std::cout << "[" << "pr" << "] CDL Data Dump" << std::endl << "==========================" << std::endl;
    int dimT = 2, dimLat = 6, dimLon = 4;
    for (auto t=0; t<dimT; t++) {
        std::cout << "Slice: (t=" << t << ")" << std::endl;
        for (auto lat=0; lat<dimLat; lat++) {
            for (auto lon=0; lon<dimLon; lon++) {
                auto idx = t * (dimLat * dimLon) + lat * (dimLon) + lon;
                std::cout << prData[idx] << ' ';
            }
            std::cout << std::endl;
        }
    }


    std::cout << "[" << "pr" << "] CDL Data Dump" << std::endl << "==========================" << std::endl;
    for (auto t=0; t<dimT; t++) {
        std::cout << "Slice: (t=" << t << ")" << std::endl;
        for (auto lon=0; lon<dimLon; lon++) {
            for (auto lat=0; lat<dimLat; lat++) {
                auto idx = t * (dimLat * dimLon) + lat * (dimLon) + lon;
                std::cout << prData[idx] << ' ';
            }
            std::cout << std::endl;
        }
    }

    std::string zLetter = "";
    WRFInput wrfFile("/scratch/Downloads/RXCwrfout_d07_2012-11-11_15-21", 0, 0, 0, zLetter, 0.0f, 0.0f, 0, 1);
    wrfFile.readDomainInfo();

}
