#include "WRFInput.h"

int main(int argc, char *argv[])
{
    WRFInput wrfFile("/scratch/Downloads/RXCwrfout_d07_2012-11-11_15-21");
    wrfFile.readDomainInfo();

}
