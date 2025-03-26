#ifndef WIN32
#include <unistd.h>
#endif

#include <iostream>
// #include "handleWINDSArgs.h"
#include "winds/WINDSInputData.h"
#include "winds/WRFInput.h"

int main(int argc, char *argv[])
{
  // WINDSArgs arguments;
  // arguments.processArguments(argc, argv);

  // WINDSInputData *WID = new WINDSInputData(arguments.qesFile);
  // if (!WID) {
  // std::cerr << "[ERROR] QES Input file: " << arguments.qesFile << " not able to be read successfully." << std::endl;
  // exit(EXIT_FAILURE);
  // }

  // Verify if the QES file enables WRF coupling
  // if (WID->simParams->wrfCoupling == false) {
  // std::cout << "QES Input file does not enable WRF Coupling." << std::endl;
  // exit(EXIT_SUCCESS);
  // }

  std::cout << "Starting Coupling!" << std::endl;

  // Open the WRF NC file for read/writing
  NcFile wrfInputFile("/uufs/chpc.utah.edu/common/home/u0240900/WRF-SFIRE/test/em_fire/hill/wrf.nc", NcFile::write);

  // Then, repeat the read/write cycle 10 times with WRF before exiting

  // Wait until WRF has run and placed the correct checksum and timestamp number in the file
  // CHSUM0_FMW
  // FRAME0_FMW
  // int wrfCHSUM0_FMW = 0;

  int wrfFRAME0_FMW = -1;

  NcDim fmwdim = wrfInputFile.getVar("U0_FMW").getDim(0);
  int fmwTimeSize = fmwdim.getSize();

  std::vector<size_t> fmw_StartIdx = { static_cast<unsigned long>(fmwTimeSize - 1) };
  std::vector<size_t> fmw_counts = { 1 };
  wrfInputFile.getVar("FRAME0_FMW").getVar(fmw_StartIdx, fmw_counts, &wrfFRAME0_FMW);

  // Initially, need frame0 to be -1 to start process
  while (wrfFRAME0_FMW == -1) {
    std::cout << "Waiting for FRAME0_FMW to be initialized..." << std::endl;
    #ifndef WIN32
    usleep(1000000);// 1 sec
    #endif
    wrfInputFile.getVar("FRAME0_FMW").getVar(fmw_StartIdx, fmw_counts, &wrfFRAME0_FMW);
  }

  for (auto runCount = 0; runCount < 10; runCount++) {
    std::cout << "Frame ========> " << wrfFRAME0_FMW << std::endl;
    std::cout << "-------------------------------------------" << std::endl;

    int fm_ny = wrfInputFile.getVar("FXLONG").getDim(1).getSize();
    int fm_nx = wrfInputFile.getVar("FXLONG").getDim(2).getSize();

    // Extract time dim size
    NcDim timeDim = wrfInputFile.getVar("U0_FMW").getDim(0);
    int timeSize = timeDim.getSize();
    std::cout << "Number of time series:  " << timeSize << std::endl;

    // Extract height dim size
    NcDim hgtDim = wrfInputFile.getVar("U0_FMW").getDim(1);
    int hgtSize = hgtDim.getSize();

    std::vector<size_t> interpWinds_StartIdx = { static_cast<unsigned long>(timeSize - 1), 0, 0, 0 };
    std::vector<size_t> interpWinds_counts = { 1,
                                               static_cast<unsigned long>(hgtSize),
                                               static_cast<unsigned long>(fm_ny),
                                               static_cast<unsigned long>(fm_nx) };

    std::vector<float> u0_fmw(1 * hgtSize * fm_ny * fm_nx);
    std::vector<float> v0_fmw(1 * hgtSize * fm_ny * fm_nx);
    std::vector<float> w0_fmw(1 * hgtSize * fm_ny * fm_nx);

    wrfInputFile.getVar("U0_FMW").getVar(interpWinds_StartIdx, interpWinds_counts, u0_fmw.data());
    wrfInputFile.getVar("V0_FMW").getVar(interpWinds_StartIdx, interpWinds_counts, v0_fmw.data());
    wrfInputFile.getVar("W0_FMW").getVar(interpWinds_StartIdx, interpWinds_counts, w0_fmw.data());

    //
    // Compute the Checksum
    //

    // read the checksum from WRF
    std::vector<size_t> chksum_StartIdx = { static_cast<unsigned long>(timeSize - 1) };
    std::vector<size_t> chksum_counts = { 1 };

    int wrfCHSUM0_FMW = 0;
    wrfInputFile.getVar("CHSUM0_FMW").getVar(chksum_StartIdx, chksum_counts, &wrfCHSUM0_FMW);
    std::cout << "WRF CHSUM0_FMW = " << wrfCHSUM0_FMW << std::endl;

    int qesChkSum = 0;
    qesChkSum = checksum(qesChkSum, u0_fmw);
    qesChkSum = checksum(qesChkSum, v0_fmw);
    qesChkSum = checksum(qesChkSum, w0_fmw);
    std::cout << "WRF file output checksum of FMW fields: " << qesChkSum << std::endl;

    assert(wrfCHSUM0_FMW == qesChkSum);

    // Read the heights
    std::vector<size_t> interpWindsHT_StartIdx = { static_cast<unsigned long>(timeSize - 1), 0, 0, 0 };
    std::vector<size_t> interpWindsHT_counts = { 1,
                                                 static_cast<unsigned long>(hgtSize) };

    std::vector<float> ht_fmw(1 * hgtSize);
    wrfInputFile.getVar("HT_FMW").getVar(interpWindsHT_StartIdx, interpWindsHT_counts, ht_fmw.data());


    //
    // Write back UF/VF fields
    //
    std::vector<size_t> startIdx = { 0, 0, 0, 0 };
    std::vector<size_t> counts = { 1,
                                   static_cast<unsigned long>(fm_ny),
                                   static_cast<unsigned long>(fm_nx) };

    // Initialize the two fields to 1.0
    std::vector<float> ufOut(fm_nx * fm_ny, 1.0);
    std::vector<float> vfOut(fm_nx * fm_ny, 1.0);

    // Compute the CHSUM
    int ufvf_chsum = 0;
    ufvf_chsum = checksum(ufvf_chsum, ufOut);
    ufvf_chsum = checksum(ufvf_chsum, vfOut);
    std::cout << "QES UF/VF output checksum: " << ufvf_chsum << std::endl;

    NcVar field_UF = wrfInputFile.getVar("UF");
    NcVar field_VF = wrfInputFile.getVar("VF");

    NcVar field_CHSUM = wrfInputFile.getVar("CHSUM_FMW");
    NcVar field_FRAME = wrfInputFile.getVar("FRAME_FMW");

    field_UF.putVar(startIdx, counts, ufOut.data());//, startIdx, counts );
    field_VF.putVar(startIdx, counts, vfOut.data());//, startIdx, counts );

    field_CHSUM.putVar(chksum_StartIdx, chksum_counts, &ufvf_chsum);

    std::cout << "Writing " << wrfFRAME0_FMW << std::endl;
    field_FRAME.putVar(chksum_StartIdx, chksum_counts, &wrfFRAME0_FMW);

    std::cout << "Checksum and frame updated!" << std::endl;
    wrfInputFile.sync();

    //
    // check for next frame info
    //
    int nextFrameNum = wrfFRAME0_FMW + 1;
    std::cout << "Waiting for next frame: " << nextFrameNum << std::endl;

    wrfInputFile.getVar("FRAME0_FMW").getVar(fmw_StartIdx, fmw_counts, &wrfFRAME0_FMW);
    while (wrfFRAME0_FMW != nextFrameNum) {
      std::cout << "Waiting for FRAME0_FMW ====> " << nextFrameNum << ", received " << wrfFRAME0_FMW << std::endl;
      // close file
      wrfInputFile.close();

      #ifndef WIN32
      usleep(6000000);// 2sec
      #endif

      // re-open
      wrfInputFile.open("/uufs/chpc.utah.edu/common/home/u0240900/WRF-SFIRE/test/em_fire/hill/wrf.nc", NcFile::write);
      wrfInputFile.getVar("FRAME0_FMW").getVar(fmw_StartIdx, fmw_counts, &wrfFRAME0_FMW);
    }

    // wait a bit
    #ifndef WIN32
    usleep(1000000);
    #endif
  }
}
