#include "QPTurbulenceField.h"

qpTurbulenceField::qpTurbulenceField()
: quicDataFile()
{
}

qpTurbulenceField::qpTurbulenceField(int nx, int ny, int nz)
: quicDataFile()
{
  
}

qpTurbulenceField::~qpTurbulenceField()
{
}

bool qpTurbulenceField::readQUICFile(const std::string &filename)
{
  // parse the prefix of the QP file.  if "dat", read as ASCII.  if
  // "bin", read as binary.
  m_asciiRead = readAsASCII(filename);

  if (beVerbose)
  {
    std::cout << "\tParsing: " << filename << " as " << (m_asciiRead ? "an ASCII" : "a BINARY") << " file." << std::endl;;
  }
  
  try {
    if (m_asciiRead)
      m_turbFile.open(filename.c_str(), std::ifstream::in);
    else 
      // read as binary file
      m_turbFile.open(filename.c_str(), std::ifstream::in | std::ios::binary);
  }
  catch (std::ifstream::failure e) {
    std::cerr << "qpTurbulenceField: [Error] Exception opening/reading file" << std::endl;
  }
		
  if (m_asciiRead)
    {
      parseASCIIHeader();
      parseASCIIData();
    }
  else
    parseBinaryData();

  m_turbFile.close();
  return true;
}


bool qpTurbulenceField::writeQUICFile(const std::string &filename)
{
  std::ofstream qpfile;
  qpfile.open(filename.c_str());

  return true;
}

void qpTurbulenceField::parseASCIIHeader()
{
  // Header
  // 
  // TITLE     = "TURBULENCE in 3D"
  // VARIABLES = "X" "Y" "Z" "SIGU" "SIGV" "SIGW" "LZ" "LEFF" "EPS" "UV" "UW" "VW"
  // ZONE T = "TURBULENCE" 
  // I=   100, J=    50,  K=    40, F=POINT
  // DT=(SINGLE SINGLE SINGLE SINGLE SINGLE SINGLE SINGLE SINGLE SINGLE SINGLE SINGLE SINGLE SINGLE)

  // Attempt to parse the header lines.  Current quic turb files have
  // 5 header lines.  We'll just store these in a list of strings for
  // now.  That may be sufficient so reproduce the file.
  
  std::string line;
  int headerLines = 5;

  for (int i=0; i<headerLines; i++)
    {
      // first thing in these files is now a comment with version information
      getline(m_turbFile, line);
      headerStrings.push_back(line);

      std::cout << "Read: " << line << std::endl;
    }
}


bool qpTurbulenceField::readAsASCII(const std::string &file)
{
  std::string filePrefix = file.substr( file.find_last_of( "." ) + 1, file.length() );

  if (filePrefix == "bin")
    return false;
  else if (filePrefix == "dat")
    return true;

  assert((filePrefix == "dat") || (filePrefix == "bin"));
}

void qpTurbulenceField::parseASCIIData()
{
  // table is built of the following columns:
  // X          Y          Z          SIGU       SIGV       SIGW       LZ         LEFF       EPS        UV         UW         VW
  // and an example:
  //
  // 5.000E-01  5.000E-01  5.000E-01  1.092E+00  1.365E+00  7.097E-01  2.000E-01  5.000E-01  3.255E-01  0.000E+00  0.000E+00  2.981E-01
  std::string line;
  std::stringstream ss(line, std::stringstream::in | std::stringstream::out);

  while (!m_turbFile.eof())
    {
      turbFieldData tf;

      // first thing in these files is now a comment with version information
      getline(m_turbFile, line);

      ss.str(line);
      ss >> tf.x >> tf.y >> tf.z >> tf.sigU >> tf.sigV >> tf.sigW >> tf.lz >> tf.leff >> tf.eps >> tf.uv >> tf.uw >> tf.vw;
      ss.clear();

      tf.print(); std::cout << std::endl;
    }	  
}


void qpTurbulenceField::parseBinaryData()
{
  m_turbFile.seekg(0, std::ios::end);
  int numBytes = m_turbFile.tellg();
  m_turbFile.seekg(0, std::ios::beg);

  // allocate memory for whole file and read everything in at once
  char *fileBuffer = new char[numBytes];
  m_turbFile.read(fileBuffer, numBytes);
  m_turbFile.close();

  short *ival;
  float *dval;

  int cd = 0;
  while (cd < numBytes)
    {
      turbFieldData tf;

      // Format write string for qp_turbfield.bin
      // write(63)sigu_c,sigv_c,sigw_c,ustarij(i,j,k),eps,elz(i,j,k)

      // Note that we need to skip over 4 or 8 bytes since that's how
      // fortran dumps binary writes (depends on 32-bit versus 64-bit,
      // respectively and I'm not sure how to determine that yet)

      // HEADER: so, extract the header values ... two 2 or 4 byte
      // values. These are "shorts" now because the header encodes the
      // sizes in 2 byte values.
      ival = reinterpret_cast<short*>(&fileBuffer[cd]);
      cd += sizeof(short);
      std::cout << *ival << ' ';

      // As of QUIC 5.72, we expect to read 24 bytes (6 * 4bytes) of
      // data containing the sigU, sigV, sigW, UStar, eps, and elz
      // values.
      //
      // if the ival read above is NOT 24 then we have a problem, so
      // throw an assert for now.
      assert( *ival == 24 );

      // this part of the header should be 0
      ival = reinterpret_cast<short*>(&fileBuffer[cd]);
      cd += sizeof(short);
      std::cout << *ival << ' ';
      assert( *ival == 0 );

      // SIGU_C
      dval = reinterpret_cast<float*>(&fileBuffer[cd]);
      cd += sizeof(float);
      tf.sigU = *dval;
      std::cout << *dval << ' ';

      // SIGV_C
      dval = reinterpret_cast<float*>(&fileBuffer[cd]);
      cd += sizeof(float);
      tf.sigV = *dval;
      std::cout << *dval << ' ';

      // SIGW_C
      dval = reinterpret_cast<float*>(&fileBuffer[cd]);
      cd += sizeof(float);
      tf.sigW = *dval;
      std::cout << *dval << ' ';

      // USTAR_{ij}(i,j,k)
      dval = reinterpret_cast<float*>(&fileBuffer[cd]);
      cd += sizeof(float);
      tf.ustar = *dval;
      std::cout << *dval << ' ';

      // EPS
      dval = reinterpret_cast<float*>(&fileBuffer[cd]);
      cd += sizeof(float);
      tf.eps = *dval;
      std::cout << *dval << ' ';

      // ELZ(i,j,k)
      dval = reinterpret_cast<float*>(&fileBuffer[cd]);
      cd += sizeof(float);
      tf.lz = *dval;
      std::cout << *dval << ' ';

      // TRAILER
      ival = reinterpret_cast<short*>(&fileBuffer[cd]);
      cd += sizeof(short);
      std::cout << *ival << ' ';

      // As of QUIC 5.72, we expect to read 24 bytes (6 * 4bytes) of
      // data containing the sigU, sigV, sigW, UStar, eps, and elz
      // values.
      //
      // if the ival read above is NOT 24 then we have a problem, so
      // throw an assert for now.
      assert( *ival == 24 );


      ival = reinterpret_cast<short*>(&fileBuffer[cd]);
      cd += sizeof(short);
      std::cout << *ival << std::endl;
    }

  delete [] fileBuffer;

}
