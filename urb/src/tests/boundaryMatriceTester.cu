#include <vector>
#include <iostream>

#include "../cpp/boundaryMatrices.h"
#include "BoundaryMatrixTestCase.h"

#include <vector>

int main()
{
  using namespace QUIC;
  
  BoundaryMatrixTestCase aTestCase;
  aTestCase.input.e = 1.f;
  aTestCase.input.o = 1.f;
  aTestCase.desiredBndry = 520;
  
  // Make a list of test cases.
  std::vector<BoundaryMatrixTestCase> testCases;
  testCases.push_back(aTestCase);
  
    aTestCase.input.clear();
    
    aTestCase.input.f = 1.f;
    aTestCase.input.o = 1.f;
    aTestCase.desiredBndry = 264;
  
  testCases.push_back(aTestCase);
  
    aTestCase.input.clear();
    
    aTestCase.input.g = 1.f;
    aTestCase.input.p = 1.f;
    aTestCase.desiredBndry = 132;
    
  testCases.push_back(aTestCase);
  
    aTestCase.input.clear();
    
    aTestCase.input.h = 1.f;
    aTestCase.input.p = 1.f;
    aTestCase.desiredBndry = 68;
  
  testCases.push_back(aTestCase);
  
    aTestCase.input.clear();
    
    aTestCase.input.m = 1.f;
    aTestCase.input.q = 1.f;
    aTestCase.desiredBndry = 34;
  
  testCases.push_back(aTestCase);
  
    aTestCase.input.clear();
    
    aTestCase.input.n = 1.f;
    aTestCase.input.q = 1.f;
    aTestCase.desiredBndry = 18;
  
  testCases.push_back(aTestCase);
  
    aTestCase.input.clear();
    
    aTestCase.desiredBndry = 0;
  
  testCases.push_back(aTestCase);
  
    aTestCase.input.clear();
    
    aTestCase.input.e = aTestCase.input.f = aTestCase.input.g = 1.f;
    aTestCase.input.h = aTestCase.input.m = aTestCase.input.n = 1.f;
    aTestCase.input.o = aTestCase.input.p = aTestCase.input.q = 1.f;
    aTestCase.desiredBndry = 1022;
  
  testCases.push_back(aTestCase);
  
    aTestCase.input.clear();
    
    aTestCase.input.slice = aTestCase.input.row = 1.f;
    aTestCase.input.col = aTestCase.input.redpass = 1.f;
    aTestCase.input.e = aTestCase.input.f = aTestCase.input.g = 1.f;
    aTestCase.input.h = aTestCase.input.m = aTestCase.input.n = 1.f;
    aTestCase.input.o = aTestCase.input.p = aTestCase.input.q = 1.f;
    aTestCase.desiredBndry = 8191;
  
  testCases.push_back(aTestCase);
  
    aTestCase.input.clear();
    
    aTestCase.input.slice = 1.f;
    aTestCase.desiredBndry = 4096;
  
  testCases.push_back(aTestCase);
  
    aTestCase.input.clear();
    
    aTestCase.input.row = 1.f;
    aTestCase.desiredBndry = 2048;
      
  testCases.push_back(aTestCase);  
  
    aTestCase.input.clear();
    
    aTestCase.input.col = 1.f;
    aTestCase.desiredBndry = 1024;
  
  testCases.push_back(aTestCase);  

    aTestCase.input.clear();
    
    aTestCase.input.redpass = 1.f;
    aTestCase.desiredBndry = 1;
  
  testCases.push_back(aTestCase);
  
  
  
  
  
  
  unsigned numberPassed = 0;
  for (unsigned i = 0; i < testCases.size(); i++)
  {
    BoundaryMatrixTestCase tc = testCases[i];
    
    //tc.encodedBndry = 0;
    
	  encodeBoundary
	  (
	    tc.encodedBndry, 
	    tc.input.e, tc.input.f, 
	    tc.input.g, tc.input.h,
	    tc.input.m, tc.input.n,
	    tc.input.o, tc.input.p, tc.input.q
	  );
	  encodeDomainBoundaryMask
	  (
	    tc.encodedBndry,
	    tc.input.slice, tc.input.row, tc.input.col
	  );
	  
	  encodePassMask(tc.encodedBndry, tc.input.redpass);
	  
// DECODE...
	  
	  decodeBoundary
	  (
	    tc.encodedBndry, 
	    tc.output.e, tc.output.f, 
	    tc.output.g, tc.output.h,
	    tc.output.m, tc.output.n,
	    tc.output.o, tc.output.p, tc.output.q
	  );
	
	  tc.output.slice  = decodeDomainSliceMask(tc.encodedBndry);
	  tc.output.row    = decodeDomainRowMask(tc.encodedBndry);
	  tc.output.col    = decodeDomainColMask(tc.encodedBndry);
	  tc.output.redpass = decodePassMask(tc.encodedBndry);
	
    if (tc.hasPassed()) {numberPassed++;}
  }

  if (numberPassed == testCases.size()) {std::cout << "All passed." << std::endl;}
  else {std::cout << "FAIL!" << std::endl;}
  
  return 0;
}
