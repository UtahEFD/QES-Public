#ifndef __BOUNDARYMATRIXTESTCASE_H__
#define __BOUNDARYMATRIXTESTCASE_H__ 1

#include <string>
#include <iomanip>

namespace QUIC
{
  class BoundaryMatrixEntryState
  {
    public:
      float e;
      float f;
      float g;
      float h;
      float m;
      float n;
      
      float o;
      float p;
      float q;
      
      bool slice;
      bool row;
      bool col;
      bool redpass;
      
    public:
      BoundaryMatrixEntryState()
      {
        this->clear();
      }

      void clear()
      {      
        e = f = g = h = m = n = 0.f;
        o = p = q = .5f;
        slice = row = col = redpass = false;
      }
      
      bool operator==(BoundaryMatrixEntryState const& thr) const
      {
        return    e == thr.e && f == thr.f
               && g == thr.g && h == thr.h
               && m == thr.m && n == thr.n
               && o == thr.o && p == thr.p && q == thr.q
               && slice == thr.slice
               && row == thr.row
               && col == thr.col
               && redpass == thr.redpass
               ;
      }
      
      bool operator!=(BoundaryMatrixEntryState const& thr) const
      {
        return !(*this == thr);
      }
  };

  class BoundaryMatrixTestCase
  {
    public:
      std::string name;
    
      BoundaryMatrixEntryState input;
      BoundaryMatrixEntryState output;
      
      int encodedBndry;
      int desiredBndry;
      
      bool didPassed;
      
    public:
      BoundaryMatrixTestCase()
      {
         this->clear();
      }
      
      void clear()
      {
        input.clear();
        output.clear();
        encodedBndry = desiredBndry = 0;
        didPassed = false;
      }
      
    public:
      bool hasPassed()
      {
        if (encodedBndry != desiredBndry)
        {
          std::cerr << "Error with " << name << std::endl;
          std::cerr << "Bad encoded boundary found" << std::endl;
          std::cerr << "Needed: " << desiredBndry << std::endl;
          std::cerr << "Found:  " << encodedBndry << std::endl;
        }
        
        if (output != input)
        {
          std::cerr << "Error with " << name << std::endl;
          std::cerr << "Output doesn't match input." << std::endl;
          std::cerr << "Input    :: Output" << std::endl;
          std::cerr << " s " << std::setw(5) << input.slice << " :: " << std::setw(5) << output.slice << std::endl;
          std::cerr << " r " << std::setw(5) << input.row << " :: " << std::setw(5) << output.row << std::endl;
          std::cerr << " c " << std::setw(5) << input.col << " :: " << std::setw(5) << output.col << std::endl;
          std::cerr << " P " << std::setw(5) << input.redpass << " :: " << std::setw(5) << output.redpass << std::endl;
          std::cerr << std::endl;
          std::cerr << " e " << std::setw(5) << input.e << " :: " << std::setw(5) << output.e << std::endl;
          std::cerr << " f " << std::setw(5) << input.f << " :: " << std::setw(5) << output.f << std::endl;
          std::cerr << " g " << std::setw(5) << input.g << " :: " << std::setw(5) << output.g << std::endl;
          std::cerr << " h " << std::setw(5) << input.h << " :: " << std::setw(5) << output.h << std::endl;
          std::cerr << " m " << std::setw(5) << input.m << " :: " << std::setw(5) << output.m << std::endl;
          std::cerr << " n " << std::setw(5) << input.n << " :: " << std::setw(5) << output.n << std::endl;
          std::cerr << " o " << std::setw(5) << input.o << " :: " << std::setw(5) << output.o << std::endl;
          std::cerr << " p " << std::setw(5) << input.p << " :: " << std::setw(5) << output.p << std::endl;
          std::cerr << " q " << std::setw(5) << input.q << " :: " << std::setw(5) << output.q << std::endl;
          
        }
        
        return didPassed = (encodedBndry == desiredBndry && output == input);
      }
  };
}

#endif
