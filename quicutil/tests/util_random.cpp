#include <iostream>
#include <cassert>
#include <cmath>
#include <algorithm>
#include <vector>

#include "util/Random.h"

double computeMean(const std::vector<double> &v);
double computeStd(const std::vector<double> &v);

int main(int argc, char *argv[])
{
  std::vector< std::vector<double> > bins(1);

  int N = 1000;

  for (unsigned int idx=0; idx<bins.size(); idx++)
    {
      bins[idx].resize(10);
      std::fill(bins[idx].begin(), bins[idx].end(), 0);  
    }

  sivelab::Random randGen;
  
  std::cout << "Simple Uniform Sample Test" << std::endl;
  std::cout << "========================================" << std::endl;

  std::vector<double> rvals;
  for (int c=0; c<10; c++)
    {
      rvals.clear();
      for (int i=0; i<N; i++)
	{
	  rvals.push_back( randGen.uniform() );
	}
      std::cout << "Test " << c << ": Mean=" << computeMean(rvals) << ", Std=" << computeStd(rvals) << std::endl;
    }
  std::cout << std::endl;

  std::cout << "Simple Normal Distribution Test" << std::endl;
  std::cout << "=========================================" << std::endl;

  for (int c=0; c<10; c++)
    {
      rvals.clear();
      for (int i=0; i<N; i++)
	{
	  rvals.push_back( randGen.normal() );
	}
      std::cout << "Test " << c << ": Mean=" << computeMean(rvals) << ", Std=" << computeStd(rvals) << std::endl;
    }
  std::cout << std::endl;

  std::cout << "Uniform Test" << std::endl;
  std::cout << "=====================================" << std::endl;
  for (int c=0; c<10; c++)
    {
      for (int i=0; i<N; i++)
	{
	  int idx = static_cast<int>( floor(randGen.uniform() * 10.0) );
	  bins[0][idx]++;
	}
      
      for (unsigned int bIdx=0; bIdx<bins.size(); bIdx++)
	{
	  std::cout << "Bin " << bIdx << ": Mean=" << computeMean(bins[bIdx]) << ", Std=" << computeStd(bins[bIdx]) << std::endl;
	  std::fill(bins[bIdx].begin(), bins[bIdx].end(), 0);  
	}
    }
}

double computeMean(const std::vector<double> &v)
{
  double sum=0.0;
  for (unsigned int i=0; i<v.size(); i++)
  {
    sum += v[i];
  }
  return sum / (double)v.size();
}

double computeStd(const std::vector<double> &v)
{
  assert(v.size() > 1);

  double mean = computeMean(v);

  double tmp=0.0;
  for (unsigned int i=0; i<v.size(); i++)
    tmp += ((v[i] - mean) * (v[i] - mean));

  return sqrt( tmp / (v.size() - 1.0) );
}
  
