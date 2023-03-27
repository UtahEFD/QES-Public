#include "util.h"

std::string util_errorReport(std::string function, int lineN, std::string expected, std::string received)
{
  return function
         + " @ Line:" + std::to_string(lineN)
         + " expected:" + expected
         + " received:" + received;
}

std::string util_errorReport(std::string function, int lineN, int expected, int received)
{
  return function
         + " @ Line:" + std::to_string(lineN)
         + " expected:" + std::to_string(expected)
         + " received:" + std::to_string(received);
}

std::string util_errorReport(std::string function, int lineN, float expected, float received)
{
  return function
         + " @ Line:" + std::to_string(lineN)
         + " expected:" + std::to_string(expected)
         + " received:" + std::to_string(received);
}

std::string util_errorReport(std::string function, int lineN, double expected, double received)
{
  return function
         + " @ Line:" + std::to_string(lineN)
         + " expected:" + std::to_string(expected)
         + " received:" + std::to_string(received);
}

std::string util_errorReport(std::string function, int lineN, char expected, char received)
{
  std::string e(1, expected), r(1, received);
  return function
         + " @ Line:" + std::to_string(lineN)
         + " expected:" + e
         + " received:" + r;
}

std::string util_errorReport(std::string function, int lineN, bool expected, bool received)
{
  return function
         + " @ Line:" + std::to_string(lineN)
         + " expected:" + (expected ? "true" : "false")
         + " received:" + (received ? "true" : "false");
}

std::string util_errorReport(std::string function, int lineN, std::string note)
{
  return function
         + " @ Line:" + std::to_string(lineN)
         + " NOTE:" + note;
}

std::string util_errorReport(std::string function, std::string note)
{
  return function
         + " NOTE:" + note;
}

bool util_checkRMSE(std::vector<float> *a, std::vector<float> *b, float tol, float &RMSE)
{

  if (a->size() != b->size()) {
    printf("ERROR: incompatible size in:  util_checkRMSE");
    exit(EXIT_FAILURE);
  }

  float error(0.0), numcell(0.0);

  for (size_t k = 0; k < a->size(); ++k) {
    error += pow((a->at(k) - b->at(k)), 2.0);
    numcell++;
  }
  RMSE = sqrt(error / numcell);

  if (RMSE < tol)
    return true;
  else
    return false;
}
