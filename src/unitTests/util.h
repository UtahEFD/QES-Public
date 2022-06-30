#pragma once
#include <string>
#include <vector>
#include <cmath>

#define TEST_PASS ""

std::string util_errorReport(std::string function, int lineN, std::string expected, std::string received);
std::string util_errorReport(std::string function, int lineN, int expected, int received);
std::string util_errorReport(std::string function, int lineN, float expected, float received);
std::string util_errorReport(std::string function, int lineN, double expected, double received);
std::string util_errorReport(std::string function, int lineN, char expected, char received);
std::string util_errorReport(std::string function, int lineN, bool expected, bool received);
std::string util_errorReport(std::string function, int lineN, int expected, int received);
std::string util_errorReport(std::string function, int lineN, std::string note);
std::string util_errorReport(std::string function, std::string note);

bool util_checkRMSE(std::vector<float> *a, std::vector<float> *b, float tol, float &RMSE);
