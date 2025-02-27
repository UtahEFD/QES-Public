#pragma once

// #include <boost/filesystem/operations.hpp>
// #include <boost/filesystem/path.hpp>
#include <iostream>

#include <filesystem>
#include <iostream>
#include <system_error>

// namespace fs = boost::filesystem;
namespace fs = std::filesystem;

class QESfs
{
public:
  QESfs();

  static void print_full_path();
  static void set_file_path(std::string);
  static std::string get_absolute_path(const std::string);

private:
  static fs::path exec_path;
  static fs::path file_path;
};
