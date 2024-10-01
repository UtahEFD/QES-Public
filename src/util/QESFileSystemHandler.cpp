#include "QESFileSystemHandler.h"

fs::path QESfs::exec_path(fs::current_path());
fs::path QESfs::file_path(fs::current_path());

void QESfs::print_full_path()
{
  //fs::path exec_path(fs::current_path());
  //exec_path = fs::current_path();
  std::cout << exec_path << std::endl;
}

void QESfs::set_file_path(std::string fileName)
{
  fs::path old_path(fileName);
  fs::path new_path(exec_path);

  //std::cout << old_path.parent_path() << std::endl;
  if (old_path.is_absolute()) {
    new_path = old_path.parent_path();
    file_path = fs::canonical(new_path);
  } else {
    new_path /= old_path.parent_path();
    file_path = fs::canonical(new_path);
  }
  //std::cout << file_path << std::endl;
}

std::string QESfs::get_absolute_path(const std::string fileName)
{
  fs::path new_path(file_path);
  fs::path filename(fileName);
  if (filename.is_absolute()) {
    return filename.string();
  } else {
    try {
      filename = filename.filename();
      new_path /= fileName;
      new_path = fs::canonical(new_path.parent_path());
      new_path /= filename;
    } catch (std::filesystem::filesystem_error const& e) {
      std::cerr << "[ERROR] cannot convert to absolute path " << fileName << std::endl;
      std::cerr << "        error message: ";
      std::cerr << e.code().message() << std::endl;
      exit(EXIT_FAILURE);
    }
    //std::cout << new_path << std::endl;
    return new_path.string();
  }
}
