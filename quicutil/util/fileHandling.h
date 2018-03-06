#ifndef __SIVELAB_FILE_HANDLING_H__
#define __SIVELAB_FILE_HANDLING_H__ 1

namespace sivelab {

  class FileHandling
  {
  public:
    FileHandling() {}
    ~FileHandling() {}

    static bool isPathAbsolute(const std::string &filename)
    {
      // An absolute path would have a / that begins the string.
      // Likewise, on Windows, it would have a C:\ style beginning.
      if (filename[0] == '/')
	return true;
    
      return false;
    }
  };

}

#endif 
