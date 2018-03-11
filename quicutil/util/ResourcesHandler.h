#ifndef RESOURCESHANDLER_H
#define RESOURCESHANDLER_H

#include <string>
#include <vector>
namespace sivelab {
    
    // Provides a common method for finding various resource files.  Using
    // find and findRecursive, files located in common search paths can be
    // found.  By default . and ./resources are added to the search paths.
    // Additional paths can be added by calling addSearchPath to add a
    // literal path, or by calling addEnvironmentVariable to add the contents
    // of an environment variable to the search path.
    //
    // Returned paths are absolute.
    class ResourcesHandler {
    public:
        // Add a path to the list of paths to be searched.  If path does not
        // refer to a valid path, this silently fails.
        static void addSearchPath(std::string path);
        
        // Adds the contents of an environment variable, if it is defined,
        // to the list of paths to be searched.  It is not an error if the
        // path is not set.
        static void addEnvironmentVariable(std::string var);
        
        // Returns a working path to the first entry
        // found in the filesystem ending in the contents of 'search'.
        // If search ends in a path separator, this will only return
        // a path to a directory.  Otherwise, this will only return non-
        // directory files.  The returned path will have a trailing path
        // separator if it is a directory.
        //
        // If no entries are found, returns the empty string.
        static std::string findRecursive(std::string search);
        
        // Returns a full path to "path" inside of any of the known resource
        // directories.  This does NOT search recursively, but it can be called
        // with a hierarchy.  e.g.,
        //
        // sivelab::ResourcesHandler::find("shaders/outline.frag");
        //
        // Leading and trailing slashes are optional.  If the input has a trailing
        // slash, the returned path is required to be a directory, and will have
        // a trailing slash.
        //
        // If the path can't be found, returns the empty string.
        static std::string find(std::string path);
    private:
        static void initCheck();
        static std::vector<std::string> searchPaths;
    };
}


#endif