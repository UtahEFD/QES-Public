#include "ResourcesHandler.h"

#include <boost/filesystem.hpp>

namespace fs = boost::filesystem;
using namespace sivelab;

std::vector<std::string> ResourcesHandler::searchPaths;

void ResourcesHandler::addSearchPath(std::string path) {
    initCheck();
    fs::path initPath(path);
    try {
        fs::path canonical = fs::canonical(initPath);
        searchPaths.push_back(canonical.string());
    } catch (...) {
        
    }
}

void ResourcesHandler::addEnvironmentVariable(std::string var) {
    initCheck();
    const char* varSetting = std::getenv(var.c_str());
    if (varSetting) {
        addSearchPath(varSetting);
    }
}

std::string ResourcesHandler::findRecursive(std::string search) {
    initCheck();
    
    if (search.size() == 0) {
        return "";
    }
    
    bool needDirectory = false;
    std::string queryString = search;
    
    if (search[search.size() - 1] == '/'
        || search[search.size() - 1] == '\\') {
        // End of query string is a directory separator.  User is therefore
        // ONLY interested in getting a directory returned.
        
        needDirectory = true;
        
        queryString.resize(queryString.size() - 1);
    }
    
    for (int pathIndex = searchPaths.size() - 1; pathIndex >= 0; pathIndex--) {
        fs::path basePath(searchPaths[pathIndex]);
        fs::recursive_directory_iterator it(basePath);
        fs::recursive_directory_iterator end;
        for ( ; it != end; ++it) {
            std::string currentPath = it->path().string();
            std::string::size_type foundPos = currentPath.rfind(queryString);
            bool pathEndsInDir = currentPath[currentPath.size() - 1] == fs::path::preferred_separator;
            bool exactMatch = foundPos + queryString.size() == currentPath.size();
            bool dirMatch = foundPos + queryString.size() + 1 == currentPath.size() && pathEndsInDir;
            if (fs::is_directory(it->path()) && needDirectory && (exactMatch || dirMatch)) {
                // Found a directory when looking for a directory, and it either exactly matches or
                // has a trailing directory separator
                if (pathEndsInDir) {
                    return it->path().string();
                } else {
                    return it->path().string() + fs::path::preferred_separator;
                }
            }
            if (exactMatch && !needDirectory) {
                if (fs::is_directory(it->path())) {
                    // Found a directory when we actually wanted a regular file,
                    // so this is not a match
                    continue;
                }
                return it->path().string();
            }
        }
    }
    return "";
}

std::string ResourcesHandler::find(std::string path) {
    initCheck();
    
    if (path.size() == 0) {
        return "";
    }
    
    bool needDirectory = false;
    
    std::string queryString = path;
    
    if (path[path.size() - 1] == fs::path::preferred_separator) {
        needDirectory = true;
        queryString.resize(queryString.size() - 1);
    }
    
    for (int pathIndex=searchPaths.size() - 1; pathIndex >= 0; pathIndex--) {
        fs::path tempPath(searchPaths[pathIndex]);
        tempPath /= queryString;
        if (fs::exists(tempPath)) {
            if (needDirectory && !fs::is_directory(tempPath)) {
                continue;
            }
            std::string ans = tempPath.string();
            if (needDirectory && ans[ans.size() - 1] != fs::path::preferred_separator) {
                ans += fs::path::preferred_separator;
            }
            return ans;
        }
    }
    return "";
}

void ResourcesHandler::initCheck() {
    if (searchPaths.size() != 0) {
        return;
    }
    
    fs::path fallback(QES_ROOT_DIR);
    fallback /= "resources/";
    if (fs::is_directory(fallback)) {
        searchPaths.push_back(fallback.string());
    }
    
    fs::path init = fs::current_path();
    searchPaths.push_back(init.string());
    init /= "resources/";
    if (fs::is_directory(init)) {
        searchPaths.push_back(init.string());
    }
}