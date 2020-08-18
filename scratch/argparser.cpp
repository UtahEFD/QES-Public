#include <iostream>
#include "util/ArgumentParsing.h"

class MyCommandLineArgs : public ArgumentParsing 
{
public:
    MyCommandLineArgs()
        : verbose(false), quicFile("")
    {
        reg("help", "help/usage information", ArgumentParsing::NONE, '?');
        reg("verbose", "turn on verbose output", ArgumentParsing::NONE, 'v');
        reg("quicproj", "Specifies the QUIC Proj file", ArgumentParsing::STRING, 'q');
    }
    ~MyCommandLineArgs() {}

    void processArguments(int argc, char *argv[]) 
    {
        processCommandLineArgs(argc, argv);

        // Process the command line arguments after registering which
        // arguments you wish to parse.
        if (isSet("help")) {
            printUsage();
            exit(EXIT_SUCCESS);
        }
        
        verbose = isSet("verbose");
        if (verbose) std::cout << "Verbose Output: ON" << std::endl;
        
        isSet( "quicproj", quicFile );
        if (verbose) std::cout << "quicproj set to " << quicFile << std::endl;
    }
    
    bool verbose;
    std::string quicFile;

private:
};


int main(int argc, char *argv[])
{
    MyCommandLineArgs cmdArgs;

    cmdArgs.processArguments( argc, argv );
    
    exit(EXIT_SUCCESS);
}
