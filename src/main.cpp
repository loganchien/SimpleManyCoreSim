#include "Processor.hpp"
#include "SimConfig.hpp"
#include "Task.hpp"

#include <iostream>
#include <vector>

#include <stdlib.h>

using namespace std;

int main(int argc, char **argv)
{
    // Check the command line options
    if (argc < 3)
    {
        cerr << "USAGE: " << argv[0]
             << " [SIMCONFIG] [TASK1] [TASK2] ..." << endl;
        exit(EXIT_FAILURE);
    }

    // Load the configuration
    if (!GlobalConfig.LoadConfig(argv[1]))
    {
        cerr << "ERROR: Can't load simulation configuration from "
             << argv[1] << endl;
        exit(EXIT_FAILURE);
    }

    // Load the tasks to run
    vector<Task> tasks;
    for (size_t i = 2; i < argc; ++i)
    {
        // TODO: Load the task from the files.
    }

    // Run simulation
    Processor processor;
    processor.StartBatch(tasks);

    return 0;
}
