#ifndef DEBUG_HPP
#define DEBUG_HPP

#ifndef NDEBUG

#include <iostream>

#define PrintLine(SS) \
    do { std::cerr << SS << std::endl; } while (0)

#else

#define PrintLine(SS) \
    do { } while (0)

#endif

#endif // DEBUG_HPP
