#ifndef __4ciu7ERJN3R8n398__
#define __4ciu7ERJN3R8n398__

#include <chrono>
#include <iostream>

struct hs_timer
{
    std::chrono::time_point<std::chrono::high_resolution_clock> start;
    std::chrono::time_point<std::chrono::high_resolution_clock> end;

    void tic(const char * name)
    {
        start = std::chrono::high_resolution_clock::now();
    }

    void toc(const char * name)
    {
        end = std::chrono::high_resolution_clock::now();
        std::cout << "[" << name << " time]: "
                  << static_cast<float>(std::chrono::duration_cast<std::chrono::microseconds>(end - start).count()) / 1e3
                  << " ms\n";
    }
};

#endif