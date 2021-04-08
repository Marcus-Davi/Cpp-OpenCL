#pragma once

#include "CL/cl.h"
#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <cstring>

class kernelLoader {
    public:

    kernelLoader(const char* filename, cl_program& clProgram, cl_context& clContext){

        std::ifstream file(filename);

        if(!file.is_open()){
            std::cerr << "Kernel source not found" << std::endl;            
        }

        std::stringstream strStream;
         strStream << file.rdbuf(); //read the file
         std::string str = strStream.str(); //str holds the content of the file
        const char * str_src = str.c_str();
        const size_t src_size = strlen(str_src);
        cl_int ret;
        // std::cout << src_size;
        clProgram = clCreateProgramWithSource(clContext,1,&str_src,&src_size,&ret);

    }

    private:

};