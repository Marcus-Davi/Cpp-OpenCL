cmake_minimum_required(VERSION 3.10)
project(opencl_test)

find_package(OpenCL REQUIRED)

include_directories(include)


add_executable(test src/main.cpp)
target_link_libraries(test ${OpenCL_LIBRARIES})


#Copy kernels to bin folder
file(GLOB kernels RELATIVE ${CMAKE_SOURCE_DIR} "kernels/*"})
foreach(kernel ${kernels})
    file(COPY ${kernel} DESTINATION ${PROJECT_BINARY_DIR}/kernels)
endforeach()