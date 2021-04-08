cmake_minimum_required(VERSION 3.10)
project(opencl_test)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)


find_package(OpenCL REQUIRED)
find_package(Eigen3 REQUIRED)


include_directories(include ${EIGEN3_INCLUDE_DIRS})


add_executable(test src/main.cpp)
add_executable(matrix src/matrix.cpp)

target_link_libraries(test ${OpenCL_LIBRARIES})
target_link_libraries(matrix ${OpenCL_LIBRARIES})


#Copy kernels to bin folder
file(GLOB kernels RELATIVE ${CMAKE_SOURCE_DIR} "kernels/*"})
foreach(kernel ${kernels})
    file(COPY ${kernel} DESTINATION ${PROJECT_BINARY_DIR}/kernels)
endforeach()