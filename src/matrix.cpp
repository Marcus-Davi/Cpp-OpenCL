#include <iostream>
#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

#include <chrono>
#include <cstring>
#include <Eigen/Dense>

#include "clErrors.h"
#include "kernelLoader.h"

#define MAX_SOURCE_SIZE (0x100000)

#define STRING_BUFFER_LEN 128
#define BLOCK_SIZE 2
typedef Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> Matrix;

using namespace std;

void Check(const char* log, const int ret)
{
        if (ret != 0)
        {
                cout << log << ": " << getClErrorString(ret) << endl;
                exit(-1);
        }
}

void DevQuery();

int main(int argc, char **argv)
{

        // DevQuery();

        cl_platform_id *platform = new cl_platform_id[2];
        cl_device_id device;
        cl_int ret;

        ret = clGetPlatformIDs(2, platform, NULL);

        ret = clGetDeviceIDs(platform[1], CL_DEVICE_TYPE_ALL, 1, &device, NULL);

        char devName[100];
        ret = clGetDeviceInfo(device, CL_DEVICE_NAME, sizeof(devName), devName, NULL);

        int squareMatrixSize;

        if (argc < 2)
        {
                squareMatrixSize = 1; // 5 x 5
        }
        else
        {
                squareMatrixSize = atoi(argv[1]);
        }

        cout << "Square Matrix Size: " << squareMatrixSize << endl;

        Matrix A(squareMatrixSize, squareMatrixSize);
        Matrix B(squareMatrixSize, squareMatrixSize);
        Matrix C(squareMatrixSize, squareMatrixSize);

        for (int i = 0; i < A.size(); ++i)
        {
                A(i) = (float)rand() / (float)RAND_MAX;
                B(i) = (float)rand() / (float)RAND_MAX;
        }

        // cout << A << endl
        //      << endl;
        // cout << B << endl
        //      << endl;
        C = A * B;
        cout << C << endl
             << endl;

             C = Matrix::Identity(squareMatrixSize,squareMatrixSize);

        // return 0;

        // Create an OpenCL context
        cl_context context = clCreateContext(NULL, 1, &device, NULL, NULL, &ret);
        // Create a command queue
        cl_command_queue command_queue = clCreateCommandQueue(context, device, 0, &ret);

        // Create memory buffers on the device for each vector
        cl_mem a_mem_obj = clCreateBuffer(context, CL_MEM_READ_ONLY,
                                          squareMatrixSize * squareMatrixSize * sizeof(float), NULL, &ret);
        cl_mem b_mem_obj = clCreateBuffer(context, CL_MEM_READ_ONLY,
                                          squareMatrixSize * squareMatrixSize * sizeof(float), NULL, &ret);
        cl_mem c_mem_obj = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
                                          squareMatrixSize * squareMatrixSize * sizeof(float), NULL, &ret);

        // Copy the lists A and B to their respective memory buffers
        ret = clEnqueueWriteBuffer(command_queue, a_mem_obj, CL_TRUE, 0,
                                   squareMatrixSize * squareMatrixSize * sizeof(float), A.data(), 0, NULL, NULL);

        ret = clEnqueueWriteBuffer(command_queue, b_mem_obj, CL_TRUE, 0,
                                   squareMatrixSize * squareMatrixSize * sizeof(float), B.data(), 0, NULL, NULL);

        cl_program clProgram;

        kernelLoader("kernels/matrix_mult_kernel.cl", clProgram, context);

        ret = clBuildProgram(clProgram, 1, &device, NULL, NULL, NULL);

        
        if (ret != 0)
        {
                char log[1000];
                clGetProgramBuildInfo(clProgram, device, CL_PROGRAM_BUILD_LOG, sizeof(log)*sizeof(char), log,NULL);
                cout << "Build Log: " << log << endl;
                cout << getClErrorString(ret) << endl;
                exit(-1);
        }

        // Create the OpenCL kernel
        cl_kernel kernel = clCreateKernel(clProgram, "matrixMult", &ret);

        Check("kernel",ret);

        // Set the arguments of the kernel
        ret = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&a_mem_obj);
        ret = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&b_mem_obj);
        ret = clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *)&c_mem_obj);
        Check("kernel arg 2",ret);
        ret = clSetKernelArg(kernel, 3, sizeof(squareMatrixSize), (void *)&squareMatrixSize);
        Check("kernel arg 3",ret);
        ret = clSetKernelArg(kernel, 4, sizeof(squareMatrixSize), (void *)&squareMatrixSize);
        Check("kernel arg 4",ret);

        // ret = clSetKernelArg(kernel, 3, sizeof(cl_mem), &squareMatrixSize);
        // ret = clSetKernelArg(kernel, 4, sizeof(cl_mem), &squareMatrixSize);

        // Execute the OpenCL kernel on the list
        size_t global_item_size[2] = {squareMatrixSize,squareMatrixSize}; // Process the entire lists
        size_t local_item_size[2] = {BLOCK_SIZE,BLOCK_SIZE};                 // Divide work items into groups of 64

        ret = clEnqueueNDRangeKernel(command_queue, kernel, 2, NULL,
                                     global_item_size, local_item_size, NULL, NULL, NULL);

        Check("kernel enque",ret);

        // Read the memory buffer C on the device to the local variable C
        // float *C = new float[]
        ret = clEnqueueReadBuffer(command_queue, c_mem_obj, CL_TRUE, 0,
                                  squareMatrixSize * squareMatrixSize * sizeof(float), C.data(), 0, NULL, NULL);


        Check("readResults",ret);
        cout << C << endl
             << endl;
}

void DevQuery()
{
        // Get platform and device information
        cl_platform_id *platforms; // multiple
        cl_device_id *devices;
        cl_uint ret_num_platforms;

        cl_int ret = clGetPlatformIDs(0, NULL, &ret_num_platforms);
        platforms = new cl_platform_id[ret_num_platforms];
        ret = clGetPlatformIDs(ret_num_platforms, platforms, NULL);

        // User-visible output - Platform information
        // for(int i = 0)

        char char_buffer[STRING_BUFFER_LEN];
        cout << "Number of Platforms: " << ret_num_platforms << endl;
        for (int i = 0; i < ret_num_platforms; ++i)
        {
                printf("Querying platform for info:\n");
                printf("\n==========================\n");
                clGetPlatformInfo(platforms[i], CL_PLATFORM_NAME, STRING_BUFFER_LEN, char_buffer, NULL);
                printf("%-40s = %s\n", "CL_PLATFORM_NAME", char_buffer);
                clGetPlatformInfo(platforms[i], CL_PLATFORM_VENDOR, STRING_BUFFER_LEN, char_buffer, NULL);
                printf("%-40s = %s\n", "CL_PLATFORM_VENDOR ", char_buffer);
                clGetPlatformInfo(platforms[i], CL_PLATFORM_VERSION, STRING_BUFFER_LEN, char_buffer, NULL);
                printf("%-40s = %s\n\n", "CL_PLATFORM_VERSION ", char_buffer);

                // Device Info
                cl_uint deviceCount;
                ret = clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_ALL, 0, NULL, &deviceCount);
                devices = new cl_device_id[deviceCount];

                ret = clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_ALL, deviceCount, devices, NULL);

                for (int j = 0; j < deviceCount; ++j)
                {
                        // print device name
                        char value[100];

                        clGetDeviceInfo(devices[j], CL_DEVICE_NAME, sizeof(value), value, NULL);
                        printf("%d. Device: %s\n", j + 1, value);

                        // print hardware device version
                        clGetDeviceInfo(devices[j], CL_DEVICE_VERSION, sizeof(value), value, NULL);
                        printf(" Hardware version: %s\n", value);
                        // free(value);

                        cl_device_type deviceType;
                        clGetDeviceInfo(devices[j], CL_DEVICE_TYPE, sizeof(deviceType), &deviceType, NULL);
                        printf(" The OpenCL device type: %ld \n", deviceType);
                        if (CL_DEVICE_TYPE_GPU & deviceType)
                                printf("   GPU\n");
                        if (CL_DEVICE_TYPE_CPU & deviceType)
                                printf("   CPU\n");
                        if (CL_DEVICE_TYPE_ACCELERATOR & deviceType)
                                printf("   Accelerator\n");
                        if (CL_DEVICE_TYPE_DEFAULT & deviceType)
                                printf("   Default\n");

                        // print software driver version
                        clGetDeviceInfo(devices[j], CL_DRIVER_VERSION, sizeof(value), value, NULL);
                        printf(" Software version: %s\n", value);
                        // free(value);

                        // print c version supported by compiler for device
                        clGetDeviceInfo(devices[j], CL_DEVICE_OPENCL_C_VERSION, sizeof(value), value, NULL);
                        printf(" OpenCL C version: %s\n", value);

                        cl_ulong memSize;
                        clGetDeviceInfo(devices[j], CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(memSize), &memSize, NULL);
                        printf(" Size of global device memory in Mbytes. %ld\n", memSize >> 20); // MByte

                        cl_uint maxFrequency;
                        clGetDeviceInfo(devices[j], CL_DEVICE_MAX_CLOCK_FREQUENCY, sizeof(maxFrequency), &maxFrequency, NULL);
                        printf(" Maximum configured clock frequency of the device in MHz: %d\n", maxFrequency);

                        cl_uint maxSamplers;
                        clGetDeviceInfo(devices[j], CL_DEVICE_MAX_SAMPLERS,
                                        sizeof(maxSamplers), &maxSamplers, NULL);
                        printf(" Maximum number of samplers that can be used in a kernel: %d\n", maxSamplers);

                        size_t maxWorkGroupSize;
                        clGetDeviceInfo(devices[j], CL_DEVICE_MAX_WORK_GROUP_SIZE,
                                        sizeof(maxWorkGroupSize), &maxWorkGroupSize, NULL);
                        printf(" Maximum number of work-items in a work-group executing a kernel using the data parallel execution model: %ld\n", maxWorkGroupSize);
                        // print parallel compute units

                        cl_uint maxWorkItemDimensions;
                        clGetDeviceInfo(devices[j], CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS,
                                        sizeof(maxWorkItemDimensions), &maxWorkItemDimensions, NULL);
                        printf(" Maximum dimensions that specify the global and local work-item IDs used by the data parallel execution model: %d\n", maxWorkItemDimensions);

                        cl_uint maxWorkItemSizes;
                        clGetDeviceInfo(devices[j], CL_DEVICE_MAX_WORK_ITEM_SIZES,
                                        sizeof(maxWorkItemSizes), &maxWorkItemSizes, NULL);
                        printf(" Maximum number of work-items that can be specified in each dimension of the work-group: %d\n", maxWorkItemSizes);

                        cl_uint maxComputeUnits;
                        clGetDeviceInfo(devices[j], CL_DEVICE_MAX_COMPUTE_UNITS,
                                        sizeof(maxComputeUnits), &maxComputeUnits, NULL);
                        printf(" Parallel compute units: %d\n", maxComputeUnits);
                }

                delete[] devices;
        }

        printf("==========================\n");
}
