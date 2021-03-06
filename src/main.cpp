#include <iostream>
#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

#include <chrono>

#include "clErrors.h"

#define MAX_SOURCE_SIZE (0x100000)

#define STRING_BUFFER_LEN 128

using namespace std;

void DevQuery();

int main(int argc, char **argv)
{

        DevQuery();

        // Create the two input vectors
        int NElements;
        if (argc < 2)
        {
                NElements = 1000000;
        }
        else
        {
                NElements = atoi(argv[1]);
        }

        std::cout << "# Elements: " << NElements << std::endl;

        std::chrono::high_resolution_clock::time_point start, end;

        // Load the kernel source code into the array source_str
        FILE *fp;
        char *source_str;
        size_t source_size;

        fp = fopen("kernels/vector_add_kernel.cl", "r");
        if (!fp)
        {
                fprintf(stderr, "Failed to load kernel.\n");
                exit(1);
        }
        source_str = (char *)malloc(MAX_SOURCE_SIZE);
        source_size = fread(source_str, 1, MAX_SOURCE_SIZE, fp);
        fclose(fp);

        cl_platform_id *platforms; // multiple
        cl_device_id *devices;
        cl_uint n_platforms;
        cl_int ret = clGetPlatformIDs(0, NULL, &n_platforms);
        platforms = new cl_platform_id[n_platforms];

        clGetPlatformIDs(n_platforms, platforms, NULL);

        //choose platform
        int deviceId = 1; // NVIDIA

        // cout << "Choose a platform/device[0-" << ret_num_platforms - 1 << "]:";
        // cin >> deviceId;
        char platName[100];
        clGetPlatformInfo(platforms[deviceId], CL_PLATFORM_NAME, sizeof(platName), platName, NULL);
        cout << "Platform: " << platName << endl;

        devices = new cl_device_id;
        ret = clGetDeviceIDs(platforms[deviceId], CL_DEVICE_TYPE_ALL, 1, devices, NULL);

        // Create an OpenCL context
        cl_context context = clCreateContext(NULL, 1, devices, NULL, NULL, &ret);

        // Create a command queue
        cl_command_queue command_queue = clCreateCommandQueue(context, devices[0], 0, &ret);

        // Create memory buffers on the device for each vector
        cl_mem a_mem_obj = clCreateBuffer(context, CL_MEM_READ_ONLY,
                                          NElements * sizeof(float), NULL, &ret);
        cl_mem b_mem_obj = clCreateBuffer(context, CL_MEM_READ_ONLY,
                                          NElements * sizeof(float), NULL, &ret);
        cl_mem c_mem_obj = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
                                          NElements * sizeof(float), NULL, &ret);

        cout << "Total Device Memory Allocated(MB): " << (NElements * sizeof(float) * 3) / 1000000 << endl;

        float *A = (float *)malloc(sizeof(float) * NElements);
        float *B = (float *)malloc(sizeof(float) * NElements);

        for (int i = 0; i < NElements; i++)
        {
                A[i] = i;
                B[i] = NElements - i;
        }

        // Copy the lists A and B to their respective memory buffers
        ret = clEnqueueWriteBuffer(command_queue, a_mem_obj, CL_TRUE, 0,
                                   NElements * sizeof(float), A, 0, NULL, NULL);
        ret = clEnqueueWriteBuffer(command_queue, b_mem_obj, CL_TRUE, 0,
                                   NElements * sizeof(float), B, 0, NULL, NULL);

        // Create a program from the kernel source
        cl_program program = clCreateProgramWithSource(context, 1,
                                                       (const char **)&source_str, (const size_t *)&source_size, &ret);

        // Build the program
        ret = clBuildProgram(program, 1, devices, NULL, NULL, NULL);

        // Create the OpenCL kernel
        cl_kernel kernel = clCreateKernel(program, "vector_add", &ret);

        // Set the arguments of the kernel
        ret = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&a_mem_obj);
        ret = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&b_mem_obj);
        ret = clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *)&c_mem_obj);

        // Execute the OpenCL kernel on the list
        size_t global_item_size = NElements; // Process the entire lists
        size_t local_item_size = 2;          // Process in groups of 64
        start = std::chrono::high_resolution_clock::now();
        ret = clEnqueueNDRangeKernel(command_queue, kernel, 1, NULL,
                                     &global_item_size, &local_item_size, 0, NULL, NULL);
        end = std::chrono::high_resolution_clock::now();

        if (ret != CL_SUCCESS)
        {
                std::cerr << getClErrorString(ret) << std::endl;
                exit(-1);
        }

        std::cout << "GPU Elapsed Time: " << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() << " us" << std::endl;

        // Read the memory buffer C on the device to the local variable C
        float *C = new float[NElements];
        ret = clEnqueueReadBuffer(command_queue, c_mem_obj, CL_TRUE, 0,
                                  NElements * sizeof(float), C, 0, NULL, NULL);

        // CPU
        float *C_cpu = new float[NElements];
        start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < NElements; ++i)
        {
                C_cpu[i] = A[i] + B[i];
        }
        end = std::chrono::high_resolution_clock::now();
        std::cout << "CPU Elapsed Time: " << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() << " us" << std::endl;

        // Checking Results
        cout << "Cheking Results..." << endl;
        bool allGood = true;
        for (int i = 0; i < NElements; ++i)
        {
                if (C_cpu[i] != C[i])
                {
                        std::cout << "Wrong Results @ i=" << i
                                  << " (" << C_cpu[i] << " != " << C[i] << ")\n";
                        allGood = false;
                        break;
                }
        }

        if (allGood)
                cout << "All Good!" << endl;

        cin.get();
        // Clean up
        ret = clFlush(command_queue);
        ret = clFinish(command_queue);
        ret = clReleaseKernel(kernel);
        ret = clReleaseProgram(program);
        ret = clReleaseMemObject(a_mem_obj);
        ret = clReleaseMemObject(b_mem_obj);
        ret = clReleaseMemObject(c_mem_obj);
        ret = clReleaseCommandQueue(command_queue);
        ret = clReleaseContext(context);
        free(A);
        free(B);
        free(C);
        return 0;
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
