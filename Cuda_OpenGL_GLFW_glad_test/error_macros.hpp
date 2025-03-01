
#ifndef _ERRORS_1180779_
#define _ERRORS_1180779_

#define ERROR_(source) perror(source), fprintf(stderr, "file: %s, line: %d\n", __FILE__, __LINE__), exit(-1)
#define ERROR_CUDA(operation) do { \
            cudaError_t status = operation; \
            if(status != cudaSuccess) \
            { \
                fprintf(stderr, "error: %s\n",  cudaGetErrorString(status)); \
                fprintf(stderr, "file: %s, line: %d\n", __FILE__, __LINE__); \
                exit(-1); \
            } \
        } while(0)

#endif