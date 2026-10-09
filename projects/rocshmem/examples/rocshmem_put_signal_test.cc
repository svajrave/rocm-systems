/******************************************************************************
 * Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 *****************************************************************************/

/*
 * First find your offload target, and if xnack is enabled/disabled using

   rocminfo | grep amdgcn

 * It should output a string like so:

   "Name:                    amdgcn-amd-amdhsa--gfx942:sramecc+:xnack-"

 * This lists the offload taret (gfx942) and that xnack is disabled (xnack-).
 * Therefore, we need to specify --offload-arch=gfx942:xnack- to our link and compile commands.
 * Please modify the compile and link commands to suit your system

 * To compile:
   hipcc -c -fgpu-rdc -x hip rocshmem_put_signal_test.cc \
         --offload-arch=<target>:<xnack>                 \
         -I/opt/rocm/include                             \
         -I$ROCSHMEM_INSTALL_DIR/include                 \
         -I$OPENMPI_UCX_INSTALL_DIR/include/

 * To link:
   hipcc -fgpu-rdc --hip-link rocshmem_put_signal_test.o -o rocshmem_put_signal_test \
         --offload-arch=<target>:<xnack>                                             \
         $ROCSHMEM_INSTALL_DIR/lib/librocshmem.a                                     \
         $OPENMPI_UCX_INSTALL_DIR/lib/libmpi.so                                      \
         -L/opt/rocm/lib -lamdhip64 -lhsa-runtime64

 * To run:
   mpirun -np 8 -x ROCSHMEM_MAX_NUM_CONTEXTS=2 ./rocshmem_put_signal_test [nelem] [iterations]

 * After the correctness check, PE 0 times blocking putmem + quiet and put_signal + quiet to
 * PE 1 and prints the average latency per operation (default 100 iterations).

 */

#include <rocshmem/rocshmem.hpp>

#include "util.h"

using namespace rocshmem;

__global__ void simple_put_signal_test(uint64_t *data, uint64_t *message, size_t nelem,
                                       uint64_t *sig_addr, int my_pe, int dst_pe)
{

    int threadId = blockIdx.x * blockDim.x + threadIdx.x;

    if (threadId == 0) {
        if (my_pe == 0) {
            rocshmem_ulong_put_signal(data, message, nelem, sig_addr, 1, ROCSHMEM_SIGNAL_SET, dst_pe);
        }
        else {
            rocshmem_ulong_wait_until(sig_addr, ROCSHMEM_CMP_EQ, 1);
            rocshmem_ulong_put_signal(data, data, nelem, sig_addr, 1, ROCSHMEM_SIGNAL_SET, dst_pe);
        }
    }

    __syncthreads();
}

/* PE 0 only: average latency of putmem + quiet and of put_signal + quiet to dst_pe.
 * ticks[0] = putmem, ticks[1] = put_signal. */
__global__ void put_latency_test(uint64_t *data, uint64_t *message, size_t nelem, uint64_t *sig_addr,
                                 int iters, int dst_pe, long long *ticks)
{
    if (blockIdx.x * blockDim.x + threadIdx.x == 0 && rocshmem_my_pe() == 0) {
        size_t nbytes = nelem * sizeof(uint64_t);

        for (int i = 0; i < 10; i++) {  /* warm up */
            rocshmem_putmem(data, message, nbytes, dst_pe);
            rocshmem_quiet();
        }
        long long start = wall_clock64();
        for (int i = 0; i < iters; i++) {
            rocshmem_putmem(data, message, nbytes, dst_pe);
            rocshmem_quiet();
        }
        ticks[0] = wall_clock64() - start;

        for (int i = 0; i < 10; i++) {  /* warm up */
            rocshmem_ulong_put_signal(data, message, nelem, sig_addr, 1, ROCSHMEM_SIGNAL_SET, dst_pe);
            rocshmem_quiet();
        }
        start = wall_clock64();
        for (int i = 0; i < iters; i++) {
            rocshmem_ulong_put_signal(data, message, nelem, sig_addr, 1, ROCSHMEM_SIGNAL_SET, dst_pe);
            rocshmem_quiet();
        }
        ticks[1] = wall_clock64() - start;
    }
}

#define MAX_ELEM 256

int main (int argc, char **argv)
{
    int nelem = MAX_ELEM;

    if (argc > 1) {
        nelem = atoi(argv[1]);
    }

    CHECK_HIP(hipSetDevice(get_launcher_local_rank()));

    rocshmem_init();

    int my_pe = rocshmem_my_pe();
    int npes =  rocshmem_n_pes();

    int dst_pe = (my_pe + 1) % npes;
    uint64_t *message = (uint64_t*)rocshmem_malloc(nelem * sizeof(uint64_t));
    uint64_t *data = (uint64_t*)rocshmem_malloc(nelem * sizeof(uint64_t));
    uint64_t *sig_addr = (uint64_t*)rocshmem_malloc(sizeof(uint64_t));
    if (NULL == data || NULL == message || NULL == sig_addr) {
        std::cout << "Error allocating memory from symmetric heap" << std::endl;
        std::cout << "data: " << data
                  << ", message: " << message
                  << ", size: " << sizeof(uint64_t) * nelem
                  << ", sig_addr: " << sig_addr
                  << std::endl;
        rocshmem_global_exit(1);
    }

    for (int i=0; i<nelem; i++) {
        message[i] = my_pe;
    }

    CHECK_HIP(hipMemset(data, 0, (nelem * sizeof(uint64_t))));
    CHECK_HIP(hipDeviceSynchronize());

    int threadsPerBlock=256;
    simple_put_signal_test<<<dim3(1), dim3(threadsPerBlock), 0, 0>>>(data, message, nelem, sig_addr, my_pe, dst_pe);
    rocshmem_barrier_all();
    CHECK_HIP(hipDeviceSynchronize());

    bool pass = true;
    for (int i=0; i<nelem; i++) {
        if (data[i] != 0) {
            pass = false;
#if VERBOSE
            printf("[%d] Error in element %d expected 0 got %d\n", my_pe, i, dst[i]);
#endif
        }
    }
    printf("[%d] Test %s \t %s\n", my_pe, argv[0], pass ? "[PASS]" : "[FAIL]");

    int iters = (argc > 2) ? atoi(argv[2]) : 100;
    long long *d_ticks;
    CHECK_HIP(hipMalloc(&d_ticks, 2 * sizeof(long long)));
    CHECK_HIP(hipMemset(d_ticks, 0, 2 * sizeof(long long)));
    rocshmem_barrier_all();
    put_latency_test<<<dim3(1), dim3(64), 0, 0>>>(data, message, nelem, sig_addr, iters, dst_pe, d_ticks);
    CHECK_HIP(hipDeviceSynchronize());
    rocshmem_barrier_all();
    if (my_pe == 0) {
        long long ticks[2] = {0, 0};
        CHECK_HIP(hipMemcpy(ticks, d_ticks, sizeof(ticks), hipMemcpyDeviceToHost));
        printf("[PE 0] putmem latency: %.2f us per op (avg of %d iterations, %zu bytes to PE %d, blocking putmem + quiet)\n",
               wall_clock_ticks_to_us(ticks[0]) / iters, iters, nelem * sizeof(uint64_t), dst_pe);
        printf("[PE 0] put_signal latency: %.2f us per op (avg of %d iterations, %zu bytes to PE %d, put_signal + quiet)\n",
               wall_clock_ticks_to_us(ticks[1]) / iters, iters, nelem * sizeof(uint64_t), dst_pe);
    }
    CHECK_HIP(hipFree(d_ticks));

    rocshmem_free(data);
    rocshmem_free(message);
    rocshmem_finalize();
    return 0;
}
