/*
 * p2p-check.cu -- does PCIe BAR1 P2P actually work between the GPUs on this
 * box, and how fast is it?
 *
 * Answers three questions the driver/topology work could not:
 *   1. Does the patched driver let CUDA do peer access (cudaDeviceCanAccessPeer)?
 *   2. What is the measured peer-copy bandwidth (GPU->GPU over PCIe BAR1)?
 *   3. How does that compare with a host-staged copy, i.e. is P2P actually
 *      being used or silently falling back?
 *
 * It also prints NVML link gen/width before and after the transfer, because
 * these GPUs downshift their PCIe link when idle -- so a link reading taken
 * while nothing is running tells you nothing.
 *
 * Build (nvcc is not on PATH here):
 *   /usr/local/cuda/bin/nvcc -O2 -o p2p-check p2p-check.cu -lnvidia-ml
 *
 * Run:
 *   ./p2p-check
 * CUDA devices: 2
 *   gpu0: NVIDIA GeForce RTX 3090  (bus 01:00.0, 24101 MiB)
 *   gpu1: NVIDIA GeForce RTX 3090  (bus 0d:00.0, 24123 MiB)
 *   [idle           ] NVML link (current/max):  gpu0 gen1/4 x8/16  gpu1 gen1/4 x4/16
 * 
 * peer access supported:  gpu0->gpu1 YES   gpu1->gpu0 YES
 * cudaDeviceEnablePeerAccess(0->1): ok
 * cudaDeviceEnablePeerAccess(1->0): ok
 * 
 * size            peer 0->1    peer 1->0  host-staged
 *                      GB/s         GB/s         GB/s
 * 64 MiB               2.33         6.58         4.30
 * 256 MiB              2.33         6.59         4.30
 *   [under load     ] NVML link (current/max):  gpu0 gen4/4 x8/16  gpu1 gen4/4 x4/16
 * 
 * reading:
 *   peer bandwidth clearly ABOVE host-staged = PCIe P2P is working
 *   peer bandwidth at or below host-staged    = falling back to a
 *   staged copy, i.e. no real P2P
 */

#include <cstdio>
#include <cstdlib>
#include <string>
#include <cuda_runtime.h>
#include <nvml.h>

#define CK(x) do {                                                            \
        cudaError_t _e = (x);                                                 \
        if (_e != cudaSuccess) {                                              \
            printf("CUDA error: %s  (%s:%d)\n", cudaGetErrorString(_e),       \
                   __FILE__, __LINE__);                                       \
            exit(1);                                                          \
        }                                                                     \
    } while (0)

static void nvml_links(const char *when)
{
    nvmlDevice_t dev;
    unsigned n = 0;

    if (nvmlInit_v2() != NVML_SUCCESS) {
        printf("  [%-15s] NVML unavailable\n", when);
        return;
    }
    nvmlDeviceGetCount_v2(&n);
    printf("  [%-15s] NVML link (current/max):", when);
    for (unsigned i = 0; i < n; i++) {
        unsigned gen = 0, maxgen = 0, width = 0, maxwidth = 0;
        if (nvmlDeviceGetHandleByIndex_v2(i, &dev) != NVML_SUCCESS)
            continue;
        nvmlDeviceGetCurrPcieLinkGeneration(dev, &gen);
        nvmlDeviceGetMaxPcieLinkGeneration(dev, &maxgen);
        nvmlDeviceGetCurrPcieLinkWidth(dev, &width);
        nvmlDeviceGetMaxPcieLinkWidth(dev, &maxwidth);
        printf("  gpu%u gen%u/%u x%u/%u", i, gen, maxgen, width, maxwidth);
    }
    printf("\n");
    nvmlShutdown();
}

static double peer_copy(int src, int dst, void *sp, void *dp, size_t bytes,
                        int iters)
{
    cudaEvent_t a, b;
    float ms = 0;

    CK(cudaEventCreate(&a));
    CK(cudaEventCreate(&b));

    CK(cudaMemcpyPeerAsync(dp, dst, sp, src, bytes, 0));   /* warm up */
    CK(cudaDeviceSynchronize());

    CK(cudaEventRecord(a));
    for (int i = 0; i < iters; i++)
        CK(cudaMemcpyPeerAsync(dp, dst, sp, src, bytes, 0));
    CK(cudaEventRecord(b));
    CK(cudaDeviceSynchronize());

    CK(cudaEventElapsedTime(&ms, a, b));
    cudaEventDestroy(a);
    cudaEventDestroy(b);

    return (double)bytes * iters / (ms / 1000.0) / 1e9;    /* GB/s */
}

static double staged_copy(int src, int dst, void *sp, void *dp, void *hp,
                          size_t bytes, int iters)
{
    cudaEvent_t a, b;
    float ms = 0;

    CK(cudaEventCreate(&a));
    CK(cudaEventCreate(&b));

    CK(cudaSetDevice(src));
    CK(cudaMemcpyAsync(hp, sp, bytes, cudaMemcpyDeviceToHost, 0));
    CK(cudaSetDevice(dst));
    CK(cudaMemcpyAsync(dp, hp, bytes, cudaMemcpyHostToDevice, 0));
    CK(cudaDeviceSynchronize());                           /* warm up */

    CK(cudaEventRecord(a));
    for (int i = 0; i < iters; i++) {
        CK(cudaSetDevice(src));
        CK(cudaMemcpyAsync(hp, sp, bytes, cudaMemcpyDeviceToHost, 0));
        CK(cudaStreamSynchronize(0));
        CK(cudaSetDevice(dst));
        CK(cudaMemcpyAsync(dp, hp, bytes, cudaMemcpyHostToDevice, 0));
        CK(cudaStreamSynchronize(0));
    }
    CK(cudaEventRecord(b));
    CK(cudaDeviceSynchronize());

    CK(cudaEventElapsedTime(&ms, a, b));
    cudaEventDestroy(a);
    cudaEventDestroy(b);

    return (double)bytes * iters / (ms / 1000.0) / 1e9;    /* GB/s */
}

int main(void)
{
    int ndev = 0;
    const size_t large = 256ull << 20;   /* 256 MiB */
    const size_t small = 64ull << 20;    /*  64 MiB */
    const int iters = 10;
    void *d0 = NULL, *d1 = NULL, *host = NULL;
    int can01 = 0, can10 = 0;

    {
        cudaError_t e = cudaGetDeviceCount(&ndev);
        if (e != cudaSuccess) {
            printf("cudaGetDeviceCount failed: %s\n", cudaGetErrorString(e));
            printf("If this is a sandboxed/restricted process, run it from a "
                   "normal shell:\nthe GPU device nodes are not openable "
                   "there.\n");
            return 1;
        }
    }
    printf("CUDA devices: %d\n", ndev);
    if (ndev < 2) {
        printf("Need two CUDA devices for a P2P test.\n");
        return 1;
    }
    for (int i = 0; i < ndev; i++) {
        cudaDeviceProp p;
        CK(cudaGetDeviceProperties(&p, i));
        printf("  gpu%d: %s  (bus %02x:%02x.0, %d MiB)\n", i, p.name,
               p.pciBusID, p.pciDeviceID, (int)(p.totalGlobalMem >> 20));
    }

    nvml_links("idle");

    /* 1. does the driver permit peer access? */
    CK(cudaDeviceCanAccessPeer(&can01, 0, 1));
    CK(cudaDeviceCanAccessPeer(&can10, 1, 0));
    printf("\npeer access supported:  gpu0->gpu1 %s   gpu1->gpu0 %s\n",
           can01 ? "YES" : "no", can10 ? "YES" : "no");

    if (can01) {
        CK(cudaSetDevice(0));
        cudaError_t e = cudaDeviceEnablePeerAccess(1, 0);
        printf("cudaDeviceEnablePeerAccess(0->1): %s\n",
               e == cudaSuccess ? "ok" : cudaGetErrorString(e));
        cudaGetLastError();
    }
    if (can10) {
        CK(cudaSetDevice(1));
        cudaError_t e = cudaDeviceEnablePeerAccess(0, 0);
        printf("cudaDeviceEnablePeerAccess(1->0): %s\n",
               e == cudaSuccess ? "ok" : cudaGetErrorString(e));
        cudaGetLastError();
    }

    /* 2. allocate and measure */
    CK(cudaSetDevice(0));
    CK(cudaMalloc(&d0, large));
    CK(cudaSetDevice(1));
    CK(cudaMalloc(&d1, large));
    CK(cudaMallocHost(&host, large));          /* pinned, for a fair staged test */

    /* Query PCIe link widths for labeling */
    unsigned w0 = 0, w1 = 0;
    {
        nvmlDevice_t dev;
        if (nvmlInit_v2() == NVML_SUCCESS) {
            if (nvmlDeviceGetHandleByIndex_v2(0, &dev) == NVML_SUCCESS)
                nvmlDeviceGetCurrPcieLinkWidth(dev, &w0);
            if (nvmlDeviceGetHandleByIndex_v2(1, &dev) == NVML_SUCCESS)
                nvmlDeviceGetCurrPcieLinkWidth(dev, &w1);
            nvmlShutdown();
        }
    }

    printf("\ngpu0 link: x%u  gpu1 link: x%u\n", w0, w1);
    printf("  (P2P bandwidth limited by the slower link in the path)\n\n");

    printf("%-14s %10s %12s %12s\n", "size", "peer 0->1", "peer 1->0",
           "host-staged");
    printf("%-14s %10s %12s %12s\n", "", "GB/s", "GB/s", "GB/s");

    for (int pass = 0; pass < 2; pass++) {
        size_t bytes = pass == 0 ? small : large;
        int it = pass == 0 ? iters : 5;
        double p01 = 0, p10 = 0, st = 0;

        if (can01)
            p01 = peer_copy(0, 1, d0, d1, bytes, it);
        if (can10)
            p10 = peer_copy(1, 0, d1, d0, bytes, it);
        st = staged_copy(0, 1, d0, d1, host, bytes, it);

        printf("%-14s %10.2f %12.2f %12.2f\n",
               pass == 0 ? "64 MiB" : "256 MiB",
               p01, p10, st);
    }

    nvml_links("under load");

    printf("\nreading:\n"
           "  peer bandwidth clearly ABOVE host-staged = PCIe P2P is working\n"
           "  peer bandwidth at or below host-staged    = falling back to a\n"
           "  staged copy, i.e. no real P2P\n");

    CK(cudaFree(d0));
    CK(cudaFree(d1));
    CK(cudaFreeHost(host));
    return 0;
}
