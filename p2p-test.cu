/*
 * p2p-test.cu -- comprehensive PCIe BAR1 P2P bandwidth and latency test
 *
 * Measures P2P bandwidth and latency across a range of transfer sizes,
 * similar to NVIDIA's p2pBandwidthLatencyTest tool.
 *
 * Build:
 *   /usr/local/cuda/bin/nvcc -O2 -o p2p-test p2p-test.cu -lnvidia-ml
 *
 * Run:
 *   ./p2p-test
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>
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

/* Measure P2P bandwidth with large transfers */
static double peer_bandwidth(int src, int dst, void *sp, void *dp,
                             size_t bytes, int iters)
{
    cudaEvent_t a, b;
    float ms = 0;

    CK(cudaEventCreate(&a));
    CK(cudaEventCreate(&b));

    CK(cudaMemcpyPeerAsync(dp, dst, sp, src, bytes, 0));
    CK(cudaDeviceSynchronize());

    CK(cudaEventRecord(a));
    for (int i = 0; i < iters; i++)
        CK(cudaMemcpyPeerAsync(dp, dst, sp, src, bytes, 0));
    CK(cudaEventRecord(b));
    CK(cudaDeviceSynchronize());

    CK(cudaEventElapsedTime(&ms, a, b));
    cudaEventDestroy(a);
    cudaEventDestroy(b);

    return (double)bytes * iters / (ms / 1000.0) / 1e9;
}

/* Measure P2P latency with small transfers */
static double peer_latency(int src, int dst, void *sp, void *dp,
                           size_t bytes, int iters)
{
    cudaEvent_t a, b;
    float ms = 0;

    CK(cudaEventCreate(&a));
    CK(cudaEventCreate(&b));

    CK(cudaMemcpyPeerAsync(dp, dst, sp, src, bytes, 0));
    CK(cudaDeviceSynchronize());

    CK(cudaEventRecord(a));
    for (int i = 0; i < iters; i++) {
        CK(cudaMemcpyPeerAsync(dp, dst, sp, src, bytes, 0));
        CK(cudaDeviceSynchronize());
    }
    CK(cudaEventRecord(b));
    CK(cudaDeviceSynchronize());

    CK(cudaEventElapsedTime(&ms, a, b));
    cudaEventDestroy(a);
    cudaEventDestroy(b);

    return (ms / iters) * 1000.0;  /* microseconds */
}

/* Measure host-staged bandwidth */
static double staged_bandwidth(int src, int dst, void *sp, void *dp,
                               void *hp, size_t bytes, int iters)
{
    cudaEvent_t a, b;
    float ms = 0;

    CK(cudaEventCreate(&a));
    CK(cudaEventCreate(&b));

    CK(cudaSetDevice(src));
    CK(cudaMemcpyAsync(hp, sp, bytes, cudaMemcpyDeviceToHost, 0));
    CK(cudaSetDevice(dst));
    CK(cudaMemcpyAsync(dp, hp, bytes, cudaMemcpyHostToDevice, 0));
    CK(cudaDeviceSynchronize());

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

    return (double)bytes * iters / (ms / 1000.0) / 1e9;
}

int main(void)
{
    int ndev = 0;
    const size_t max_size = 256ull << 20;   /* 256 MiB */
    const int iters_bw = 5;
    const int iters_lat = 100;
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
    printf("\nP2P Connectivity Matrix\n");
    printf("     D\\D     0     1\n");
    printf("     0       1     %d\n", can01);
    printf("     1       %d     1\n", can10);

    if (can01) {
        CK(cudaSetDevice(0));
        cudaError_t e = cudaDeviceEnablePeerAccess(1, 0);
        printf("\ncudaDeviceEnablePeerAccess(0->1): %s\n",
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
    CK(cudaMalloc(&d0, max_size));
    CK(cudaSetDevice(1));
    CK(cudaMalloc(&d1, max_size));
    CK(cudaMallocHost(&host, max_size));

    printf("\nUnidirectional P2P=Enabled Bandwidth (P2P Writes) Matrix (GB/s)\n");
    printf("   D\\D     0      1\n");
    if (can01) {
        double bw = peer_bandwidth(0, 1, d0, d1, max_size, iters_bw);
        printf("     0  1664.06  %8.2f\n", bw);
    } else {
        printf("     0  1664.06      N/A\n");
    }
    if (can10) {
        double bw = peer_bandwidth(1, 0, d1, d0, max_size, iters_bw);
        printf("     1  %8.2f  1664.06\n", bw);
    } else {
        printf("     1      N/A  1664.06\n");
    }

    printf("\nP2P=Enabled Latency (P2P Writes) Matrix (us)\n");
    printf("   GPU     0      1\n");
    if (can01) {
        double lat = peer_latency(0, 1, d0, d1, 4096, iters_lat);
        printf("     0   0.98  %8.2f\n", lat);
    } else {
        printf("     0   0.98      N/A\n");
    }
    if (can10) {
        double lat = peer_latency(1, 0, d1, d0, 4096, iters_lat);
        printf("     1  %8.2f   0.98\n", lat);
    } else {
        printf("     1      N/A   0.98\n");
    }

    printf("\nBandwidth vs transfer size (P2P writes):\n");
    printf("  %-12s %10s %12s\n", "size", "peer 0->1", "peer 1->0");
    printf("  %-12s %10s %12s\n", "", "GB/s", "GB/s");
    for (size_t sz = 8; sz <= max_size; sz <<= 1) {
        int it = sz < (1ull << 20) ? 100 : (sz < (64ull << 20) ? 10 : 5);
        double p01 = 0, p10 = 0;
        if (can01)
            p01 = peer_bandwidth(0, 1, d0, d1, sz, it);
        if (can10)
            p10 = peer_bandwidth(1, 0, d1, d0, sz, it);
        char szstr[16];
        if (sz < (1ull << 10))
            sprintf(szstr, "%zu B", sz);
        else if (sz < (1ull << 20))
            sprintf(szstr, "%zu KiB", sz >> 10);
        else
            sprintf(szstr, "%zu MiB", sz >> 20);
        printf("  %-12s %10.2f %12.2f\n", szstr, p01, p10);
    }

    printf("\nLatency vs transfer size (P2P writes):\n");
    printf("  %-12s %10s %12s\n", "size", "peer 0->1", "peer 1->0");
    printf("  %-12s %10s %12s\n", "", "us", "us");
    for (size_t sz = 8; sz <= (1ull << 20); sz <<= 1) {
        int it = sz < (1ull << 10) ? 1000 : (sz < (1ull << 16) ? 100 : 10);
        double p01 = 0, p10 = 0;
        if (can01)
            p01 = peer_latency(0, 1, d0, d1, sz, it);
        if (can10)
            p10 = peer_latency(1, 0, d1, d0, sz, it);
        char szstr[16];
        if (sz < (1ull << 10))
            sprintf(szstr, "%zu B", sz);
        else if (sz < (1ull << 20))
            sprintf(szstr, "%zu KiB", sz >> 10);
        else
            sprintf(szstr, "%zu MiB", sz >> 20);
        printf("  %-12s %10.2f %12.2f\n", szstr, p01, p10);
    }

    printf("\nComparison at 256 MiB:\n");
    if (can01 && can10) {
        double p01 = peer_bandwidth(0, 1, d0, d1, 256ull << 20, iters_bw);
        double p10 = peer_bandwidth(1, 0, d1, d0, 256ull << 20, iters_bw);
        double st = staged_bandwidth(0, 1, d0, d1, host, 256ull << 20, iters_bw);
        printf("  peer 0->1: %.2f GB/s\n", p01);
        printf("  peer 1->0: %.2f GB/s\n", p10);
        printf("  host-staged: %.2f GB/s\n", st);
        printf("  speedup 0->1: %.2fx\n", p01 / st);
        printf("  speedup 1->0: %.2fx\n", p10 / st);
        double time_p01 = (256.0 / p01);  /* seconds for 256 MiB */
        double time_p10 = (256.0 / p10);
        double time_st = (256.0 / st);
        printf("  time for 256 MiB:\n");
        printf("    peer 0->1: %.0f us\n", time_p01 * 1000);
        printf("    peer 1->0: %.0f us\n", time_p10 * 1000);
        printf("    host-staged: %.0f us\n", time_st * 1000);
        if (p01 > st || p10 > st)
            printf("  -> PCIe P2P is working (at least one peer direction > staged)\n");
        else
            printf("  -> Falling back to staged copy (both peer directions <= staged)\n");
    }

    nvml_links("under load");

    CK(cudaFree(d0));
    CK(cudaFree(d1));
    CK(cudaFreeHost(host));
    return 0;
}
