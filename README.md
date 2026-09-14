# NVIDIA driver 610.57.04 with P2P for RTX 30, 40, and 50 series

Patched NVIDIA kernel modules for GPU-to-GPU transfers over PCIe BAR1 or NVLink,
including between generations. No extra NVIDIA module parameters are needed.

Based on [tinygrad's P2P patch](https://github.com/tinygrad/open-gpu-kernel-modules/blob/550.54.15-p2p/README.md).

## Supported configurations

The patch supports RTX 30-, 40-, and 50-series GPUs, including models below the
3090, 4090, and 5090.

For PCIe P2P, each GPU needs Resizable BAR and a BAR1 aperture large enough to map
its usable VRAM. The IOMMU must be configured for passthrough. Consumer Turing
(RTX 20 series) is not supported by the BAR1 path.

Same-generation pairs only need the patched kernel modules. The models can differ,
such as RTX 5090 and RTX PRO 6000 Blackwell. Mixed-generation pairs, such as RTX 3090
and RTX 5090, also need the `libcuda` patch below.

NVLink is used where available; other pairs use PCIe BAR1. This also works in systems
with both NVLink-connected pairs and GPUs without NVLink.

## How it works

BAR1 exposes GPU memory over PCIe so another GPU can read and write it directly.
Unlike NVIDIA's proprietary PCIe P2P protocol, this works across GPU generations.

With P2P, GPU B reads GPU A's memory directly over the PCIe fabric (or NVLink if
present). The CPU and host RAM are not involved in the data path, eliminating
the bottleneck of staging through system memory.

| Metric | Staged copy (no P2P) | PCIe P2P (BAR1) | NVLink P2P |
|--------|---------------------|-----------------|------------|
| Path | GPU → Host RAM → GPU | GPU ↔ PCIe ↔ GPU | GPU ↔ NVLink ↔ GPU |
| Bandwidth | Limited by host memory controller | Limited by PCIe link (e.g. Gen4 x4 ≈ 6 GB/s, x16 ≈ 25 GB/s per direction) | Limited by NVLink version (e.g. NVLink 2.0 ≈ 300 GB/s) |
| CPU involvement | DMA setup only | DMA setup only | DMA setup only |
| Cross-generation | Yes | Yes (with this patch) | Limited |

Actual bandwidth depends on the PCIe topology. GPUs behind a chipset PCH (e.g. Intel Z790, AMD X870)
are limited by the PCH's PCIe link, which is often Gen3 x4 or Gen4 x4. For best P2P bandwidth,
both GPUs should be on CPU-attached PCIe slots.

When the firmware console is at physical VRAM offset zero, it can share the static
BAR1 mapping instead of taking up a separate window. Other layouts keep the driver's
separate console mapping.

> [!WARNING]
> IOMMU passthrough (`iommu=pt`) removes DMA isolation. Do not use it with untrusted devices.

## How to use

1. Enable Above 4G Decoding and Resizable BAR in system firmware.
2. Enable DMA passthrough mode for the IOMMU:
   - Edit `/etc/default/grub`
   - Add `amd_iommu=on iommu=pt` to `GRUB_CMDLINE_LINUX_DEFAULT` (use `intel_iommu=on iommu=pt` on Intel)
   - Run `sudo update-grub`
3. Install the [NVIDIA 610.57.04 driver](https://www.nvidia.com/en-us/drivers/details/274513/).
4. Run `./install.sh` in this repo.

   The install script handles several subtleties that a bare `make modules_install`
   would miss:

   - **Secure Boot module signing**: On systems with Secure Boot enabled and kernel
     lockdown, modules must be signed with a key enrolled in the firmware. The script
     signs all modules after installation using your MOK key pair (default:
     `/var/lib/shim-signed/mok/MOK.{priv,der}`).
   - **DKMS shadowing**: The NVIDIA `.run` installer's DKMS registration keeps copies
     in `/lib/modules/$(uname -r)/updates/dkms`, which `depmod` prefers over the
     `kernel/drivers/video` location. The script removes these competing copies so
     `modprobe nvidia` loads the patched build, not the stock one.
   - **Module reload**: The script unloads old modules and loads the new ones, so
     `nvidia-smi` reports the freshly installed driver rather than a previously
     loaded one.
   - **Ownership**: Running `sudo make` leaves root-owned artifacts in the checkout.
     The script fixes ownership after privileged steps so the next build works.

   Options: `SKIP_SIGN=1` (skip signing), `SKIP_BUILD=1` (reuse existing build),
   `JOBS=N` (override parallelism).

   To verify P2P is working, run the included bandwidth test:

   ```bash
   $ ./p2p-check
   CUDA devices: 2
     gpu0: NVIDIA GeForce RTX 3090  (bus 01:00.0, 24101 MiB)
     gpu1: NVIDIA GeForce RTX 3090  (bus 0d:00.0, 24123 MiB)

   peer access supported:  gpu0->gpu1 YES   gpu1->gpu0 YES
   cudaDeviceEnablePeerAccess(0->1): ok
   cudaDeviceEnablePeerAccess(1->0): ok

   size            peer 0->1    peer 1->0  host-staged
                    GB/s         GB/s         GB/s
   64 MiB               2.33         6.58         4.30
   256 MiB              2.33         6.59         4.30

   reading:
     peer bandwidth clearly ABOVE host-staged = PCIe P2P is working
     peer bandwidth at or below host-staged    = falling back to a
     staged copy, i.e. no real P2P
   ```

   If peer bandwidth is at or below host-staged, check that your IOMMU is in
   passthrough mode and that ACS is disabled (see "Potential issues" below).

5. For mixed-generation P2P, back up and patch the system `libcuda` as described below.
6. Reboot the server.

## Mixed-generation `libcuda` patch

Set `LIBCUDA` to your installed library, back it up, and run
[`patch-libcuda-p2p.py`](patch-libcuda-p2p.py):

```
LIBCUDA=/usr/lib/x86_64-linux-gnu/libcuda.so.1
sudo cp "$LIBCUDA" "$LIBCUDA.bak"
sudo ./patch-libcuda-p2p.py "$LIBCUDA"
```

## Forcing 3090s to use PCIe instead of NVLink

To use PCIe BAR1 even with an NVLink bridge installed, add this to
`/etc/modprobe.d/nvidia.conf`:

```
options nvidia NVreg_RegistryDwords="RMForceP2PType=1"
```

## Experimental: faster cudaHostRegister for hugepage-backed memory

An experimental fast path speeds up `cudaHostRegister` for buffers backed by hugetlb
pages (including 2 MiB and 1 GiB) and reduces their device page tables. It is enabled
automatically for hugepage-aligned buffers covering whole hugepages within one VMA.
Transparent hugepages and other layouts use the normal registration path.

## Potential issues

If P2P transfers are slow, make sure your IOMMU is in passthrough (`pt`) mode and that ACS
is disabled. ACS on root ports forces all GPU-to-GPU traffic through the CPU root complex,
killing P2P bandwidth. ACS can be disabled in BIOS, with the
`pcie_acs_override=downstream,multifunction` kernel parameter (if your kernel supports it),
or with an ACS override patch applied to the kernel.

## Sample benchmark results

### p2pBandwidthLatencyTest (9-GPU system)

9-GPU system (1x RTX PRO 6000 Blackwell + 8x RTX 5090) on a dual-socket AMD EPYC 9575F (Turin):

```
P2P Connectivity Matrix
     DD     0     1     2     3     4     5     6     7     8
     0	     1     1     1     1     1     1     1     1     1
     1	     1     1     1     1     1     1     1     1     1
     ...  (all pairs connected)

Unidirectional P2P=Enabled Bandwidth (P2P Writes) Matrix (GB/s)
   DD     0      1      2      3      4      5      6      7      8 
     0 1617.49  55.59  55.62  55.64  55.64  56.58  56.58  56.58  56.58 
     1  55.60 1656.95  56.57  56.55  56.57  55.63  55.63  55.62  55.64 
     ...

Bidirectional P2P=Enabled Bandwidth Matrix (GB/s)
   DD     0      1      2      3      4      5      6      7      8 
     0 1600.87 111.17 111.04 111.14 111.12 111.35 111.34 111.39 111.38 
     1 111.12 1636.90 111.38 111.39 111.34 111.10 111.08 111.11 111.08 
     ...

P2P=Enabled Latency (P2P Writes) Matrix (us)
   GPU     0      1      2      3      4      5      6      7      8 
     0   1.02   0.38   0.42   0.36   0.37   0.37   0.44   0.36   0.36 
     1   0.45   0.98   0.37   0.44   0.38   0.43   0.45   0.45   0.39 
     ...
```

### nccl-tests all_reduce_perf (8x RTX 5090)

```
#                                                              out-of-place                       in-place          
#       size         count      type   redop    root     time   algbw   busbw  #wrong     time   algbw   busbw  #wrong 
#        (B)    (elements)                               (us)  (GB/s)  (GB/s)             (us)  (GB/s)  (GB/s)         
            8             2     float     sum      -1    25.47    0.00    0.00       0    24.82    0.00    0.00       0
      ...
     33554432       8388608     float     sum      -1  1284.81   26.12   45.70       0  1286.28   26.09   45.65       0
    134217728      33554432     float     sum      -1  5355.16   25.06   43.86       0  5329.43   25.18   44.07       0
```

---

Based on NVIDIA Linux Open GPU Kernel Module Source 610.57.04.
See the [upstream README](https://github.com/NVIDIA/open-gpu-kernel-modules)
for build instructions, supported architectures, toolchains, and the full device table.
