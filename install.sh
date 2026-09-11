#!/bin/bash
#
# Build, install, sign and load the locally built NVIDIA open kernel modules.
#
# Why this is more than `make modules_install`:
#
#   * This machine boots with Secure Boot enabled and the kernel is in
#     lockdown=integrity.  The kernel then rejects every module that is not
#     signed with a key enrolled in the firmware.  Unsigned modules (or modules
#     signed with a key that was never enrolled) fail with
#         "Loading of module with unavailable key is rejected"
#     in the kernel log, no /dev/nvidia* node appears, and nvidia-smi answers
#     "NVIDIA-SMI has failed because it couldn't communicate with the NVIDIA
#     driver".  So the modules must be signed *after* they are installed.
#
#   * `make modules_install` writes to /lib/modules/$(uname -r)/kernel/drivers/
#     video, while the NVIDIA .run installer's DKMS registration keeps copies in
#     /lib/modules/$(uname -r)/updates/dkms.  depmod prefers updates/, so
#     `modprobe nvidia` would load the *other* build while nvidia-modeset.ko /
#     nvidia-drm.ko come from this build -- a mixed module set where the custom
#     kernel changes are not the ones actually running.
#
#   * Unloading the old modules and reloading the new ones is part of the job;
#     a bare `nvidia-smi` at the end reports the previously loaded driver.
#
# Usage:
#   ./install.sh              build, install, sign, load, verify
#   SKIP_SIGN=1 ./install.sh  skip signing (only for a Secure Boot disabled box)
#   SKIP_BUILD=1 ./install.sh reuse the existing build
#
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
KVER="$(uname -r)"
MODROOT="/lib/modules/${KVER}"
INSTALL_DIR="${MODROOT}/kernel/drivers/video"
SIGN_FILE="/usr/src/linux-headers-${KVER}/scripts/sign-file"

# Module signing key pair.  The default is the Ubuntu/DKMS pair already present
# on this box; enroll its certificate once (see the hint printed below) and both
# these modules and any DKMS rebuild become loadable.
MOK_KEY="${MOK_KEY:-/var/lib/shim-signed/mok/MOK.priv}"
MOK_CRT="${MOK_CRT:-/var/lib/shim-signed/mok/MOK.der}"

# Load order matters: nvidia first, then its clients.
MODULES=(nvidia nvidia-uvm nvidia-modeset nvidia-drm nvidia-peermem)

JOBS="${JOBS:-$(nproc)}"

say()  { printf '\n=== %s\n' "$*"; }
warn() { printf 'WARNING: %s\n' "$*" >&2; }
die()  { printf '\nERROR: %s\n' "$*" >&2; exit 1; }

# Never run the build as root.  It leaves root-owned artifacts in the checkout,
# and the ownership fix below would then chown the tree to root as well (that is
# exactly how this repository ended up root-owned on 2026-09-10).
if [ "$(id -u)" = 0 ]; then
    die "do not run this with sudo or as root.
     Run it as your normal user instead:  ./install.sh
     The script elevates only the individual steps that need root."
fi

# Prompt for sudo once, here, instead of in the middle of the install.
sudo -v || die "this script needs sudo to install, sign and load kernel modules"

secure_boot_on() {
    local out
    out="$(mokutil --sb-state 2>/dev/null || true)"
    case "$out" in *"SecureBoot enabled"*) return 0 ;; esac
    return 1
}

# NOTE ON STYLE, this bit us twice: never write `producer | grep -q pattern`
# here.  This script runs with `set -o pipefail`, and `grep -q` exits as soon as
# it matches, which can SIGPIPE the producer (mokutil --test-key exits 1 when the
# key IS enrolled, dkms status keeps writing warnings).  pipefail then reports
# the whole pipeline as failed even though the pattern matched -- a silent false
# negative.  Capture the output instead and match it in the shell.
#
# mokutil's exit status is not a reliable signal (--test-key exits 0 when the key
# is NOT enrolled and 1 when it IS), so match the message text and fall back to
# comparing our certificate's SHA1 against --list-enrolled.  Advisory only.
cert_enrolled() {
    [ -r "$MOK_CRT" ] || return 1
    local out fp
    out="$(mokutil --test-key "$MOK_CRT" 2>/dev/null || true)"
    if [[ "$out" == *"is already enrolled"* ]]; then
        return 0
    fi
    fp="$(openssl x509 -inform DER -in "$MOK_CRT" -noout -fingerprint -sha1 2>/dev/null || true)"
    fp="${fp##*=}"
    [ -n "$fp" ] || return 1
    out="$(mokutil --list-enrolled 2>/dev/null || true)"
    if [[ "${out,,}" == *"${fp,,}"* ]]; then
        return 0
    fi
    return 1
}

enroll_hint() {
    cat >&2 <<EOF
    sudo mokutil --import ${MOK_CRT}
    # choose a one-time password, then reboot
    sudo reboot
    # at the blue "Perform MOK management" screen:
    #   Enroll MOK -> Continue -> Yes -> enter the password -> Reboot
EOF
}

# --------------------------------------------------------------------------
say "1/8  build (kernel ${KVER})"
# --------------------------------------------------------------------------
[ -d "/lib/modules/${KVER}/build" ] \
    || die "no kernel headers for ${KVER}: install linux-headers-${KVER}"

# `sudo make modules` in the past leaves root-owned artifacts in the tree, which
# then break a non-root build with "Permission denied".
if [ -n "$(find "${REPO}" -path "${REPO}/.git" -prune -o -user root -print -quit 2>/dev/null)" ]; then
    warn "root-owned files from an earlier 'sudo make' found; fixing ownership"
    sudo chown -R "$(id -u):$(id -g)" "${REPO}"
fi

if [ "${SKIP_BUILD:-0}" = 1 ]; then
    say "SKIP_BUILD=1: reusing kernel-open/*.ko"
else
    make -C "${REPO}" modules -j "${JOBS}"
fi

for m in "${MODULES[@]}"; do
    [ -f "${REPO}/kernel-open/${m}.ko" ] \
        || die "${REPO}/kernel-open/${m}.ko was not built"
done

# --------------------------------------------------------------------------
say "2/8  Secure Boot / module signing preflight"
# --------------------------------------------------------------------------
if secure_boot_on && [ "${SKIP_SIGN:-0}" != 1 ]; then
    [ -x "${SIGN_FILE}" ] || die "missing ${SIGN_FILE} (install linux-headers-${KVER})"
    # Existence only: the private key is deliberately 0600 root-only, and
    # sign-file is invoked through sudo, so the invoking user must NOT be able to
    # read it.  Readability is checked as root instead.
    for f in "${MOK_CRT}" "${MOK_KEY}"; do
        [ -e "$f" ] || die "Secure Boot is on but ${f} does not exist.
     Create a key pair, enroll its certificate, or point MOK_KEY/MOK_CRT at one."
    done
    sudo test -r "${MOK_KEY}" || die "cannot read ${MOK_KEY} even as root"
    # Deliberately advisory: the mokutil answer is a hint, not a gate.  Signing
    # is harmless, and a false negative here must not cost a round trip -- a
    # wrongly blocked run is much worse than a wrongly attempted one.
    if cert_enrolled; then
        say "Secure Boot on, signing key enrolled -> modules will be signed"
    else
        warn "Secure Boot is on and ${MOK_CRT} does not report as enrolled."
        warn "Signing anyway; if the modules then refuse to load, enroll the key with:"
        enroll_hint
    fi
else
    warn "Secure Boot off or SKIP_SIGN=1: modules will be installed unsigned"
fi

# --------------------------------------------------------------------------
say "3/8  unload currently loaded modules"
# --------------------------------------------------------------------------
if [ -n "$(lsmod | grep -E '^nvidia' || true)" ]; then
    for m in nvidia-drm nvidia_modeset nvidia_uvm nvidia; do
        if [ -n "$(lsmod | grep -E "^${m} " || true)" ]; then
            if ! sudo rmmod "$m" 2>/dev/null; then
                die "$m is still in use (GUI session / CUDA process).
     Log out of the graphical session, or run 'sudo systemctl isolate multi-user.target',
     then re-run ./install.sh"
            fi
        fi
    done
    say "modules unloaded"
else
    say "nothing loaded"
fi

# --------------------------------------------------------------------------
say "4/8  drop competing nvidia module copies"
# --------------------------------------------------------------------------
# DKMS copies in updates/ shadow ${INSTALL_DIR} in depmod's search order.
# Parsed in the shell on purpose: `sed ... | head -1` under pipefail can SIGPIPE
# sed and abort the whole script via `set -e`.
dkms_out="$(dkms status 2>/dev/null || true)"
if [[ "$dkms_out" == nvidia/* ]]; then
    dkms_version="${dkms_out#nvidia/}"
    dkms_version="${dkms_version%%,*}"
    say "removing DKMS registration nvidia/${dkms_version} (it shadowed this build)"
    sudo dkms remove "nvidia/${dkms_version}" --all || warn "dkms remove reported an error, continuing"
fi

# Any leftover nvidia modules outside ${INSTALL_DIR} (updates/dkms, stale
# /kernel/drivers/video copies of a previous build, ...) win over our build.
while IFS= read -r stale; do
    say "removing stale ${stale}"
    sudo rm -f "$stale"
done < <(find "${MODROOT}" -name 'nvidia.ko*' -o -name 'nvidia-uvm.ko*' \
             -o -name 'nvidia-modeset.ko*' -o -name 'nvidia-drm.ko*' \
             -o -name 'nvidia-peermem.ko*' 2>/dev/null | grep -v "^${INSTALL_DIR}/" || true)

# --------------------------------------------------------------------------
say "5/8  install to ${INSTALL_DIR}"
# --------------------------------------------------------------------------
sudo make -C "${REPO}" modules_install
# kbuild install runs as root and can leave root-owned objects behind, which
# then break the next non-root build.
sudo chown -R "$(id -u):$(id -g)" "${REPO}"
ls -l "${INSTALL_DIR}"/nvidia*.ko

# --------------------------------------------------------------------------
say "6/8  sign installed modules"
# --------------------------------------------------------------------------
if secure_boot_on && [ "${SKIP_SIGN:-0}" != 1 ]; then
    for m in "${MODULES[@]}"; do
        [ -f "${INSTALL_DIR}/${m}.ko" ] || continue
        sudo "${SIGN_FILE}" sha256 "$MOK_KEY" "$MOK_CRT" "${INSTALL_DIR}/${m}.ko"
    done
else
    say "skipped"
fi

# --------------------------------------------------------------------------
say "7/8  depmod"
# --------------------------------------------------------------------------
sudo depmod -a "${KVER}"

for m in "${MODULES[@]}"; do
    [ -f "${INSTALL_DIR}/${m}.ko" ] || continue
    printf '%-16s signer: %s\n' "$m" "$(modinfo -F signer "${INSTALL_DIR}/${m}.ko" 2>/dev/null || echo '<unsigned>')"
done

# --------------------------------------------------------------------------
say "8/8  load and verify"
# --------------------------------------------------------------------------
# Fail loudly if depmod still resolves a module to a copy we did not build.
for m in nvidia nvidia_uvm nvidia_modeset nvidia_drm; do
    resolved="$(modprobe -n -v "$m" 2>/dev/null | sed -n 's/^insmod \([^ ]*\).*/\1/p' | tail -1)"
    [ -n "$resolved" ] || continue
    case "$resolved" in
        "${INSTALL_DIR}"/*) ;;
        *) die "modprobe would load ${resolved} instead of ${INSTALL_DIR}/${m}.ko" ;;
    esac
done

# Options such as NVreg_EnableP2P=1 come from /etc/modprobe.d/*.conf.
# Only these four are required; the failure gate below keys off them alone.
load_failed=0
for m in nvidia nvidia_uvm nvidia_modeset nvidia_drm; do
    if sudo modprobe "$m"; then
        printf 'loaded %s\n' "$m"
    else
        warn "modprobe ${m} failed"
        load_failed=1
    fi
done

if [ "$load_failed" = 1 ] || [ ! -d /sys/module/nvidia ]; then
    say "diagnosis"
    journalctl -k -n 60 --no-pager 2>/dev/null \
        | grep -iE 'nvidia|verification|unavailable key|key was rejected|lockdown' | tail -15 || true
    cat >&2 <<'EOF'

If the log shows "Loading of module with unavailable key is rejected" or
"Key was rejected by service", the signature is fine but the firmware does not
trust the signing key yet.  Check, and if needed enroll it:

    mokutil --test-key /var/lib/shim-signed/mok/MOK.der     # note: exit 1 means ENROLLED
EOF
    enroll_hint
    exit 1
fi

# Optional, and expected to fail here: nvidia-peermem needs the Mellanox
# peer-memory API (NV_MLNX_IB_PEER_MEM_SYMBOLS_PRESENT).  This kernel exports no
# ib_register_peer_memory_client, so conftest #undefs it and the module's init
# returns -EINVAL by design ("could not insert 'nvidia_peermem': Invalid
# argument").  It is needed only for GPUDirect RDMA through an InfiniBand/RoCE
# NIC -- never for the NVLink / BAR1 P2P between the two GA102 boards.
if sudo modprobe nvidia_peermem 2>/dev/null; then
    printf 'loaded %s\n' nvidia_peermem
else
    say "nvidia_peermem not loaded (expected on this kernel, not needed for NVLink P2P)"
fi

echo
lsmod | grep -E '^nvidia' || warn "no nvidia module is loaded"

if ! nvidia-smi; then
    cat >&2 <<'EOF'

nvidia-smi failed even though the kernel modules are loaded.  Check in order:

    cat /proc/driver/nvidia/version      # NVRM build behind the loaded modules
    ls -l /dev/nvidia*                   # device nodes for the GPUs and UVM
    nvidia-smi --query-gpu=name --format=csv   # NVML itself

A version mismatch between the kernel modules and the userspace NVML library
(/usr/lib/x86_64-linux-gnu/libnvidia-ml.so.1) also produces this: both must be
the same driver release.
EOF
    exit 1
fi
