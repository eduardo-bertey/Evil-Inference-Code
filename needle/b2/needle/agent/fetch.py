import os
import platform
import sys
import zipfile

HF_REPO = "Cactus-Compute/needle2"

# Names the engine wheel fetched from HF_REPO. Decoupled from the package
# __version__ (which the release workflow bumps every week): bump this only
# when new engine wheels are actually uploaded to HF_REPO.
ENGINE_VERSION = "2.0.0"


def _lib_name():
    if sys.platform == "darwin":
        return "libneedle.dylib"
    if sys.platform == "win32":
        return "libneedle.dll"
    return "libneedle.so"


def _is_musl():
    if sys.platform != "linux":
        return False
    libc, _ = platform.libc_ver()
    if libc:
        return False
    try:
        with open("/proc/self/maps", "rb") as maps:
            return b"musl" in maps.read()
    except OSError:
        return True


def _platform_tag():
    machine = platform.machine().lower()
    if sys.platform == "darwin":
        arch = "arm64" if machine in ("arm64", "aarch64") else "x86_64"
        return "macosx_11_0_" + arch
    if sys.platform == "win32":
        return "win_arm64" if machine in ("arm64", "aarch64") else "win_amd64"
    arch = "aarch64" if machine in ("aarch64", "arm64") else "x86_64"
    family = "musllinux_1_2_" if _is_musl() else "manylinux2014_"
    return family + arch


def fetch_library(version, dest_dir):
    from huggingface_hub import hf_hub_download

    tag = _platform_tag()
    wheel = "cactus_needle-{}-py3-none-{}.whl".format(version, tag)
    path = hf_hub_download(repo_id=HF_REPO, filename="python/" + wheel, repo_type="model")
    lib = _lib_name()
    with zipfile.ZipFile(path) as archive:
        data = archive.read("needle/" + lib)
    out = os.path.join(dest_dir, lib)
    with open(out, "wb") as handle:
        handle.write(data)
    return out
