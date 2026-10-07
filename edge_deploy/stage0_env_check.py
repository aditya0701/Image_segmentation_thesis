"""
Stage 0: environment check.

Records OS, CPU, Python, GPU/NPU drivers and the versions of the packages used
in this project. Uses only the standard library, so it runs before any
environment is set up. Run it on every machine and keep the JSON next to the
results, so every benchmark number can be tied to exact hardware and software.

Usage:
    python edge_deploy/stage0_env_check.py
Writes:
    edge_deploy/results/env_<hostname>.json
"""
import importlib.metadata as md
import json
import platform
import shutil
import socket
import subprocess
import sys
from pathlib import Path

PACKAGES = [
    "torch", "torchvision", "micro_sam", "segment-anything", "torch_em",
    "numpy", "tifffile", "scipy", "scikit-image", "pandas",
    "onnx", "onnxruntime", "onnxruntime-gpu", "onnxscript",
    "tensorrt", "pycuda", "openvino", "nncf",
]


def run(cmd):
    """Run a shell command and return its stdout, or the error text."""
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=60, shell=isinstance(cmd, str))
        return (out.stdout or out.stderr).strip()
    except Exception as e:  # noqa: BLE001  (we want to record any failure)
        return f"ERROR: {e}"


def windows_devices():
    """GPU and NPU names plus driver versions via PowerShell CIM queries."""
    ps = (
        "Get-CimInstance Win32_PnPSignedDriver | "
        "Where-Object { $_.DeviceClass -in 'DISPLAY','COMPUTEACCELERATOR' } | "
        "Select-Object DeviceName,DriverVersion,DriverDate | ConvertTo-Json"
    )
    return run(["powershell", "-NoProfile", "-Command", ps])


def linux_devices():
    return run("lspci | grep -i -E 'vga|3d|display|processing accelerator'")


def package_versions():
    versions = {}
    for name in PACKAGES:
        try:
            versions[name] = md.version(name)
        except md.PackageNotFoundError:
            versions[name] = None
    return versions


def torch_info():
    try:
        import torch
    except ImportError:
        return None
    info = {"version": torch.__version__, "cuda_built": torch.version.cuda,
            "cuda_available": torch.cuda.is_available()}
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        info.update(gpu=p.name, compute_capability=f"{p.major}.{p.minor}",
                    vram_gb=round(p.total_memory / 1024**3, 2),
                    cudnn=torch.backends.cudnn.version())
    return info


def openvino_info():
    try:
        import openvino as ov
    except ImportError:
        return None
    core = ov.Core()
    return {d: core.get_property(d, "FULL_DEVICE_NAME") for d in core.available_devices}


def main():
    system = platform.system()
    report = {
        "hostname": socket.gethostname(),
        "os": f"{system} {platform.release()} ({platform.version()})",
        "machine": platform.machine(),
        "cpu": platform.processor(),
        "python": sys.version.split()[0],
        "python_executable": sys.executable,
        "devices": windows_devices() if system == "Windows" else linux_devices(),
        "nvidia_smi": run(["nvidia-smi"]) if shutil.which("nvidia-smi") else "nvidia-smi not found on PATH",
        "nvcc": run(["nvcc", "--version"]) if shutil.which("nvcc") else "nvcc not found on PATH",
        "packages": package_versions(),
        "torch": torch_info(),
        "openvino_devices": openvino_info(),
    }
    out_dir = Path(__file__).parent / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"env_{report['hostname']}.json"
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
