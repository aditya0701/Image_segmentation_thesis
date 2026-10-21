"""
Import-test every package the project needs and print its version.
A package built for a different numpy version fails here, not halfway through a stage.

Usage:
    D:\\my_envs\\sam-edge\\python.exe edge_deploy\\check_imports.py
"""
import importlib

MODULES = ["numpy", "torch", "torchvision", "vigra", "nifty", "elf", "torch_em", "micro_sam",
           "skimage", "scipy", "tifffile", "pandas", "psutil",
           "onnx", "onnxscript", "onnxruntime", "openvino", "nncf"]

failed = []
for name in MODULES:
    try:
        mod = importlib.import_module(name)
        print(f"OK    {name:<12} {getattr(mod, '__version__', '')}")
    except Exception as e:  # noqa: BLE001  (report any import failure)
        print(f"FAIL  {name:<12} {type(e).__name__}: {e}")
        failed.append(name)

# The one micro_sam call Stage 1 depends on.
try:
    from micro_sam.automatic_segmentation import get_predictor_and_segmenter  # noqa: F401
    print("OK    micro_sam.automatic_segmentation.get_predictor_and_segmenter")
except Exception as e:  # noqa: BLE001
    print(f"FAIL  get_predictor_and_segmenter: {e}")
    failed.append("get_predictor_and_segmenter")

print("\nALL OK" if not failed else f"\n{len(failed)} FAILED: {failed}")
