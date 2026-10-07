"""
One-time conversion: best.pt (full training checkpoint, pickle) -> best_weights_only.pt.

best.pt also stores training objects that reference code in the training
script (e.g. the augmentation class ChannelSafeIntensityAug). Unpickling needs
that code to exist, which breaks loading from any other script. This keeps only the two
things inference needs, as plain tensors:
    model_state    SAM weights (image encoder, prompt encoder, mask decoder)
    decoder_state  UNETR decoder for automatic instance segmentation (AIS)

Checks: (1) the new file loads with weights_only=True and matches best.pt exactly;
(2) its image encoder matches best.safetensors exactly.
The original files are not modified.

Usage:
    D:/my_envs/sam-edge/python.exe edge_deploy/extract_weights.py
"""
import pickle
import types
from collections import OrderedDict

import numpy as np
import torch
from safetensors.torch import load_file

import common

SRC = common.CHECKPOINT_FULL
DST = common.CHECKPOINT
SAFETENSORS = SRC.with_name("best.safetensors")


class PlaceholderUnpickler(pickle.Unpickler):
    """best.pt references objects defined in the training script's __main__
    (ChannelSafeIntensityAug, clip_to_one, scale_to_255, ...). They only describe
    the training data pipeline, so an empty placeholder is substituted for any
    object that cannot be found. The weights are plain tensors and unaffected."""

    def find_class(self, module, name):
        try:
            return super().find_class(module, name)
        except (AttributeError, ModuleNotFoundError):
            print(f"  placeholder for missing {module}.{name}")
            return type(name, (), {})


placeholder_pickle = types.ModuleType("placeholder_pickle")
placeholder_pickle.__dict__.update(pickle.__dict__)
placeholder_pickle.Unpickler = PlaceholderUnpickler


def plain(state):
    return OrderedDict((k, v.detach().clone()) for k, v in state.items())


def same(a, b):
    return a.keys() == b.keys() and all(torch.equal(a[k], b[k]) for k in a)


def main():
    full = torch.load(SRC, map_location="cpu", weights_only=False, pickle_module=placeholder_pickle)
    print("Keys in best.pt:", list(full.keys()))
    slim = {"model_state": plain(full["model_state"]), "decoder_state": plain(full["decoder_state"])}
    torch.save(slim, DST)

    # Check 1: reload in safe mode and compare with the original, tensor by tensor.
    reloaded = torch.load(DST, map_location="cpu", weights_only=True)
    for key in slim:
        ok = same(full[key], reloaded[key])
        print(f"{key}: {len(reloaded[key])} tensors, identical to best.pt: {ok}")
        assert ok

    # Check 2: the image encoder equals the one in best.safetensors.
    st = load_file(SAFETENSORS)
    enc_pt = {k: v for k, v in reloaded["model_state"].items() if k.startswith("sam.image_encoder.")}
    enc_st = {k: v for k, v in st.items() if k.startswith("sam.image_encoder.")}
    max_diff = max(float((enc_pt[k] - enc_st[k]).abs().max()) for k in enc_st) if enc_pt.keys() == enc_st.keys() else np.inf
    ok = same(enc_pt, enc_st)
    print(f"Image encoder: {len(enc_pt)} tensors, identical to best.safetensors: {ok} (max abs diff {max_diff})")
    assert ok

    print(f"Sizes: best.pt {SRC.stat().st_size / 1e6:.0f} MB -> {DST.name} {DST.stat().st_size / 1e6:.0f} MB")


if __name__ == "__main__":
    main()
