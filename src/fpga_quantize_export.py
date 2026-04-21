import argparse
import json
import math
import os
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch


@dataclass(frozen=True)
class QuantSpec:
    num_bits: int = 8
    signed: bool = True
    symmetric: bool = True
    frac_bits: Optional[int] = None  # if set, fixed scale = 2^-frac_bits

    @property
    def qmin(self) -> int:
        if self.signed:
            return -(1 << (self.num_bits - 1))
        return 0

    @property
    def qmax(self) -> int:
        if self.signed:
            return (1 << (self.num_bits - 1)) - 1
        return (1 << self.num_bits) - 1


def _as_numpy_fp32(t: torch.Tensor) -> np.ndarray:
    return t.detach().cpu().to(torch.float32).numpy()


def quantize_tensor(
    t: torch.Tensor,
    spec: QuantSpec,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Quantize a tensor to integers with either:
    - fixed-point scaling (frac_bits set): x_int = round(x * 2^frac_bits)
    - dynamic scaling (frac_bits None): scale chosen per-tensor

    Returns (q, meta) where:
    - q is np.ndarray of dtype int8/int16/int32
    - meta contains scale (float), zero_point (int), frac_bits (optional), clip info
    """
    x = _as_numpy_fp32(t)

    if spec.frac_bits is not None:
        scale = float(2 ** (-spec.frac_bits))
        x_scaled = x / scale  # == x * 2^frac_bits
        q = np.rint(x_scaled)
        q = np.clip(q, spec.qmin, spec.qmax)
        meta: Dict[str, Any] = {
            "scheme": "fixed_point",
            "num_bits": spec.num_bits,
            "signed": spec.signed,
            "symmetric": spec.symmetric,
            "frac_bits": spec.frac_bits,
            "scale": scale,
            "zero_point": 0,
        }
    else:
        # Symmetric per-tensor scale: scale = max_abs / qmax (avoid div-by-zero)
        max_abs = float(np.max(np.abs(x))) if x.size else 0.0
        if max_abs == 0.0 or not math.isfinite(max_abs):
            scale = 1.0
        else:
            scale = max_abs / float(spec.qmax if spec.signed else spec.qmax)
            if scale == 0.0 or not math.isfinite(scale):
                scale = 1.0

        q = np.rint(x / scale)
        q = np.clip(q, spec.qmin, spec.qmax)
        meta = {
            "scheme": "symmetric_per_tensor",
            "num_bits": spec.num_bits,
            "signed": spec.signed,
            "symmetric": spec.symmetric,
            "frac_bits": None,
            "scale": float(scale),
            "zero_point": 0,
            "max_abs": max_abs,
        }

    # Pick smallest int dtype that can hold qmin/qmax
    if spec.num_bits <= 8:
        q = q.astype(np.int8 if spec.signed else np.uint8)
    elif spec.num_bits <= 16:
        q = q.astype(np.int16 if spec.signed else np.uint16)
    else:
        q = q.astype(np.int32 if spec.signed else np.uint32)

    return q, meta


def quantize_state_dict(
    state_dict: Dict[str, torch.Tensor],
    spec: QuantSpec,
) -> Tuple[Dict[str, np.ndarray], Dict[str, Dict[str, Any]]]:
    q_tensors: Dict[str, np.ndarray] = {}
    meta: Dict[str, Dict[str, Any]] = {}

    for name, t in state_dict.items():
        if not isinstance(t, torch.Tensor):
            continue
        q, m = quantize_tensor(t, spec)
        q_tensors[name] = q
        meta[name] = m

    return q_tensors, meta


def _c_identifier(name: str) -> str:
    out = []
    for ch in name:
        if ch.isalnum():
            out.append(ch)
        else:
            out.append("_")
    ident = "".join(out)
    if ident and ident[0].isdigit():
        ident = "_" + ident
    return ident


def write_c_header(
    q_tensors: Dict[str, np.ndarray],
    meta: Dict[str, Dict[str, Any]],
    out_path: str,
    header_guard: str = "FPGA_WEIGHTS_H",
) -> None:
    lines = []
    lines.append(f"#ifndef {header_guard}")
    lines.append(f"#define {header_guard}")
    lines.append("")
    lines.append("#include <stdint.h>")
    lines.append("")
    lines.append("// Quantized weights/biases export.")
    lines.append("// Each tensor has an integer array plus a floating scale (or fixed-point frac_bits).")
    lines.append("")

    for name in sorted(q_tensors.keys()):
        arr = q_tensors[name]
        m = meta[name]
        ident = _c_identifier(name)
        shape = "x".join(str(s) for s in arr.shape) if arr.shape else "scalar"

        if arr.dtype == np.int8:
            ctype = "int8_t"
        elif arr.dtype == np.uint8:
            ctype = "uint8_t"
        elif arr.dtype == np.int16:
            ctype = "int16_t"
        elif arr.dtype == np.uint16:
            ctype = "uint16_t"
        elif arr.dtype == np.int32:
            ctype = "int32_t"
        elif arr.dtype == np.uint32:
            ctype = "uint32_t"
        else:
            raise ValueError(f"Unsupported dtype for {name}: {arr.dtype}")

        flat = arr.reshape(-1)
        lines.append(f"// {name} ({shape})")
        lines.append(f"static const {ctype} {ident}[{flat.size}] = {{")
        # Keep lines reasonably short
        row = []
        for i, v in enumerate(flat.tolist()):
            row.append(str(int(v)))
            if len(row) >= 24 or i == flat.size - 1:
                lines.append("  " + ", ".join(row) + ("," if i != flat.size - 1 else ""))
                row = []
        lines.append("};")
        lines.append(f"static const float {ident}__scale = {float(m['scale']):.12g}f;")
        if m.get("frac_bits") is not None:
            lines.append(f"static const int {ident}__frac_bits = {int(m['frac_bits'])};")
        lines.append("")

    lines.append(f"#endif  // {header_guard}")
    lines.append("")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def load_checkpoint_state_dict(checkpoint_path: str, device: str = "cpu") -> Dict[str, torch.Tensor]:
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        return ckpt["model_state_dict"]
    if isinstance(ckpt, dict):
        # Sometimes users save state_dict directly.
        if all(isinstance(v, torch.Tensor) for v in ckpt.values()):
            return ckpt  # type: ignore[return-value]
    raise ValueError(
        f"Unsupported checkpoint format at {checkpoint_path}. "
        "Expected dict with 'model_state_dict' or a raw state_dict."
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Quantize/export PyTorch weights for FPGA.")
    ap.add_argument("--checkpoint", required=True, help="Path to .pth/.pt checkpoint (state_dict or dict with model_state_dict).")
    ap.add_argument("--out_dir", default="fpga_export", help="Output directory for exported files.")
    ap.add_argument("--bits", type=int, default=8, choices=[8, 16, 32], help="Integer bit-width.")
    ap.add_argument("--unsigned", action="store_true", help="Use unsigned integers (default signed).")
    ap.add_argument("--fixed_frac_bits", type=int, default=None, help="If set, use fixed-point scale = 2^-fixed_frac_bits.")
    args = ap.parse_args()

    spec = QuantSpec(
        num_bits=int(args.bits),
        signed=not bool(args.unsigned),
        symmetric=True,
        frac_bits=args.fixed_frac_bits,
    )

    os.makedirs(args.out_dir, exist_ok=True)
    state_dict = load_checkpoint_state_dict(args.checkpoint, device="cpu")
    q_tensors, meta = quantize_state_dict(state_dict, spec)

    # Save as NPZ + JSON meta (nice for python + for generating verilog later)
    npz_path = os.path.join(args.out_dir, "weights_int.npz")
    np.savez_compressed(npz_path, **q_tensors)

    meta_path = os.path.join(args.out_dir, "weights_meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(
            {"quant_spec": spec.__dict__, "tensors": meta},
            f,
            indent=2,
            sort_keys=True,
        )

    header_path = os.path.join(args.out_dir, "weights.h")
    write_c_header(q_tensors, meta, header_path, header_guard="FPGA_EXPORTED_WEIGHTS_H")

    print(f"Wrote: {npz_path}")
    print(f"Wrote: {meta_path}")
    print(f"Wrote: {header_path}")


if __name__ == "__main__":
    main()

