import argparse
import os
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from src.waveform_models import RNNConfig, WaveformRNNClassifier, build_mlp
from src.fpga_quantize_export import QuantSpec, quantize_state_dict


def load_waveform_csv(path: str) -> Tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(path)
    y = df["label"].to_numpy(dtype=np.int64)
    X = df.drop(columns=["label", "label_name"], errors="ignore").to_numpy(dtype=np.float32)
    return X, y


def make_short_sequences(X: np.ndarray, seq_len: int) -> np.ndarray:
    # Take the first seq_len points from each waveform row.
    if X.shape[1] < seq_len:
        raise ValueError(f"seq_len={seq_len} > available length {X.shape[1]}")
    return X[:, :seq_len]


def train_quick(
    model: nn.Module,
    X_train: np.ndarray,
    y_train: np.ndarray,
    epochs: int,
    batch_size: int,
    lr: float,
    device: str,
    is_rnn: bool,
) -> None:
    model.to(device)
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    n = X_train.shape[0]
    for _ in range(epochs):
        idx = np.random.permutation(n)
        for start in range(0, n, batch_size):
            j = idx[start : start + batch_size]
            xb = torch.tensor(X_train[j], dtype=torch.float32, device=device)
            yb = torch.tensor(y_train[j], dtype=torch.long, device=device)
            if is_rnn:
                xb = xb.unsqueeze(-1)  # (B,T,1)

            opt.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            opt.step()


@torch.no_grad()
def predict(model: nn.Module, X: np.ndarray, device: str, is_rnn: bool) -> np.ndarray:
    model.to(device)
    model.eval()
    xb = torch.tensor(X, dtype=torch.float32, device=device)
    if is_rnn:
        xb = xb.unsqueeze(-1)
    logits = model(xb)
    return torch.argmax(logits, dim=1).cpu().numpy()


def dequantize_state_dict(q: Dict[str, np.ndarray], meta: Dict[str, Dict], device: str) -> Dict[str, torch.Tensor]:
    out: Dict[str, torch.Tensor] = {}
    for name, arr in q.items():
        scale = float(meta[name]["scale"])
        out[name] = torch.tensor(arr.astype(np.float32) * scale, device=device)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Check trained network outcomes on short time-series samples (float vs quantized).")
    ap.add_argument("--model", choices=["mlp", "rnn"], default="rnn")
    ap.add_argument("--train_csv", default=os.path.join("datasets", "train_waveforms.csv"))
    ap.add_argument("--test_csv", default=os.path.join("datasets", "test_waveforms.csv"))
    ap.add_argument("--seq_len", type=int, default=32, help="Short sample length to test (e.g. 8/16/32).")
    ap.add_argument("--num_test", type=int, default=64, help="How many test rows to evaluate.")
    ap.add_argument("--epochs", type=int, default=3, help="Quick training epochs if no checkpoint is provided.")
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--checkpoint_out", default="trained_waveform_model.pth", help="Where to save quick-trained model.")
    ap.add_argument("--load_checkpoint", default=None, help="Optional: path to checkpoint to load instead of training.")
    ap.add_argument("--bits", type=int, default=8, choices=[8, 16], help="Quantization bit-width.")
    ap.add_argument("--fixed_frac_bits", type=int, default=None, help="If set, fixed-point quantization with frac bits.")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    Xtr, ytr = load_waveform_csv(args.train_csv)
    Xte, yte = load_waveform_csv(args.test_csv)

    Xtr_s = make_short_sequences(Xtr, args.seq_len)
    Xte_s = make_short_sequences(Xte, args.seq_len)

    # Keep test small (low time-series samples + few rows)
    Xte_s = Xte_s[: args.num_test]
    yte = yte[: args.num_test]

    num_classes = int(max(ytr.max(), yte.max()) + 1)

    if args.model == "mlp":
        model = build_mlp(input_size=args.seq_len, num_classes=num_classes)
        is_rnn = False
        extra: Dict = {"arch": "mlp", "seq_len": args.seq_len, "num_classes": num_classes}
    else:
        cfg = RNNConfig(input_size=1, hidden_size=64, num_layers=1, num_classes=num_classes)
        model = WaveformRNNClassifier(cfg)
        is_rnn = True
        extra = {"arch": "rnn", **cfg.__dict__, "seq_len": args.seq_len}

    if args.load_checkpoint is None:
        train_quick(
            model=model,
            X_train=Xtr_s,
            y_train=ytr,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            device=device,
            is_rnn=is_rnn,
        )
        torch.save({"model_state_dict": model.state_dict(), "meta": extra}, args.checkpoint_out)
        ckpt_path = args.checkpoint_out
    else:
        ckpt_path = args.load_checkpoint
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])

    # Float outcomes
    yhat_float = predict(model, Xte_s, device=device, is_rnn=is_rnn)
    acc_float = float((yhat_float == yte).mean())

    # Quantize weights/biases to ints (FPGA-friendly)
    spec = QuantSpec(num_bits=args.bits, signed=True, symmetric=True, frac_bits=args.fixed_frac_bits)
    q, qmeta = quantize_state_dict(model.state_dict(), spec)

    # Simulate "FPGA integer weights" by dequantizing them back to float and running again.
    model_q = type(model)(model.cfg) if args.model == "rnn" else build_mlp(input_size=args.seq_len, num_classes=num_classes)
    model_q.load_state_dict(dequantize_state_dict(q, qmeta, device=device), strict=False)

    yhat_q = predict(model_q, Xte_s, device=device, is_rnn=is_rnn)
    acc_q = float((yhat_q == yte).mean())

    # Print a few sample outcomes
    print(f"Device: {device}")
    print(f"Checkpoint used: {ckpt_path}")
    print(f"Short seq_len: {args.seq_len}, num_test: {args.num_test}")
    print(f"Accuracy float: {acc_float:.4f}")
    print(f"Accuracy quant(sim): {acc_q:.4f}  (bits={args.bits}, fixed_frac_bits={args.fixed_frac_bits})")
    print("")
    print("First 10 predictions (label | float | quant):")
    for i in range(min(10, len(yte))):
        print(f"{int(yte[i])} | {int(yhat_float[i])} | {int(yhat_q[i])}")


if __name__ == "__main__":
    main()

