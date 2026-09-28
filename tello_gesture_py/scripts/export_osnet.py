"""Convert the OSNet person re-identification weights to ONNX, once.

The flight stack runs OSNet through ONNX Runtime, like the face model, so it
never needs PyTorch. This script is the one place PyTorch is used: run it in a
separate environment that has torch installed, not in the project's .venv.

    pip install torch --index-url https://download.pytorch.org/whl/cpu
    pip install onnx onnxruntime
    python tello_gesture_py/scripts/export_osnet.py --weights osnet_x0_25_msmt17.pth

Weights: K. Zhou's release (MIT), https://huggingface.co/kaiyangzhou/osnet,
file osnet_x0_25_msmt17_combineall_256x128_amsgrad_ep150_stp60_lr0.0015_b64_fb10_
softmax_labelsmooth_flip_jitter.pth (x0_25; the x1_0 file is named alike). They were trained on MSMT17, whose terms are
the dataset's own (research use), as with the face model.

The architecture below reproduces OSNet from K. Zhou, Y. Yang, A. Cavallaro and
T. Xiang, "Omni-Scale Feature Learning for Person Re-Identification", ICCV 2019,
following deep-person-reid (torchreid, MIT). Every weight must load, apart from
the training classifier, or the export stops.

Output: models/third_party/osnet_<variant>_msmt17.onnx, input 1x3x256x128 (RGB,
ImageNet-normalised), output a 512-d embedding.
"""
import argparse
import os

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError:
    raise SystemExit("export_osnet needs PyTorch, which the project's .venv does not have "
                     "by design; run it in a separate environment (see the top of this file)")


class ConvLayer(nn.Module):
    def __init__(self, cin, cout, k, stride=1, padding=0):
        super().__init__()
        self.conv = nn.Conv2d(cin, cout, k, stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(cout)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class Conv1x1(nn.Module):
    def __init__(self, cin, cout, stride=1):
        super().__init__()
        self.conv = nn.Conv2d(cin, cout, 1, stride=stride, padding=0, bias=False)
        self.bn = nn.BatchNorm2d(cout)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class Conv1x1Linear(nn.Module):
    def __init__(self, cin, cout, stride=1):
        super().__init__()
        self.conv = nn.Conv2d(cin, cout, 1, stride=stride, padding=0, bias=False)
        self.bn = nn.BatchNorm2d(cout)

    def forward(self, x):
        return self.bn(self.conv(x))


class LightConv3x3(nn.Module):
    """1x1 then depthwise 3x3: the lite convolution OSNet is built from."""

    def __init__(self, cin, cout):
        super().__init__()
        self.conv1 = nn.Conv2d(cin, cout, 1, stride=1, padding=0, bias=False)
        self.conv2 = nn.Conv2d(cout, cout, 3, stride=1, padding=1, bias=False, groups=cout)
        self.bn = nn.BatchNorm2d(cout)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv2(self.conv1(x))))


class ChannelGate(nn.Module):
    """The unified aggregation gate: per-channel weights for each scale stream."""

    def __init__(self, c, reduction=16):
        super().__init__()
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(c, c // reduction, 1, bias=True, padding=0)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(c // reduction, c, 1, bias=True, padding=0)
        self.gate_activation = nn.Sigmoid()

    def forward(self, x):
        g = self.gate_activation(self.fc2(self.relu(self.fc1(self.global_avgpool(x)))))
        return x * g


class OSBlock(nn.Module):
    """Four streams of 1-4 lite convolutions (four receptive-field sizes), gated and summed."""

    def __init__(self, cin, cout, bottleneck_reduction=4):
        super().__init__()
        mid = cout // bottleneck_reduction
        self.conv1 = Conv1x1(cin, mid)
        self.conv2a = LightConv3x3(mid, mid)
        self.conv2b = nn.Sequential(LightConv3x3(mid, mid), LightConv3x3(mid, mid))
        self.conv2c = nn.Sequential(*[LightConv3x3(mid, mid) for _ in range(3)])
        self.conv2d = nn.Sequential(*[LightConv3x3(mid, mid) for _ in range(4)])
        self.gate = ChannelGate(mid)
        self.conv3 = Conv1x1Linear(mid, cout)
        self.downsample = Conv1x1Linear(cin, cout) if cin != cout else None

    def forward(self, x):
        identity = x
        x1 = self.conv1(x)
        x2 = (self.gate(self.conv2a(x1)) + self.gate(self.conv2b(x1))
              + self.gate(self.conv2c(x1)) + self.gate(self.conv2d(x1)))
        x3 = self.conv3(x2)
        if self.downsample is not None:
            identity = self.downsample(identity)
        return F.relu(x3 + identity)


class OSNet(nn.Module):
    def __init__(self, layers=(2, 2, 2), channels=(64, 256, 384, 512), feature_dim=512):
        super().__init__()
        self.conv1 = ConvLayer(3, channels[0], 7, stride=2, padding=3)
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)
        self.conv2 = self._make_layer(layers[0], channels[0], channels[1], True)
        self.conv3 = self._make_layer(layers[1], channels[1], channels[2], True)
        self.conv4 = self._make_layer(layers[2], channels[2], channels[3], False)
        self.conv5 = Conv1x1(channels[3], channels[3])
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(nn.Linear(channels[3], feature_dim),
                                nn.BatchNorm1d(feature_dim), nn.ReLU(inplace=True))

    @staticmethod
    def _make_layer(n, cin, cout, reduce_spatial):
        blocks = [OSBlock(cin, cout)] + [OSBlock(cout, cout) for _ in range(1, n)]
        if reduce_spatial:
            blocks.append(nn.Sequential(Conv1x1(cout, cout), nn.AvgPool2d(2, stride=2)))
        return nn.Sequential(*blocks)

    def forward(self, x):
        x = self.maxpool(self.conv1(x))
        x = self.conv5(self.conv4(self.conv3(self.conv2(x))))
        v = self.global_avgpool(x).flatten(1)
        return self.fc(v)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--weights", required=True)
    ap.add_argument("--variant", choices=["x1_0", "x0_25"], default="x0_25",
                    help="x0_25 (default): 0.2M parameters, ~6 ms per crop on a laptop CPU; x1_0: 2.2M, ~160 ms")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    a.out = a.out or f"models/third_party/osnet_{a.variant}_msmt17.onnx"
    channels = {"x1_0": (64, 256, 384, 512), "x0_25": (16, 64, 96, 128)}[a.variant]

    ckpt = torch.load(a.weights, map_location="cpu", weights_only=False)
    sd = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    sd = {k[7:] if k.startswith("module.") else k: v for k, v in sd.items()}
    sd = {k: v for k, v in sd.items() if not k.startswith("classifier.")}

    model = OSNet(channels=channels).eval()
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing or unexpected:
        raise SystemExit(f"architecture does not match the weights:\n  missing {missing}\n"
                         f"  unexpected {unexpected}")
    print(f"loaded {len(sd)} tensors, all matched")

    x = torch.randn(1, 3, 256, 128)
    with torch.no_grad():
        ref = model(x).numpy()
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    torch.onnx.export(model, x, a.out, input_names=["input"], output_names=["embedding"],
                      dynamic_axes={"input": {0: "batch"}, "embedding": {0: "batch"}},
                      opset_version=17, dynamo=False)

    import onnxruntime as ort
    got = ort.InferenceSession(a.out, providers=["CPUExecutionProvider"]).run(
        None, {"input": x.numpy()})[0]
    err = float(np.abs(got - ref).max())
    print(f"wrote {a.out}  ({os.path.getsize(a.out) / 1e6:.1f} MB), "
          f"max |onnx - torch| = {err:.2e}")
    if err > 1e-3:
        raise SystemExit("ONNX output disagrees with PyTorch")


if __name__ == "__main__":
    main()
