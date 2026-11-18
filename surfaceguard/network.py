from __future__ import annotations

import torch
from torch import nn


class Block(nn.Module):
    def __init__(self, cin: int, cout: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.layers(x)


class TinyUNet(nn.Module):
    def __init__(self, width: int = 12):
        super().__init__()
        self.a = Block(1, width)
        self.b = Block(width, width * 2)
        self.c = Block(width * 2, width * 4)
        self.pool = nn.MaxPool2d(2)
        self.up_b = Block(width * 6, width * 2)
        self.up_a = Block(width * 3, width)
        self.head = nn.Conv2d(width, 4, 1)

    def forward(self, x):
        import torch.nn.functional as F
        a = self.a(x)
        b = self.b(self.pool(a))
        c = self.c(self.pool(b))
        x = self.up_b(torch.cat([F.interpolate(c, size=b.shape[-2:], mode="bilinear", align_corners=False), b], dim=1))
        x = self.up_a(torch.cat([F.interpolate(x, size=a.shape[-2:], mode="bilinear", align_corners=False), a], dim=1))
        return self.head(x)


def load_checkpoint(path, device="cpu"):
    # weights_only avoids loading arbitrary Python objects from external checkpoints.
    artifact = torch.load(path, map_location=device, weights_only=True)
    if artifact.get("class_names") != ["background", "label_1", "label_2", "label_3"]:
        raise ValueError("Checkpoint class mapping does not match this application")
    model = TinyUNet(int(artifact["width"])).to(device)
    model.load_state_dict(artifact["weights"])
    model.eval()
    return model, artifact


@torch.inference_mode()
def predict(model, gray, device="cpu"):
    x = torch.from_numpy(gray.astype("float32") / 255).unsqueeze(0).unsqueeze(0).to(device)
    return torch.softmax(model(x)[0], dim=0).cpu().numpy()
