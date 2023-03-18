import numpy as np
from torchvision.models import vgg16, VGG16_Weights
import torch
import torch.nn as nn

PI = 3.1415926535


class Model(nn.Module):
    def __init__(self, bbox2d_dim=None, bbox3d_dim=None, pool_size=7, dropout=0.2):
        super().__init__()
        self.pool_size = pool_size
        self.backbone = vgg16(weights=VGG16_Weights.IMAGENET1K_V1).features
        self.pool = nn.AdaptiveAvgPool2d(output_size=self.pool_size)
        self.bbox2d_processor = nn.Sequential(
            nn.Identity() if bbox2d_dim is None else DimensionEmbedding(bbox2d_dim, freq=1e4),
            nn.Linear(2 if bbox2d_dim is None else 4 * bbox2d_dim, 512), nn.ReLU(),
            nn.Linear(512, 2048), nn.ReLU(),
        )
        self.bbox3d_estimator = nn.Sequential(
            nn.Linear(self.pool_size * self.pool_size * 512, 512), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(512, 128), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128, 3), nn.ReLU(),
        )
        self.bbox3d_processor = nn.Sequential(
            nn.Identity() if bbox3d_dim is None else DimensionEmbedding(bbox3d_dim, freq=1e4),
            nn.Linear(3 if bbox3d_dim is None else 6 * bbox3d_dim, 512), nn.ReLU(),
            nn.Linear(512, 2048), nn.ReLU(),
        )
        self.orientation_estimator = nn.Sequential(
            nn.Linear(self.pool_size * self.pool_size * 512 + 2048 * 2, 512), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(512, 1), nn.Tanh(),
        )
        self.criterion = nn.MSELoss()

    def forward(self, x, bbox_2d):
        feats = self.backbone(x)
        flattened_feats = self.pool(feats).contiguous().view(-1, self.pool_size * self.pool_size * 512)
        embedding_2d = self.bbox2d_processor(bbox_2d)
        pred_3d = self.bbox3d_estimator(flattened_feats)
        embedding_3d = self.bbox3d_processor(pred_3d)
        embedding_all = torch.cat([flattened_feats, embedding_2d, embedding_3d], dim=-1)
        pred_orientation = self.orientation_estimator(embedding_all) * PI
        return pred_3d, pred_orientation

    def loss(self, x, bbox_2d, bbox_3d, orientation, weight_3d):
        pred_3d, pred_orientation = self.forward(x, bbox_2d)
        loss_3d = self.criterion(pred_3d, bbox_3d)
        loss_sin = self.criterion(pred_orientation.sin(), orientation.sin())
        loss_cos = self.criterion(pred_orientation.cos(), orientation.cos())
        return loss_sin + loss_cos + weight_3d * loss_3d, \
            {'sin': loss_sin.item(), 'cos': loss_cos.item(), '3d': loss_3d.item()}

    def aos(self, x, bbox_2d, orientation):
        _, pred_orientation = self.forward(x, bbox_2d)
        aos = (1 + (orientation - pred_orientation).cos()) / 2
        return aos.squeeze().sum().item()


# similar high-freq embedding strategy to the timestep embedding strategy of diffusion models
class DimensionEmbedding(nn.Module):
    def __init__(self, dim, freq):
        super().__init__()
        self.dim = dim
        self.freq = freq

    def forward(self, x):
        emb = np.log(self.freq) / (self.dim - 1)
        emb = torch.exp(torch.arange(self.dim) * -emb).to(x.device)
        emb = x.unsqueeze(-1) * emb.unsqueeze(0).unsqueeze(0)
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        emb = emb.contiguous().view(emb.size(0), -1)
        return emb

