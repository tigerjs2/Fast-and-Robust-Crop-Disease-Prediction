import torch
from torchvision import models
import torch.nn as nn


class FeatureExtractor(nn.Module):
    def __init__(self, backbone_name='mobilenet_v3_large', weights='DEFAULT', embed_dim=512):
        super(FeatureExtractor, self).__init__()
        self.backbone_name = backbone_name

        # 백본(feature map 추출부)과 출력 채널 수를 받아서 공통 임베딩 차원으로 투영
        self.features, in_channels = self._build_backbone(backbone_name, weights)
        self.projection = nn.Sequential(
            nn.Conv2d(in_channels, embed_dim, kernel_size=1),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True)
        )

    def _build_backbone(self, backbone_name, weights):
        # 추후 백본 추가 시 이 분기에 elif만 확장하면 됨
        if backbone_name == 'mobilenet_v3_large':
            model = models.mobilenet_v3_large(weights=weights)
            # torchvision 버전에 따라 마지막 feature 채널 수가 달라질 수 있으므로
            # 마지막 Conv2d의 출력 채널을 동적으로 읽는다.
            in_channels = None
            for module in reversed(model.features):
                if isinstance(module, nn.Conv2d):
                    in_channels = module.out_channels
                    break
                if hasattr(module, "out_channels"):
                    in_channels = module.out_channels
                    break
            if in_channels is None:
                raise RuntimeError("Failed to infer backbone output channels for mobilenet_v3_large")
            return model.features, in_channels

        raise ValueError(
            f"Unsupported backbone: {backbone_name}. "
            "Add new backbones in FeatureExtractor._build_backbone()."
        )

    def forward(self, x):
        # x: (batch, 3, H, W)
        x = self.features(x)
        # x: (batch, in_channels, h, w) -> (batch, embed_dim, h, w)
        x = self.projection(x)
        # attention 입력 형태에 맞춰 (batch, seq_len, embed_dim)으로 변환
        x = x.flatten(2).transpose(1, 2)
        return x


class MobileNetv3FeatureExtractor(FeatureExtractor):
    def __init__(self, weights='DEFAULT', embed_dim=512):
        super(MobileNetv3FeatureExtractor, self).__init__(
            backbone_name='mobilenet_v3_large',
            weights=weights,
            embed_dim=embed_dim
        )


class CrossAttentionModule(nn.Module):
    def __init__(self, embed_dim=512, num_heads=8, attn_dropout=0.0, residual_scale=1.0):
        super(CrossAttentionModule, self).__init__()
        # image token(query)과 text token(key/value) 간 cross-attention
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=attn_dropout,
            batch_first=True
        )
        # residual 결합 후 안정화를 위한 정규화
        self.norm = nn.LayerNorm(embed_dim)
        # attention 영향도를 조절하는 스케일 계수
        self.residual_scale = residual_scale

    def forward(self, image_features, text_queries, return_attention=False):
        # image_features: (batch, seq_len, embed_dim)
        # text_queries: (batch, num_queries, embed_dim) or (batch, embed_dim)
        if text_queries.dim() == 2:
            # 단일 텍스트 쿼리 입력 시 query 길이 1인 형태로 맞춤
            text_queries = text_queries.unsqueeze(1)

        attn_output, attn_weights = self.multihead_attn(
            query=image_features,
            key=text_queries,
            value=text_queries
        )

        # attention 결과를 원본 이미지 feature에 residual로 더해 강조 정보 반영
        enhanced_features = self.norm(image_features + self.residual_scale * attn_output)

        if return_attention:
            # 필요 시 attention 가중치를 함께 반환해 해석/디버깅에 활용
            return enhanced_features, attn_weights
        return enhanced_features