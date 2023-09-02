import torch
import torch.nn as nn
from torch import Tensor
import numpy as np
import timm
    
class FeatureExtractor(nn.Module):
    def __init__(self, model_name, dropout=0.5, dim_spatial=None, pooling=(3, 3), pretrained=False, device="cuda"):
        super().__init__()
        
        self.d_model = {
            'resnet18': 512,
            'resnet18_skip_last': 256, 
            'resnet18_skip3&4': 128,
            'resnet34': 512,
            'resnet34_skip_last': 256,
            'resnet34_skip3&4': 128,
            'resnet50': 2048,
            'resnet50_skip_last': 1024,
            'resnet101': 2048,
            'resnext50_32x4d': 2048,
            'resnext101_32x4d': 2048,
            'resnext101_32x8d': 2048,
            'seresnet50': 2048,
            'seresnet18': 512, 
            'seresnet18_skip_last': 256, 
            'seresnet18_skip3&4': 128,
            'seresnet34': 512,
            'seresnet34_skip_last': 256, 
            'efficientnet_b3': 1536,
            'efficientnet_b4': 1792,
            'vgg16': 4096 
        }
        assert model_name in self.d_model
        
        self.model_name = model_name if 'skip' not in model_name else model_name.split('_')[0]
        self.fe =  timm.create_model(self.model_name, pretrained=pretrained).to(device)
        self.dropout = dropout
        
        if 'resne' in model_name:
            self.fe.fc = nn.Identity()
        elif 'efficientnet' in model_name:
            self.fe.classifier = nn.Identity()
        elif 'vgg' in model_name:
            self.fe.head = nn.Identity()
        
        if 'skip_last' in model_name:
            self.fe.layer4 = nn.Identity()
        elif 'skip3&4' in model_name:
            self.fe.layer3 = nn.Identity()
            self.fe.layer4 = nn.Identity()
            
        self.num_filter = self.d_model[model_name]
        
        # set resolution and pooling
        assert type(pooling) in {tuple, list}
        self.pooling = pooling
        self.resolution = self.pooling[0]*self.pooling[1]
        self.fe.global_pool = nn.Identity()
        
        # bottleneck layer
        if dim_spatial is not None:
            assert isinstance(dim_spatial, int) and dim_spatial > 0
            
            self.bottleneck = nn.Sequential(
                nn.Dropout(p=dropout),
                nn.Linear(self.num_filter*self.resolution, dim_spatial),
                nn.BatchNorm1d(dim_spatial),
                nn.ReLU()
            )
            self.dim_spatial = dim_spatial
            
        else:
            self.bottleneck = nn.Identity()
            self.dim_spatial = self.num_filter*self.resolution
    
    def forward(self, x):
        features = self.fe(x)
        features = nn.functional.adaptive_avg_pool2d(features, self.pooling)
        features = features.view(len(x), -1)
        
        return self.bottleneck(features)

class TemporalModel(nn.Module):
    def __init__(self, feature_extractor, input_as_window):
        super().__init__()
        
        self.feature_extractor = feature_extractor
        self.input_as_window = input_as_window
        
    def forward(self, x):
        
        features = self.feature_extractor(x.view(-1, x.shape[-3], x.shape[-2], x.shape[-1]))
        
        return features.view(*x.shape[0:2], -1)
    
    def expand_channel(self, x):
        return x.unfold(1, 3, 1).permute(0, 1, 4, 2, 3).contiguous()
    
    def infer(self, x, with_logit=False, standardise=False):
        
        if standardise:
            x = (x - 60.761)/82.9350
        
        if self.input_as_window:
            if x.dim() == 5:
                assert x.shape[0] == 1
            elif x.dim() == 4:
                x = x.unsqueeze(0)
            
        else:
            if x.dim() == 4:
                assert x.shape[0] == 1
            elif x.dim() == 3:
                x = x.unsqueeze(0)
        
        self.eval()
        logits = self.forward(x)
        
        if logits[0] < 0:
            for n, lg in enumerate(logits):
                if lg > 0:
                    break
            pred = torch.argmin(logits[:n+1]**2)
        else:
            pred = torch.tensor(0)
        
        if not self.input_as_window:
            pred += 1
        
        return (pred, logits) if with_logit else pred

class InversionTimeLSTM(TemporalModel):
    def __init__(self, feature_extractor, num_layers=2, hidden_size=512, dropout=0.2, input_as_window=False):
        super().__init__(feature_extractor, input_as_window)
        
        self.input_dropout = nn.Dropout(p=dropout)
        self.temporal_block = nn.LSTM(
            input_size=self.feature_extractor.dim_spatial, hidden_size=hidden_size, num_layers=num_layers, 
            dropout=dropout, bidirectional=True, batch_first=True
        )
        
        self.decoder = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(hidden_size*2, 1)
        )
    
    def forward(self, x):
        if not self.input_as_window:
            x = self.expand_channel(x)
        
        lstm_in = super().forward(x)
        temporal_features = self.temporal_block(self.input_dropout(lstm_in))[0]
                
        logits = self.decoder(temporal_features).squeeze()
            
        return logits

