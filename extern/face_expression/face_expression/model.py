import torch
from torch import nn


class BaselineFaceExpression(torch.nn.Module):
    def __init__(self, in_features, out_features, hid_features, n_layers=2):
        super().__init__()
        
        assert hid_features > 1
        
        backbone_layers = []
        for i in range(n_layers - 1):
            current_in_features = in_features if i == 0 else hid_features
            
            backbone_layers.extend([
                nn.Linear(current_in_features, hid_features),
                nn.ReLU(inplace=True)
            ])
            
        self.backbone = nn.Sequential(*backbone_layers)
            
        # final
        self.final = nn.Linear(hid_features, out_features)
        
        
    def forward(self, keypoints_2d, beta):
        bs = keypoints_2d.shape[0]
        
        x = torch.cat([keypoints_2d.view(bs, -1), beta], dim=1)
        x = self.backbone(x)
        x = self.final(x)

        expression = x[:, :10]
        jaw_pose = x[:, 10:14]
        
        return expression, jaw_pose


class SiameseModel(torch.nn.Module):
    def __init__(self,
                 *,
                 n_keypoints=468, beta_size=10,
                 emb_size=32, hid_size=32,
                 expression_size=10, jaw_pose_size=3,
                 use_beta=True,
                 use_keypoints_3d=False
        ):

        super().__init__()

        self.use_beta = use_beta
        self.use_keypoints_3d = use_keypoints_3d

        keypoint_input_size = 3 * n_keypoints if use_keypoints_3d else 2 * n_keypoints
        self.keypoint_backbone = nn.Sequential(
            nn.Linear(keypoint_input_size, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),

            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),

            nn.Linear(128, emb_size),            
        )

        if self.use_beta:
            self.beta_backbone = nn.Sequential(
                nn.Linear(beta_size, 32),
                nn.BatchNorm1d(32),
                nn.ReLU(inplace=True),

                nn.Linear(32, 64),
                nn.BatchNorm1d(64),
                nn.ReLU(inplace=True),

                nn.Linear(64, emb_size),          
            )
            
        self.mix_backbone = nn.Sequential(
            nn.Linear(2 * emb_size if use_beta else emb_size, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),

            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),

            nn.Linear(64, hid_size),
            nn.BatchNorm1d(hid_size),
            nn.ReLU(inplace=True)   
        )

        self.expression_head = nn.Linear(hid_size, expression_size)
        self.jaw_pose_head = nn.Linear(hid_size, jaw_pose_size)
        
        
    def forward(self, keypoints, beta):
        bs = keypoints.shape[0]

        keypoints_emb = self.keypoint_backbone(keypoints.view(bs, -1))
        if self.use_beta:
            beta_emb = self.beta_backbone(beta)
        
        if self.use_beta:
            emb = torch.cat([keypoints_emb, beta_emb], dim=1)
        else:
            emb = keypoints_emb

        feature = self.mix_backbone(emb)

        expression = self.expression_head(feature)
        jaw_pose = self.jaw_pose_head(feature)
        
        return expression, jaw_pose


class SiameseModelSmall(torch.nn.Module):
    def __init__(self,
                 *,
                 n_keypoints=468, beta_size=10,
                 emb_size=32, hid_size=32,
                 expression_size=10, jaw_pose_size=3
        ):

        super().__init__()

        self.keypoint_backbone = nn.Sequential(
            nn.Linear(2 * n_keypoints, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),

            nn.Linear(32, emb_size),      
        )

        self.beta_backbone = nn.Sequential(
            nn.Linear(beta_size, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),

            nn.Linear(32, emb_size),        
        )
            
        self.mix_backbone = nn.Sequential(
            nn.Linear(2 * emb_size, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
        )

        self.expression_head = nn.Linear(hid_size, expression_size)
        self.jaw_pose_head = nn.Linear(hid_size, jaw_pose_size)
        
        
    def forward(self, keypoints_2d, beta):
        bs = keypoints_2d.shape[0]

        keypoints_2d_emb = self.keypoint_backbone(keypoints_2d.view(bs, -1))
        beta_emb = self.beta_backbone(beta)
        
        emb = torch.cat([keypoints_2d_emb, beta_emb], dim=1)
        feature = self.mix_backbone(emb)

        expression = self.expression_head(feature)
        jaw_pose = self.jaw_pose_head(feature)
        
        return expression, jaw_pose



class SiameseModelDropout(torch.nn.Module):
    def __init__(self,
                 *,
                 n_keypoints=468, beta_size=10,
                 emb_size=32, hid_size=32,
                 expression_size=10, jaw_pose_size=3,
                 dropout_p=0.5
        ):

        super().__init__()

        self.keypoint_backbone = nn.Sequential(
            nn.Linear(2 * n_keypoints, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_p),

            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_p),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_p),

            nn.Linear(128, emb_size),            
        )

        self.beta_backbone = nn.Sequential(
            nn.Linear(beta_size, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),

            nn.Linear(32, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),

            nn.Linear(64, emb_size),          
        )
            
        self.mix_backbone = nn.Sequential(
            nn.Linear(2 * emb_size, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_p),

            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_p),

            nn.Linear(64, hid_size),
            nn.BatchNorm1d(hid_size),
            nn.ReLU(inplace=True)   
        )

        self.expression_head = nn.Linear(hid_size, expression_size)
        self.jaw_pose_head = nn.Linear(hid_size, jaw_pose_size)
        
    def forward(self, keypoints_2d, beta):
        bs = keypoints_2d.shape[0]

        keypoints_2d_emb = self.keypoint_backbone(keypoints_2d.view(bs, -1))
        beta_emb = self.beta_backbone(beta)
        
        emb = torch.cat([keypoints_2d_emb, beta_emb], dim=1)
        feature = self.mix_backbone(emb)

        expression = self.expression_head(feature)
        jaw_pose = self.jaw_pose_head(feature)
        
        return expression, jaw_pose