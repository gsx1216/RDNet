from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from .smt import smt_t



class ConvBR(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride=1, padding=1, dilation=1):
        super(ConvBR, self).__init__()
        self.conv = nn.Conv2d(in_channel, out_channel,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU(inplace=True)
        self.init_weight()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)


def conv3x3_bn_relu(in_planes, out_planes, k=3, s=1, p=1, b=False):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=k, stride=s, padding=p, bias=b),
        nn.BatchNorm2d(out_planes),
        # nn.ReLU(inplace=True),
        nn.GELU(),
    )

class ImprovedTimeMixer(nn.Module):
    def __init__(self, num_scales=4, channels=[512, 256, 128, 64]):
        super(ImprovedTimeMixer, self).__init__()
        self.num_scales = num_scales
        self.prev_feats = [None] * num_scales
        self.is_testing = False

        self.refine = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(ch, ch, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(ch),
                nn.ReLU(inplace=True)
            ) for ch in channels
        ])

        self.controller = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(ch * 2, ch, kernel_size=1, bias=False),
                nn.ReLU(inplace=True),
                nn.Conv2d(ch, ch, kernel_size=1),
                nn.Sigmoid()
            ) for ch in channels
        ])

    def forward(self, current_feats):
        fused_feats = []
        for i, curr in enumerate(current_feats):
            prev = self.prev_feats[i]

            if self.is_testing:
                fusion = self.refine[i](curr)
                fused_feats.append(fusion)

            else:
                if prev is None:
                    self.prev_feats[i] = curr.detach()
                    fused_feats.append(curr)
                else:
                    weight = self.controller[i](torch.cat([curr, prev], dim=1))
                    fusion = weight * curr + (1.0 - weight) * prev
                    fusion = self.refine[i](fusion)
                    fused_feats.append(fusion)
                    self.prev_feats[i] = fusion.detach()

        return fused_feats

    def set_testing(self, testing=False):
        self.is_testing = testing

class CA_Block(nn.Module):
    def __init__(self, in_dim):
        super(CA_Block, self).__init__()
        self.gamma = nn.Parameter(torch.ones(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        B, C, H, W = x.size()
        q_x = x.view(B, C, -1)
        k_x = x.view(B, C, -1).permute(0, 2, 1)
        energy = torch.bmm(q_x, k_x)
        attention = self.softmax(energy)
        v_x = x.view(B, C, -1)
        out = torch.bmm(attention, v_x).view(B, C, H, W)
        out = self.gamma * out + x
        return out


class SA_Block(nn.Module):
    def __init__(self, in_dim):
        super(SA_Block, self).__init__()
        self.query_conv = nn.Conv2d(in_dim, in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_dim, in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_dim, in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.ones(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        B, C, H, W = x.size()
        q_x = self.query_conv(x).view(B, -1, W * H)
        k_x = self.key_conv(x).view(B, -1, W * H)
        q_x = q_x.permute(0, 2, 1)
        energy = torch.bmm(q_x, k_x)
        attention = self.softmax(energy)
        v_x = self.value_conv(x).view(B, -1, W * H)
        out = torch.bmm(v_x, attention.permute(0, 2, 1)).view(B, C, H, W)
        out = self.gamma * out + x
        return out
class CoarseEstimationModule(nn.Module):
    def __init__(self, channels=[512, 256, 128, 64]):
        super(CoarseEstimationModule, self).__init__()
        self.feature_transforms = nn.ModuleList([
            nn.Sequential(
                nn.Upsample(size=384, mode='bilinear', align_corners=True),
                nn.Conv2d(ch, 64, kernel_size=1)
            ) for ch in channels
        ])
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(256, 1, kernel_size=7, padding=3),
            nn.Sigmoid()
        )
        self.estimation_head = nn.Sequential(
            nn.Conv2d(256, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, features):
        transformed_features = [
            transform(feat) for feat, transform in zip(features, self.feature_transforms)
        ]
        concat_features = torch.cat(transformed_features, dim=1)
        spatial_attn = self.spatial_attention(concat_features)
        attended_features = concat_features * spatial_attn
        coarse_estimation = self.estimation_head(attended_features)
        return coarse_estimation


class LearnableFilter(nn.Module):
    def __init__(self):
        super(LearnableFilter, self).__init__()

        self.filter_3x3 = nn.Parameter(torch.randn(1, 1, 3, 3))
        self.filter_5x5 = nn.Parameter(torch.randn(1, 1, 5, 5))
        self.filter_7x7 = nn.Parameter(torch.randn(1, 1, 7, 7))

    def forward(self, x, coarse_estimation):
        B, C, H, W = x.shape

        if coarse_estimation.shape[2:] != (H, W):
            coarse_estimation = F.interpolate(
                coarse_estimation, size=(H, W), mode='bilinear', align_corners=False
            )

        outputs = []
        for i in range(B):
            foreground_pixels = coarse_estimation[i].sum()
            ratio = foreground_pixels / (H * W)

            if ratio < 0.35:
                kernel = self.filter_3x3
            elif ratio < 0.5:
                kernel = self.filter_5x5
            else:
                kernel = self.filter_7x7

            kernel_expanded = kernel.expand(C, -1, -1, -1)
            xi = x[i:i+1]
            out = F.conv2d(xi, kernel_expanded, groups=C, padding=kernel.shape[-1] // 2)

            outputs.append(out)

        return torch.cat(outputs, dim=0)


class AttentionFilterFusion(nn.Module):
    def __init__(self, in_channels):
        super(AttentionFilterFusion, self).__init__()
        self.ca_block = CA_Block(in_channels)
        self.sa_block = SA_Block(in_channels)
        self.learnable_filter = LearnableFilter()
        self.conv = ConvBR(in_channels, in_channels, 3, stride=1, padding=1)

        self.gate = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x, coarse_estimation):
        ca_out = self.ca_block(x)
        sa_out = self.sa_block(x)
        sa_filtered = self.learnable_filter(sa_out, coarse_estimation)

        fusion_weight = self.gate(torch.cat((ca_out, sa_filtered), dim=1))
        fused_output = fusion_weight * ca_out + (1 - fusion_weight) * sa_filtered
        fused_output = self.conv(fused_output)
        return fused_output


class MLPBlock(nn.Module):

    def __init__(self,
                 dim,
                 n_div=2,
                 mlp_ratio=4.,
                 act_layer=nn.GELU,
                 norm_layer=nn.BatchNorm2d,
                 pconv_fw_type='split_cat'
                 ):
        super().__init__()
        self.dim = dim
        self.mlp_ratio = mlp_ratio
        self.n_div = n_div

        mlp_hidden_dim = int(dim * mlp_ratio)

        mlp_layer: List[nn.Module] = [
            nn.Conv2d(dim, mlp_hidden_dim, 1, bias=False),
            norm_layer(mlp_hidden_dim),
            act_layer(),
            nn.Conv2d(mlp_hidden_dim, dim, 1, bias=False)
        ]

        self.mlp = nn.Sequential(*mlp_layer)

        self.spatial_mixing = Partial_conv(
            dim,
            n_div,
            pconv_fw_type
        )
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    def forward(self, x: Tensor) -> Tensor:
        shortcut = x
        x = self.spatial_mixing(x)
        x = shortcut + self.mlp(x)
        x = self.up(x)
        return x


class Partial_conv(nn.Module):

    def __init__(self, dim, n_div, forward):
        super().__init__()
        self.dim_conv3 = dim // n_div
        self.dim_untouched = dim - self.dim_conv3
        self.partial_conv3 = nn.Conv2d(self.dim_conv3, self.dim_conv3, 3, 1, 1, bias=False)

        if forward == 'slicing':
            self.forward = self.forward_slicing
        elif forward == 'split_cat':
            self.forward = self.forward_split_cat
        else:
            raise NotImplementedError

    def forward_slicing(self, x: Tensor) -> Tensor:
        # only for inference
        x = x.clone()  # !!! Keep the original input intact for the residual connection later
        x[:, :self.dim_conv3, :, :] = self.partial_conv3(x[:, :self.dim_conv3, :, :])

        return x

    def forward_split_cat(self, x: Tensor) -> Tensor:
        # for training/inference
        x1, x2 = torch.split(x, [self.dim_conv3, self.dim_untouched], dim=1)
        x1 = self.partial_conv3(x1)
        x = torch.cat((x1, x2), 1)

        return x



class Mynet(nn.Module):
    def __init__(self, norm_layer=nn.LayerNorm):
        super(Mynet, self).__init__()
        self.smt = smt_t()  # Backbone
        self.coarse_estimation_module = CoarseEstimationModule()
        self.time_mixer = ImprovedTimeMixer(num_scales=4,channels=[512, 256, 128, 64])
        self.att1 = AttentionFilterFusion(in_channels=512)  # 适用于 f1
        self.att2 = AttentionFilterFusion(in_channels=256)  # 适用于 f2
        self.att3 = AttentionFilterFusion(in_channels=128)  # 适用于 f3
        self.att4 = AttentionFilterFusion(in_channels=64)  # 适用于 f4

        self.PCM1 = MLPBlock(dim=64)
        self.PCM2 = MLPBlock(dim=64)
        self.PCM3 = MLPBlock(dim=64)
        self.PCM4 = MLPBlock(dim=64)

        self.dwc3 = conv3x3_bn_relu(128, 128, k=1, s=1, p=0)
        self.dwc2 = conv3x3_bn_relu(256, 256, k=1, s=1, p=0)
        self.dwc1 = conv3x3_bn_relu(512, 512, k=1, s=1, p=0)

        self.dwcon1 = conv3x3_bn_relu(768, 256)
        self.dwcon2 = conv3x3_bn_relu(384, 128)
        self.dwcon3 = conv3x3_bn_relu(192, 64)

        self.xf_11 = nn.Conv2d(512, 64, kernel_size=1)
        self.xf_12 = nn.Conv2d(512, 64, kernel_size=1)
        self.xf_13 = nn.Conv2d(512, 64, kernel_size=1)
        self.xf_14 = nn.Conv2d(512, 64, kernel_size=1)
        self.xf_22 = nn.Conv2d(256, 64, kernel_size=1)
        self.xf_23 = nn.Conv2d(256, 64, kernel_size=1)
        self.xf_24 = nn.Conv2d(256, 64, kernel_size=1)
        self.xf_33 = nn.Conv2d(128, 64, kernel_size=1)
        self.xf_34 = nn.Conv2d(128, 64, kernel_size=1)

        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.deconv_layer_1 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.GELU(),
            self.upsample2
        )
        self.deconv_layer_2 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=1, bias=False),
        )
        self.deconv_layer_3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=1, bias=False),
        )
        self.deconv_layer_4 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=1, bias=False),
        )
        self.predict_layer_1 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.GELU(),
            self.upsample2,
            nn.Conv2d(in_channels=32, out_channels=1, kernel_size=3, padding=1, bias=True),
        )

        self.predtrans2 = nn.Conv2d(64, 1, kernel_size=3, padding=1)
        self.predtrans3 = nn.Conv2d(64, 1, kernel_size=3, padding=1)
        self.predtrans4 = nn.Conv2d(64, 1, kernel_size=3, padding=1)

    def forward(self, x):
        rgb_list = self.smt(x)

        r1 = rgb_list[3]  # 512,12
        r2 = rgb_list[2]  # 256,24
        r3 = rgb_list[1]  # 128,48
        r4 = rgb_list[0]  # 64,96,96

        coarse_estimation = self.coarse_estimation_module([r1, r2, r3, r4])

        f1, f2, f3, f4 = self.time_mixer([r1, r2, r3, r4])

        F1 = self.att1(f1, coarse_estimation)  # 512,12

        F1_up = F.interpolate(self.dwc1(F1), size=24, mode='bilinear')
        F2_con = torch.cat((f2, F1_up), 1)
        F2_con = self.dwcon1(F2_con)  # 256,24

        F2 = self.att2(F2_con, coarse_estimation)  # 256,24

        F2_up = F.interpolate(self.dwc2(F2), size=48, mode='bilinear')
        F3_con = torch.cat((f3, F2_up), dim=1)
        F3_con = self.dwcon2(F3_con)  # 128,48

        F3 = self.att3(F3_con, coarse_estimation)  # 128,48

        F3_up = F.interpolate(self.dwc3(F3), size=96, mode='bilinear')
        F4_con = torch.cat((f4, F3_up), dim=1)
        F4_con = self.dwcon3(F4_con)  # 64,96

        F4 = self.att4(F4_con, coarse_estimation)

        xf_11 = self.xf_11(F1)# 64, 96
        xf_12 = F.interpolate(self.xf_12(F1), size=24, mode='bilinear')
        xf_13 = F.interpolate(self.xf_13(F1), size=48, mode='bilinear')
        xf_14 = F.interpolate(self.xf_14(F1), size=96, mode='bilinear')
        xf_22 = self.xf_22(F2)
        xf_23 = F.interpolate(self.xf_23(F2), size=48, mode='bilinear')
        xf_24 = F.interpolate(self.xf_24(F2), size=96, mode='bilinear')
        xf_33 = self.xf_33(F3)
        xf_34 = F.interpolate(self.xf_34(F3), size=96, mode='bilinear')
        xf_44 = F4

        xf_4 = xf_44 + xf_14 + xf_24 + xf_34
        xf_3 = xf_33 + xf_23 + xf_13
        xf_2 = xf_22 + xf_12
        xf_1 = xf_11

        xf_1 = self.PCM1(xf_1)

        xc_1_2 = torch.cat((xf_1, xf_2), 1)
        df_f_2 = self.deconv_layer_2(xc_1_2)
        df_f_2 = self.PCM2(df_f_2)

        xc_1_3 = torch.cat((df_f_2, xf_3), 1)
        df_f_3 = self.deconv_layer_3(xc_1_3)
        df_f_3 = self.PCM3(df_f_3)

        xc_1_4 = torch.cat((df_f_3, xf_4), 1)
        df_f_4 = self.deconv_layer_4(xc_1_4)
        df_f_4 = self.PCM4(df_f_4)
        y1 = self.predict_layer_1(df_f_4)
        y2 = F.interpolate(self.predtrans2(df_f_3), size=384, mode='bilinear')
        y3 = F.interpolate(self.predtrans3(df_f_2), size=384, mode='bilinear')
        y4 = F.interpolate(self.predtrans4(xf_1), size=384, mode='bilinear')
        s_coarse = F.interpolate(coarse_estimation, size=384, mode='bilinear', align_corners=True)
        return  y1, y2, y3, y4, s_coarse
    def load_pre(self, pre_model):
        state = torch.load(pre_model, map_location='cpu', weights_only=False)['model']
        self.smt.load_state_dict(state)
        print(f"loading pre_model {pre_model}")



