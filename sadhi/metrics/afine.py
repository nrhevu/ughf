import torch
from torch import nn as nn
from torch.nn import functional as F


class AFINEQhead(nn.Module):
    def __init__(
        self,
        chns=(3, 768, 768, 768, 768, 768, 768, 768, 768, 768, 768, 768, 768),
        feature_out_channel=1,
        input_dim=768,
        hidden_dim=128,
        mean=(0.48145466, 0.4578275, 0.40821073),
        std=(0.26862954, 0.26130258, 0.27577711),
    ):
        super(AFINEQhead, self).__init__()

        self.chns = chns
        self.feature_out_channel = feature_out_channel
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.register_buffer("mean", torch.tensor(mean).view(1, -1, 1, 1))
        self.register_buffer("std", torch.tensor(std).view(1, -1, 1, 1))

        self.proj_feat = nn.Linear(input_dim * 2, hidden_dim)
        self.proj_head = nn.Sequential(
            nn.Linear(
                self.chns[0] * 2 + hidden_dim * (len(self.chns) - 1), hidden_dim * 6
            ),
            nn.GELU(),
            nn.Linear(hidden_dim * 6, self.feature_out_channel),
        )

    def forward(self, x, h_list_x):
        x = x * self.std + self.mean

        img_feature_x = x.flatten(2).permute(0, 2, 1)

        feature_list_x = []

        feature_list_x.append(img_feature_x)
        for h_x in h_list_x:
            feature_list_x.append(F.relu(h_x))

        final_feature_list_x = []

        for k in range(len(self.chns)):
            x_mean = feature_list_x[k].mean(1, keepdim=True)

            x_var = ((feature_list_x[k] - x_mean) ** 2).mean(1, keepdim=True)

            concat_x_feature = torch.cat((x_mean.flatten(1), x_var.flatten(1)), dim=1)

            if k != 0:
                concat_x_feature = self.proj_feat(concat_x_feature)

            final_feature_list_x.append(concat_x_feature)

        concat_final_feature_lixt_x = torch.cat(final_feature_list_x, dim=1)

        n_x = self.proj_head(concat_final_feature_lixt_x)

        return n_x


class AFINEDhead(nn.Module):
    def __init__(
        self,
        chns=(3, 768, 768, 768, 768, 768, 768, 768, 768, 768, 768, 768, 768),
        mean=(0.48145466, 0.4578275, 0.40821073),
        std=(0.26862954, 0.26130258, 0.27577711),
    ):
        super(AFINEDhead, self).__init__()

        self.chns = chns

        self.register_parameter(
            "alpha", nn.Parameter(torch.randn(1, 1, sum(self.chns)), requires_grad=True)
        )
        self.register_parameter(
            "beta", nn.Parameter(torch.randn(1, 1, sum(self.chns)), requires_grad=True)
        )
        self.alpha.data.normal_(0.1, 0.01)
        self.beta.data.normal_(0.1, 0.01)

        self.softplus = nn.Softplus()

        self.register_buffer("mean", torch.tensor(mean).view(1, -1, 1, 1))
        self.register_buffer("std", torch.tensor(std).view(1, -1, 1, 1))

    def forward(self, x, y, h_list_x, h_list_y):
        ### the input image should be generalized back to its original values
        x = x * self.std + self.mean
        y = y * self.std + self.mean

        # print(f"mean is {self.mean}, std is {self.std}")

        img_feature_x = x.flatten(2).permute(0, 2, 1)
        img_feature_y = y.flatten(2).permute(0, 2, 1)

        feature_list_x = []
        feature_list_y = []

        feature_list_x.append(img_feature_x)
        for h_x in h_list_x:
            feature_list_x.append(F.relu(h_x))

        feature_list_y.append(img_feature_y)
        for h_y in h_list_y:
            feature_list_y.append(F.relu(h_y))

        dist1 = 0
        dist2 = 0
        c1 = 1e-10
        c2 = 1e-10

        alpha_ = self.softplus(self.alpha)
        beta_ = self.softplus(self.beta)

        w_sum = alpha_.sum() + beta_.sum() + 1e-10
        alpha = torch.split(alpha_ / w_sum, self.chns, dim=2)
        beta = torch.split(beta_ / w_sum, self.chns, dim=2)

        for k in range(len(self.chns)):
            x_mean = feature_list_x[k].mean(1, keepdim=True)
            y_mean = feature_list_y[k].mean(1, keepdim=True)

            S1 = (2 * x_mean * y_mean + c1) / (x_mean**2 + y_mean**2 + c1)
            dist1 = dist1 + (alpha[k] * S1).sum(2, keepdim=True)

            x_var = ((feature_list_x[k] - x_mean) ** 2).mean(1, keepdim=True)
            y_var = ((feature_list_y[k] - y_mean) ** 2).mean(1, keepdim=True)
            xy_cov = (feature_list_x[k] * feature_list_y[k]).mean(
                1, keepdim=True
            ) - x_mean * y_mean
            S2 = (2 * xy_cov + c2) / (x_var + y_var + c2)
            dist2 = dist2 + (beta[k] * S2).sum(2, keepdim=True)

        score = 1 - (dist1 + dist2).squeeze(2)

        return score
