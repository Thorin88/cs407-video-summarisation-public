from torch import nn

from anchor_free import anchor_free_helper
from modules.models import build_base_model

class FeatureDSNetAF(nn.Module):
    def __init__(self, base_model, num_feature, num_hidden, num_head):
        super().__init__()
        self.base_model = build_base_model(base_model, num_feature, num_head)
        self.layer_norm = nn.LayerNorm(num_feature)

        self.fc1 = nn.Sequential(
            # 1024,128
            nn.Linear(num_feature, num_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.LayerNorm(num_hidden)
        )
        self.fc_cls = nn.Linear(num_hidden, 1)
        self.fc_loc = nn.Linear(num_hidden, 2)
        self.fc_ctr = nn.Linear(num_hidden, 1)

    def forward(self, x):
        _, seq_len, _ = x.shape
        # Base model used is configurable in args
        out = self.base_model(x)
        # x_j = w_j + v_j
        out = out + x
        out = self.layer_norm(out)
        # Based on the paper, it looks like the feature extraction
        # is the output of this layer. (see page 6)

        # out = self.fc1(out)

        # pred_cls = self.fc_cls(out).sigmoid().view(seq_len)
        # pred_loc = self.fc_loc(out).exp().view(seq_len, 2)
        #
        # pred_ctr = self.fc_ctr(out).sigmoid().view(seq_len)
        #
        # return pred_cls, pred_loc, pred_ctr
        return out

    # Do not use this for feature extraction
    def predict(self, seq):

        return None
