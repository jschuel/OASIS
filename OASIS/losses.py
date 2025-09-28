import torch
import torch.nn as nn
import torch.nn.functional as F

def masked_tv(I_hat: torch.Tensor, valid: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    I_hat: [B,2,H,W], valid: [B,1,H,W]
    TV over edges fully inside 'valid' (anisotropic), normalized by valid edge count.
    """
    vy = valid[..., 1:, :] * valid[..., :-1, :]
    vx = valid[..., :, 1:] * valid[..., :, :-1]

    dy = (I_hat[..., 1:, :] - I_hat[..., :-1, :]).abs() * vy
    dx = (I_hat[..., :, 1:] - I_hat[..., :, :-1]).abs() * vx

    num = dy.sum() + dx.sum()
    den = vy.sum() + vx.sum() + eps
    return num / den

class SegRegLoss(nn.Module):
    """
    Segmentation–regression loss
    Includes: 1. Channel and region-weighted regression loss
              2. smoothness loss (masked_tv)
    """
    def __init__(self,
                 alpha_reg=1.0, alpha_tv=0.01,
                 w_ch=(5.0, 1.0),                 # per-channel (ER, NR)
                 w_regions=(3.0, 1.0, 6.0)):      # (ER-only, NR-only, Overlap)
        super().__init__()
        self.alpha_reg   = alpha_reg
        self.alpha_tv    = alpha_tv
        self.register_buffer("w_ch", torch.tensor(w_ch, dtype=torch.float32).view(1,2,1,1))
        self.w_eronly, self.w_nronly, self.w_ov = w_regions

    @staticmethod
    def _region_weights_from_targets(t: torch.Tensor, valid: torch.Tensor,
                                     w_eronly: float, w_nronly: float, w_ov: float):
        """
        Determines region weights given ER and NR targets.
        """
        y_er = (t[:, 0:1] > 0).float()
        y_nr = (t[:, 1:2] > 0).float()

        M_eronly = y_er * (1 - y_nr) * valid
        M_nronly = y_nr * (1 - y_er) * valid
        M_ov     = y_er * y_nr       * valid

        W = (w_eronly * M_eronly +
             w_nronly * M_nronly +
             w_ov     * M_ov)

        bg_valid = (valid * (1 - (M_eronly + M_nronly + M_ov).clamp(0,1))).float()
        W = W + 1.0 * bg_valid  # weight 1 in remaining valid

        scale = (valid.sum() / (W.sum() + 1e-6)).detach()
        return W * scale  # [B,1,H,W]

    def forward(self, I_hat: torch.Tensor, I_in: torch.Tensor,
                t: torch.Tensor, valid: torch.Tensor):
        v2 = valid.expand(-1, 2, -1, -1)
        #region weights
        Wreg = self._region_weights_from_targets(t, valid, self.w_eronly, self.w_nronly, self.w_ov)
        #channel weights
        wch = self.w_ch.to(I_hat.device)
        L = (wch * (I_hat - t).abs()) * v2 * Wreg
        denom_reg = (wch * v2 * Wreg).sum() + 1e-6
        l_reg = L.sum() / denom_reg

        l_tv = masked_tv(I_hat, valid)

        loss = (self.alpha_reg   * l_reg +
                self.alpha_tv    * l_tv)

        return {"loss": loss, "reg": l_reg.detach(),
                "tv": l_tv.detach()}
