"""Model training interface."""

import torch as th
from torch.distributions import Beta
from torch.nn import functional as F
import ttools
from ttools.modules import image_operators as imops


LOG = ttools.get_logger(__name__)


class Interface(ttools.ModelInterface):
    def __init__(
        self,
        model,
        device="cpu",
        lr=1e-4,
        w_beta=0,
        w_probs=0,
        lr_bg=None,
        background=None,
    ):
        self.model = model

        if lr_bg is None:
            lr_bg = lr

        self.opt = th.optim.AdamW(model.parameters(), lr=lr)

        self.device = device
        self.model.to(device)
        self.background = background
        if background is not None:
            self.background.to(device)
            self.opt_bg = th.optim.AdamW(self.background.parameters(), lr=lr_bg)
        else:
            self.opt_bg = None

        self.w_beta = w_beta
        self.w_probs = w_probs

        self.beta = Beta(th.tensor(2.0).to(device), th.tensor(2.0).to(device))

        self.loss = th.nn.MSELoss()
        self.loss.to(device)

    def forward(self, im, hard=False):
        if self.background is not None:
            bg, _ = self.background()
        else:
            bg = None
        return self.model(im, bg, hard=hard)

    def training_step(self, batch):
        im = batch["im"].to(self.device)

        fwd_data = self.forward(im)
        fwd_data_hard = self.forward(im, hard=True)

        out = fwd_data["reconstruction"]
        layers = fwd_data["layers"]
        out_hard = fwd_data_hard["reconstruction"]
        layers_hard = fwd_data_hard["layers"]
        im = imops.crop_like(im[:, -1, :, :, :], out)

        # NOTE: now we have two dicts
        # learned_dict = fwd_data["dict"]
        learned_dict_player = fwd_data["dict_player"]
        learned_dict_non_player = fwd_data["dict_non_player"]
        # dict_codes = fwd_data["dict_codes"]
        dict_codes_player = fwd_data["dict_codes_player"]
        dict_codes_non_player = fwd_data["dict_codes_non_player"]

        im_codes = fwd_data["im_codes"]
        # NOTE: now we have two sets of weights
        # weights = fwd_data["weights"]
        weights_player = fwd_data["weights_player"]
        weights_non_player = fwd_data["weights_non_player"]
        probs = fwd_data["probs"]

        rec_loss = self.loss(out, im)
        beta_loss = (
            self.beta.log_prob(weights_player.clamp(1e-5, 1 - 1e-5)).exp().mean()
            + self.beta.log_prob(weights_non_player.clamp(1e-5, 1 - 1e-5)).exp().mean()
            + self.beta.log_prob(probs.clamp(1e-5, 1 - 1e-5)).exp().mean()
        ) / 2

        probs_loss = probs.abs()

        self.opt.zero_grad()
        if self.opt_bg is not None:
            self.opt_bg.zero_grad()

        w_probs = (
            th.tensor(self.w_probs).to(probs_loss)[None, :, None].expand_as(probs_loss)
        )
        loss = rec_loss + self.w_beta * beta_loss + (w_probs * probs_loss).mean()

        loss.backward()

        self.opt.step()
        if self.opt_bg is not None:
            self.opt_bg.step()

        with th.no_grad():
            psnr = -10 * th.log10(F.mse_loss(out, im))
            psnr_hard = -10 * th.log10(F.mse_loss(out_hard, im))

        return {
            "rec_loss": rec_loss.item(),
            "beta_loss": beta_loss.item(),
            "psnr": psnr.item(),
            "psnr_hard": psnr_hard.item(),
            "out": out.detach(),
            "layers": layers.detach(),
            "out_hard": out_hard.detach(),
            "layers_hard": layers_hard.detach(),
            # "dict": learned_dict.detach(),
            "dict_player": learned_dict_player.detach(),
            "dict_non_player": learned_dict_non_player.detach(),
            "dict_codes_player": dict_codes_player.detach(),
            "dict_codes_non_player": dict_codes_non_player.detach(),
            "probs_loss": probs_loss.mean().item(),
            "im_codes": im_codes.detach(),
            # "dict_codes": dict_codes.detach(),
            "background": fwd_data["background"].detach(),
        }
