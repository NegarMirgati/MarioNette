"""Model architecture.

Defintions:
    - layer: refers to a raster layer in the composite. Each layer is assembled
      from multiple patches. A rendering can have multiple layers ordered from
      back to front.
"""

import numpy as np
import torch as th
from torch import nn
import torch.nn.functional as F

from .partialconv import PartialConv2d
from .partialconv3d import PartialConv3d


class Dictionary(nn.Module):
    def __init__(
        self,
        num_classes,
        patch_size,
        num_chans,
        bottleneck_size=128,
        no_layernorm=False,
    ):
        super().__init__()

        self.patch_size = patch_size
        self.num_chans = num_chans
        self.no_layernorm = no_layernorm

        self.latent = nn.Parameter(th.randn(num_classes, bottleneck_size))
        self.decode = nn.Sequential(
            nn.Linear(bottleneck_size, 8 * bottleneck_size),
            nn.GroupNorm(8, 8 * bottleneck_size),
            nn.ReLU(inplace=True),
            nn.Linear(8 * bottleneck_size, num_chans * patch_size[0] * patch_size[1]),
            nn.Sigmoid(),
        )

    def forward(self, x=None):
        if x is None and not self.no_layernorm:
            x = F.layer_norm(self.latent, (self.latent.shape[-1],))
        out = self.decode(x).view(-1, self.num_chans, *self.patch_size)
        return out, x


class _DownBlock(nn.Module):
    def __init__(
        self,
        in_ch,
        out_ch,
        return_mask=True,
        is_sequential=False,
    ):
        super().__init__()
        self.return_mask = return_mask
        if not is_sequential:
            self.conv1 = PartialConv2d(in_ch, out_ch, 3, padding=1, return_mask=True)
            self.conv2 = PartialConv2d(
                out_ch, out_ch, 3, padding=1, stride=2, return_mask=True
            )
        else:
            self.conv1 = PartialConv3d(in_ch, out_ch, 3, padding=1, return_mask=True)
            self.conv2 = PartialConv3d(
                out_ch, out_ch, 3, padding=1, stride=2, return_mask=True
            )
        self.norm1 = nn.GroupNorm(8, out_ch)
        self.norm2 = nn.GroupNorm(8, out_ch)

        self.nonlinearity = nn.LeakyReLU(inplace=True)

    def forward(self, x, mask=None):
        y, mask = self.conv1(x, mask)
        y = self.nonlinearity(self.norm1(y))
        y, mask = self.conv2(y, mask)
        y = self.nonlinearity(self.norm2(y))

        if self.return_mask:
            return y, mask
        else:
            return y


class Encoder(nn.Module):
    """Encodes image data into a grid of patch latent codes used for affinity
    matching.
    The encoder is a chain of blocks that each dowsamples by 2x. It outputs
    a list of latent codes for the patches in each layer. Each layer can have a
    variable (power of two) number of patches. The latent codes are predicted
    from the corresponding block.
    im
     |
     V
    block1 -> (optional) latent codes for all layers with 2x2 patches
     |
     V
    block2 -> (optional) latent codes for all layers with 4x4 patches
     |
     V
    ...    -> ...
    Args:
        num_channels(int): number of image channels (e.g. 4 for RGBA images).
        canvas_size(int): size of the (square) input image.
        layer_sizes(list of int): list of patch count along the x (resp. y)
            dimension for each layer. The number of layers is the length of
            the list.
    """

    def __init__(
        self,
        num_channels,
        canvas_size,
        layer_sizes,
        dim_z=1024,
        no_layernorm=False,
        is_sequential=False,
    ):
        super().__init__()

        self.canvas_size = canvas_size
        self.layer_sizes = layer_sizes
        self.num_channels = num_channels
        self.no_layernorm = no_layernorm

        # This should represent number of down blocks
        num_ds = int(np.log2(canvas_size / min(layer_sizes)))

        self.blocks = nn.ModuleList()
        self.heads = nn.ModuleList()
        for i in range(num_ds):
            in_ch = num_channels if i == 0 else dim_z
            self.blocks.append(
                _DownBlock(in_ch, dim_z, return_mask=True, is_sequential=is_sequential)
            )
            for lsize in layer_sizes:
                if canvas_size // (2 ** (i + 1)) == lsize:
                    #                              (in_ch, out_ch, kernel_s, stride=1)
                    self.heads.append(PartialConv2d(dim_z, dim_z, 3, padding=1))

    def forward(self, x):
        out = [None] * len(self.layer_sizes)

        y = x
        mask = None
        for my_counter, _block in enumerate(self.blocks):
            y, mask = _block(y, mask)  # encoding + downsampling step

            if my_counter == len(self.blocks) - 1:
                print("y shape", y.shape)
                print("mask shape", mask.shape)
                # y = F.avg_pool3d(y, kernel_size=(1, 1, 2), stride=1)
                y = th.flatten(y, -2, -1)
                mask = th.flatten(mask, -2, -1)

            # Look for layers whose spatial dimension match the current block
            for i, l in enumerate(self.layer_sizes):
                if y.shape[-2] == l:
                    # size match, output the latent codes for this layer
                    out[i] = self.heads[i](y, mask).permute(0, 2, 3, 1).contiguous()
                    if not self.no_layernorm:
                        out[i] = F.layer_norm(out[i], (out[i].shape[-1],))

        # Check all outputs were set
        for o in out:
            if o is None:
                raise RuntimeError("Unexpected output count for Encoder.")

        return out


class Model(nn.Module):
    def __init__(
        self,
        # learned_dict,
        # NOTE: new dictionary for playe sprites
        learned_dict_player,
        learned_dict_non_player,
        layer_size,
        num_layers,
        sequence_length=2,
        patch_size=1,
        canvas_size=128,
        dim_z=128,
        shuffle_all=False,
        bg_color=None,
        no_layernorm=False,
        no_spatial_transformer=False,
        spatial_transformer_bg=False,
        straight_through_probs=False,
    ):
        super().__init__()

        self.layer_size = layer_size
        self.num_layers = num_layers
        self.canvas_size = canvas_size
        self.patch_size = canvas_size // layer_size
        self.dim_z = dim_z
        self.shuffle_all = shuffle_all
        self.no_spatial_transformer = no_spatial_transformer
        self.spatial_transformer_bg = spatial_transformer_bg
        self.straight_through_probs = straight_through_probs

        self.im_encoder = Encoder(
            3,
            canvas_size,
            [layer_size] * num_layers,
            dim_z,
            no_layernorm=no_layernorm,
            is_sequential=True,
        )

        self.project = nn.Sequential(
            nn.Linear(dim_z, dim_z),
            (
                nn.LayerNorm(dim_z, elementwise_affine=False)
                if not no_layernorm
                else nn.Identity()
            ),
        )

        # self.encoder_xform = Encoder(
        #     7, self.patch_size * 2, [1], dim_z, no_layernorm=no_layernorm
        # )
        # NOTE: player dynamics encoder
        self.encoder_player_dynamics = Encoder(
            7, self.patch_size * 2, [1], dim_z, no_layernorm=no_layernorm
        )
        # NOTE: non-player dynamics encoder
        self.encoder_non_player_dynamics = Encoder(
            7, self.patch_size * 2, [1], dim_z, no_layernorm=no_layernorm
        )
        self.probs = nn.Sequential(
            nn.Linear(dim_z, dim_z),
            nn.GroupNorm(8, dim_z),
            nn.LeakyReLU(),
            nn.Linear(dim_z, 1),
            nn.Sigmoid(),
        )

        if self.no_spatial_transformer:
            self.xforms_x = nn.Sequential(
                nn.Linear(dim_z, dim_z),
                nn.GroupNorm(8, dim_z),
                nn.LeakyReLU(),
                nn.Linear(dim_z, self.patch_size + 1),
                nn.Softmax(dim=-1),
            )
            self.xforms_y = nn.Sequential(
                nn.Linear(dim_z, dim_z),
                nn.GroupNorm(8, dim_z),
                nn.LeakyReLU(),
                nn.Linear(dim_z, self.patch_size + 1),
                nn.Softmax(dim=-1),
            )
        else:
            # NOTE: based on last discussing, this won't be needed anymore!
            # self.shifts = nn.Sequential(
            #     nn.Linear(dim_z, dim_z),
            #     nn.GroupNorm(8, dim_z),
            #     nn.LeakyReLU(),
            #     nn.Linear(dim_z, 2),
            #     nn.Tanh(),
            # )
            # NOTE: adding two encoders, one for player dynamics, and one for other sprite dynamics
            # NOTE: the below are transform sprites block
            self.player_shifts = nn.Sequential(
                nn.Linear(dim_z, dim_z),
                nn.GroupNorm(8, dim_z),
                nn.LeakyReLU(),
                nn.Linear(dim_z, 2),
                nn.Tanh(),
            )
            self.non_player_shift = nn.Sequential(
                nn.Linear(dim_z, dim_z),
                nn.GroupNorm(8, dim_z),
                nn.LeakyReLU(),
                nn.Linear(dim_z, 2),
                nn.Tanh(),
            )

        # self.learned_dict = learned_dict
        # NOTE: new dictionary for player sprites
        self.learned_dict_player = learned_dict_player
        self.learned_dict_non_player = learned_dict_non_player

        if bg_color is None:
            self.bg_encoder = Encoder(
                3, canvas_size, [1], dim_z, no_layernorm=no_layernorm
            )
            if self.spatial_transformer_bg:
                self.bg_shift = nn.Sequential(
                    nn.Linear(dim_z, dim_z),
                    nn.GroupNorm(8, dim_z),
                    nn.LeakyReLU(),
                    nn.Linear(dim_z, 1),
                    nn.Tanh(),
                )
            else:
                self.bg_x = nn.Sequential(
                    nn.Linear(dim_z, dim_z),
                    nn.GroupNorm(8, dim_z),
                    nn.LeakyReLU(),
                    nn.Linear(dim_z, 3 * canvas_size + 1),
                    nn.Softmax(dim=-1),
                )
        else:
            self.bg_color = nn.Parameter(th.tensor(bg_color), requires_grad=False)

    def forward(self, im, bg, hard=False, custom_dict=None, rng=None, custom_bg=None):
        bs = im.shape[0]

        # learned_dict, dict_codes = self.learned_dict()
        # NOTE: new player and non-player dict
        learned_dict_player, dict_codes_player = self.learned_dict_player()
        learned_dict_non_player, dict_codes_non_player = self.learned_dict_non_player()

        if rng is not None:
            # learned_dict = learned_dict[rng]
            # dict_codes = dict_codes[rng]
            learned_dict_player = learned_dict_player[rng]
            dict_codes_player = dict_codes_player[rng]
            learned_dict_non_player = learned_dict_non_player[rng]
            dict_codes_non_player = dict_codes_non_player[rng]

        im_codes = th.stack(self.im_encoder(im), dim=1)
        probs = self.probs(im_codes.flatten(0, 3))
        if self.straight_through_probs:
            probs = probs.round() - probs.detach() + probs

        # NOTE: now we have two sets of logits, one applying player codes and one non-player codes

        # logits = (self.project(im_codes) @ dict_codes.transpose(0, 1)) / np.sqrt(
        #     im_codes.shape[-1]
        # )
        logits_player = (
            self.project(im_codes) @ dict_codes_player.transpose(0, 1)
        ) / np.sqrt(im_codes.shape[-1])
        logits_non_player = (
            self.project(im_codes) @ dict_codes_non_player.transpose(0, 1)
        ) / np.sqrt(im_codes.shape[-1])
        # NOTE: applying softmax on logits - now we need one for each player and non-player

        # weights = F.softmax(logits, dim=-1)
        weights_player = F.softmax(logits_player, dim=-1)
        weights_non_player = F.softmax(logits_non_player, dim=-1)

        # NOTE: let's obtain the patches now

        # patches = (weights[..., None, None, None] * learned_dict).sum(4)

        patches_player = (
            weights_player[..., None, None, None] * learned_dict_player
        ).sum(4)
        patches_non_player = (
            weights_non_player[..., None, None, None] * learned_dict_non_player
        ).sum(4)

        # patches = patches.flatten(0, 3)
        patches_player = patches_player.flatten(0, 3)
        patches_non_player = patches_non_player.flatten(0, 3)

        # NOTE: now create images of those patches
        # im_patches = F.pad(im, (self.patch_size // 2,) * 4)
        im_patches_player = F.pad(im, (self.patch_size // 2,) * 4)
        im_patches_non_player = F.pad(im, (self.patch_size // 2,) * 4)

        # im_patches = im_patches.unfold(2, self.patch_size * 2, self.patch_size).unfold(
        #     3, self.patch_size * 2, self.patch_size
        # )
        im_patches_player = im_patches_player.unfold(
            2, self.patch_size * 2, self.patch_size
        ).unfold(3, self.patch_size * 2, self.patch_size)
        im_patches_non_player = im_patches_non_player.unfold(
            2, self.patch_size * 2, self.patch_size
        ).unfold(3, self.patch_size * 2, self.patch_size)
        # im_patches = (
        #     im_patches.reshape(
        #         bs,
        #         3,
        #         self.layer_size,
        #         self.layer_size,
        #         2 * self.patch_size,
        #         2 * self.patch_size,
        #     )
        #     .permute(0, 2, 3, 1, 4, 5)
        #     .contiguous()
        # )
        im_patches_player = (
            im_patches_player.reshape(
                bs,
                3,
                self.layer_size,
                self.layer_size,
                2 * self.patch_size,
                2 * self.patch_size,
            )
            .permute(0, 2, 3, 1, 4, 5)
            .contiguous()
        )
        im_patches_non_player = (
            im_patches_non_player.reshape(
                bs,
                3,
                self.layer_size,
                self.layer_size,
                2 * self.patch_size,
                2 * self.patch_size,
            )
            .permute(0, 2, 3, 1, 4, 5)
            .contiguous()
        )
        # im_patches = (
        #     im_patches[:, None].repeat(1, self.num_layers, 1, 1, 1, 1, 1).flatten(0, 3)
        # )
        im_patches_player = (
            im_patches_player[:, None]
            .repeat(1, self.num_layers, 1, 1, 1, 1, 1)
            .flatten(0, 3)
        )

        im_patches_non_player = (
            im_patches_non_player[:, None]
            .repeat(1, self.num_layers, 1, 1, 1, 1, 1)
            .flatten(0, 3)
        )

        # codes_xform = (
        #     self.encoder_xform(th.cat([im_patches, patches], dim=1))[0]
        #     .squeeze(-2)
        #     .squeeze(-2)
        # )
        #

        # NOTE: perform spatial transformation for player and non-player separately
        codes_xform_player = (
            self.encoder_player_dynamics(
                th.cat([im_patches_player, patches_player], dim=1)
            )[0]
            .squeeze(-2)
            .squeeze(-2)
        )

        codes_xform_non_player = (
            self.encoder_non_player_dynamics(
                th.cat([im_patches_non_player, patches_non_player], dim=1)
            )[0]
            .squeeze(-2)
            .squeeze(-2)
        )

        if hard:
            weights = th.eye(weights.shape[-1]).to(weights)[weights.argmax(-1)]
            probs = probs.round()
            patches = (weights[..., None, None, None] * learned_dict).sum(4)
            patches = patches.flatten(0, 3)

        if custom_dict is not None:
            learned_dict = custom_dict
            patches = (weights[..., None, None, None] * learned_dict).sum(4)
            patches = patches.flatten(0, 3)

        # NOTE: now we have two patches, one for player and one for non-player
        # patches = patches * probs[:, :, None, None]
        patches = patches * probs[:, :, None, None]

        if self.no_spatial_transformer:
            xforms_x = self.xforms_x(codes_xform)
            xforms_y = self.xforms_y(codes_xform)

            if hard:
                xforms_x = th.eye(xforms_x.shape[-1]).to(xforms_x)[xforms_x.argmax(-1)]
                xforms_y = th.eye(xforms_y.shape[-1]).to(xforms_y)[xforms_y.argmax(-1)]

            patches = F.pad(patches, (self.patch_size // 2,) * 4)
            patches = patches.unfold(2, self.patch_size * 2, 1)
            patches = (patches * xforms_y[:, None, :, None, None]).sum(2)
            patches = patches.unfold(2, self.patch_size * 2, 1)
            patches = (patches * xforms_x[:, None, :, None, None]).sum(2)
        else:
            # shifts = self.shifts(codes_xform) / 2
            # NOTE: preform shift on player xforms
            shifts_player = self.player_shifts(codes_xform_player) / 2

            # NOTE: preform shift on non-player xforms
            shifts_non_player = self.non_player_shifts(codes_xform_non_player) / 2

            theta = th.eye(2)[None].repeat(shifts.shape[0], 1, 1).to(shifts)
            theta = th.cat([theta, -shifts[:, :, None]], dim=-1)
            grid = F.affine_grid(
                theta,
                [patches.shape[0], 1, self.patch_size * 2, self.patch_size * 2],
                align_corners=False,
            )

            patches_rgb, patches_a = th.split(patches, [3, 1], dim=1)
            patches_rgb = F.grid_sample(
                patches_rgb,
                grid,
                align_corners=False,
                padding_mode="border",
                mode="bilinear",
            )
            patches_a = F.grid_sample(
                patches_a,
                grid,
                align_corners=False,
                padding_mode="zeros",
                mode="bilinear",
            )
            patches = th.cat([patches_rgb, patches_a], dim=1)

        patches = patches.view(
            bs,
            self.num_layers,
            self.layer_size,
            self.layer_size,
            -1,
            2 * self.patch_size,
            2 * self.patch_size,
        ).permute(0, 1, 4, 2, 5, 3, 6)

        group1 = patches[..., ::2, :, ::2, :].contiguous()
        group1 = group1.view(
            bs, self.num_layers, -1, self.canvas_size, self.canvas_size
        )
        group1 = group1[..., self.patch_size // 2 :, self.patch_size // 2 :]
        group1 = F.pad(group1, (0, self.patch_size // 2, 0, self.patch_size // 2))

        group2 = patches[..., 1::2, :, 1::2, :].contiguous()
        group2 = group2.view(
            bs, self.num_layers, -1, self.canvas_size, self.canvas_size
        )
        group2 = group2[..., : -self.patch_size // 2, : -self.patch_size // 2]
        group2 = F.pad(group2, (self.patch_size // 2, 0, self.patch_size // 2, 0))

        group3 = patches[..., 1::2, :, ::2, :].contiguous()
        group3 = group3.view(
            bs, self.num_layers, -1, self.canvas_size, self.canvas_size
        )
        group3 = group3[..., : -self.patch_size // 2, self.patch_size // 2 :]
        group3 = F.pad(group3, (0, self.patch_size // 2, self.patch_size // 2, 0))

        group4 = patches[..., ::2, :, 1::2, :].contiguous()
        group4 = group4.view(
            bs, self.num_layers, -1, self.canvas_size, self.canvas_size
        )
        group4 = group4[..., self.patch_size // 2 :, : -self.patch_size // 2]
        group4 = F.pad(group4, (self.patch_size // 2, 0, 0, self.patch_size // 2))

        layers = th.stack([group1, group2, group3, group4], dim=2)
        layers_out = layers.clone()

        if self.shuffle_all:
            layers = layers.flatten(1, 2)[:, th.randperm(4 * self.num_layers)]
        else:
            layers = layers[:, :, th.randperm(4)].flatten(1, 2)

        if bg is not None:
            bg_codes = self.bg_encoder(im)[0].squeeze(-2).squeeze(-2)
            if not self.spatial_transformer_bg:
                bg_x = self.bg_x(bg_codes)
                bgs = bg.squeeze(0).unfold(2, self.canvas_size, 1)
                out = (bgs[None] * bg_x[:, None, None, :, None]).sum(3)
            else:
                shift = self.bg_shift(bg_codes) * 3 / 4
                shift = th.cat([shift, th.zeros_like(shift)], dim=-1)
                theta = th.eye(2)[None].repeat(shift.shape[0], 1, 1).to(shift)
                theta[:, 0, 0] = 1 / 4
                theta = th.cat([theta, -shift[:, :, None]], dim=-1)
                grid = F.affine_grid(
                    theta,
                    [bs, 1, self.canvas_size, self.canvas_size],
                    align_corners=False,
                )

                out = F.grid_sample(
                    bg.repeat(bs, 1, 1, 1),
                    grid,
                    align_corners=False,
                    padding_mode="border",
                    mode="bilinear",
                )

        else:
            if custom_bg is not None:
                out = (
                    custom_bg[None, :, None, None]
                    .clamp(0, 1)
                    .repeat(bs, 1, self.canvas_size, self.canvas_size)
                )
            else:
                out = (
                    self.bg_color[None, :, None, None]
                    .clamp(0, 1)
                    .repeat(bs, 1, self.canvas_size, self.canvas_size)
                )
            bg = (
                self.bg_color[None, :, None, None]
                .clamp(0, 1)
                .repeat(1, 1, self.canvas_size, self.canvas_size)
            )

        rgb, a = th.split(layers, [3, 1], dim=2)

        for i in range(4 * self.num_layers):
            out = (1 - a[:, i]) * out + a[:, i] * rgb[:, i]

        ret = {
            # NOTE: now we have two sets of weights
            # "weights": weights,
            "weights_player": weights_player,
            "weights_non_player": weights_non_player,
            "probs": probs.view(bs, self.num_layers, -1),
            "layers": layers_out,
            "patches": patches,
            # "dict_codes": dict_codes,
            "dict_codes_player": dict_codes_player,
            "dict_codes_non_player": dict_codes_non_player,
            "im_codes": im_codes.flatten(0, 1),
            "reconstruction": out,
            "dict": learned_dict,
            "dict_player": learned_dict_player,
            "background": bg,
        }

        if not self.no_spatial_transformer:
            # ret["shifts"] = shifts
            ret["shifts_player"] = shifts_player
            ret["shifts_non_player"] = shifts_non_player

        return ret
