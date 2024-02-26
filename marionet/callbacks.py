"""Visualizatin helpers."""

import torch as th
import numpy as np
from sklearn.manifold import TSNE
from torch.nn import functional as F
from ttools import callbacks as cb

import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


class VizCallback(cb.TensorBoardImageDisplayCallback):
    def __init__(self, suffix, *args, **kwargs):
        self.suffix = suffix
        super().__init__(*args, **kwargs)

    def tag(self):
        return f"out{self.suffix}_output_diff"

    def visualized_image(self, batch, fwd_data):
        image = batch["im"].detach()
        print("here ......", image.size())
        image = image[:, -1, :, :, :]
        image = F.pad(image, (1, 1, 1, 1), value=0)

        out = fwd_data[f"out{self.suffix}"].cpu().detach()[:, :3]
        out = F.pad(out, (1, 1, 1, 1), value=0)

        diff = (image - out).abs() * 4

        layers = [image, out, diff]
        for i in range(fwd_data[f"layers{self.suffix}"].shape[1]):
            layer = fwd_data[f"layers{self.suffix}"][:, i].cpu().detach()
            rgb, a = th.split(layer, [3, 1], dim=2)

            psz = layer.shape[-1]
            out = th.ones(1, 3, psz // 8, 8, psz // 8, 8)
            out[:, :, ::2, :, ::2] = 0.8
            out[:, :, 1::2, :, 1::2] = 0.8
            out = out.flatten(2, 3).flatten(3, 4)

            for i in range(4):
                out = out * (1 - a[:, i]) + rgb[:, i] * a[:, i]
            layers.append(F.pad(out, (1, 1, 1, 1), value=0))

        viz = th.cat(layers, 3)
        return th.clamp(viz, 0, 1)


class DictCallbackPlayer(cb.TensorBoardImageDisplayCallback):
    def tag(self):
        return "player_dict"

    def visualized_image(self, batch, fwd_data):
        dictionary = fwd_data["dict_player"].cpu()
        dict_codes = fwd_data["dict_codes_player"].cpu().detach()

        tsne = TSNE(n_components=1, metric="cosine", square_distances=True)
        xf_codes = tsne.fit_transform(dict_codes).squeeze()
        dictionary = dictionary[xf_codes.argsort()]

        color = dictionary[:, :-1]
        alpha = dictionary[:, -1:]

        # Checkerboard background for tranparency
        psz = dictionary.shape[-1]
        bg = th.ones(1, 3, psz // 8, 8, psz // 8, 8)
        bg[:, :, ::2, :, ::2] = 0.8
        bg[:, :, 1::2, :, 1::2] = 0.8
        bg = bg.flatten(2, 3).flatten(3, 4)

        return th.clamp(bg * (1 - alpha) + color * alpha, 0, 1)


class DictCallbackNonPlayer(cb.TensorBoardImageDisplayCallback):
    def tag(self):
        return "non_player_dict"

    def visualized_image(self, batch, fwd_data):
        dictionary_n = fwd_data["dict_non_player"].cpu()
        dict_codes_n = fwd_data["dict_codes_non_player"].cpu().detach()

        tsne = TSNE(n_components=1, metric="cosine", square_distances=True)
        xf_codes = tsne.fit_transform(dict_codes_n).squeeze()
        dictionary_n = dictionary_n[xf_codes.argsort()]

        color = dictionary_n[:, :-1]
        alpha = dictionary_n[:, -1:]

        # Checkerboard background for tranparency
        psz = dictionary_n.shape[-1]
        bg = th.ones(1, 3, psz // 8, 8, psz // 8, 8)
        bg[:, :, ::2, :, ::2] = 0.8
        bg[:, :, 1::2, :, 1::2] = 0.8
        bg = bg.flatten(2, 3).flatten(3, 4)

        return th.clamp(bg * (1 - alpha) + color * alpha, 0, 1)


class BackgroundCallback(cb.TensorBoardImageDisplayCallback):
    def tag(self):
        return "background"

    def visualized_image(self, batch, fwd_data):
        bg = fwd_data["background"].cpu()
        return bg
