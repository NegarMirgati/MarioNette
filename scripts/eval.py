"""Train script."""

import os
import logging

import torch as th
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from torchinfo import summary
from PIL import Image

import ttools

from marionet import datasets, models, callbacks
from marionet.interfaces import Interface

LOG = logging.getLogger(__name__)

th.backends.cudnn.deterministic = True


def _worker_init_fn(_):
    np.random.seed()


def _set_seed(seed):
    np.random.seed(seed)
    th.manual_seed(seed)
    th.cuda.manual_seed(seed)


def image_grid(imgs, rows, cols):
    assert len(imgs) == rows * cols

    w, h = imgs[0].size
    grid = Image.new("RGBA", size=(cols * w, rows * h))
    grid_w, grid_h = grid.size

    for i, img in enumerate(imgs):
        grid.paste(img, box=(i % cols * w, i // cols * h))
    return grid


def main(args):
    LOG.info(f"Using seed {args.seed}.")

    device = "cuda" if th.cuda.is_available() and args.cuda else "cpu"

    # learned_dict = models.Dictionary(args.num_classes,
    #                                  (args.canvas_size // args.layer_size*2,
    #                                   args.canvas_size // args.layer_size*2),
    #                                  4, bottleneck_size=args.dim_z)
    learned_dict_player = models.Dictionary(
        args.num_classes,
        (
            args.canvas_size // args.layer_size * 2,
            args.canvas_size // args.layer_size * 2,
        ),
        4,
        bottleneck_size=args.dim_z,
    )
    learned_dict_player.to(device)
    learned_dict_non_player = models.Dictionary(
        args.num_classes,
        (
            args.canvas_size // args.layer_size * 2,
            args.canvas_size // args.layer_size * 2,
        ),
        4,
        bottleneck_size=args.dim_z,
    )
    learned_dict_non_player.to(device)

    model = models.Model(
        learned_dict_player, learned_dict_non_player, args.layer_size, args.num_layers
    )
    model.eval()

    model_checkpointer = ttools.Checkpointer(
        os.path.join(args.checkpoint_dir, "model"), model
    )
    model_checkpointer.load_latest()

    # with th.no_grad():
    #     fwd_data = model(im, None, hard=True)

    learned_dict_player, dict_codes_player = model.learned_dict_player()
    learned_dict_non_player, dict_codes_non_player = model.learned_dict_non_player()
    print(f"model:{model}")
    print(f"model_summary:")
    summary(model)
    print(f"learned_dict player:{learned_dict_player.shape}")
    print(f"learned_dict non_player:{learned_dict_non_player.shape}")
    print(f"dict_codes player:{dict_codes_player.shape}")
    print(f"dict_codes non_player:{dict_codes_non_player.shape}")

    imgs = []

    for idx in range(learned_dict_player.shape[0]):
        t = (
            learned_dict_player[idx].permute(1, 2, 0).detach().cpu().numpy() * 256
        ).astype(np.uint8)
        img = Image.fromarray(t, mode="RGBA")
        img.save(f"./output_images/player/my_image_{idx}.png")

        imgs.append(img)

    imgs = []
    for group in range(len(imgs) // 15):
        image_grid(imgs[group * 15 : (group + 1) * 15], 3, 5).save(
            f"./output_images/player/my_result_grid_{group}.png"
        )

    for idx in range(learned_dict_non_player.shape[0]):
        t = (
            learned_dict_non_player[idx].permute(1, 2, 0).detach().cpu().numpy() * 256
        ).astype(np.uint8)
        img = Image.fromarray(t, mode="RGBA")
        img.save(f"./output_images/non_player/my_image_{idx}.png")

        imgs.append(img)

    for group in range(len(imgs) // 15):
        image_grid(imgs[group * 15 : (group + 1) * 15], 3, 5).save(
            f"./output_images/non_player/my_result_{group}.png"
        )


if __name__ == "__main__":
    parser = ttools.BasicArgumentParser()

    # Representation
    parser.add_argument("--layer_size", type=int, default=8, help="size of anchor grid")
    parser.add_argument("--num_layers", type=int, default=2, help="number of layers")
    parser.add_argument(
        "--num_classes", type=int, default=150, help="size of dictioanry"
    )
    parser.add_argument(
        "--canvas_size", type=int, default=128, help="spatial size of the canvas"
    )

    # Model
    parser.add_argument("--dim_z", type=int, default=128)
    parser.add_argument("--load_model", type=str)
    parser.add_argument("--load_bg", type=str)
    parser.add_argument("--no_layernorm", action="store_true", default=False)

    # Training options
    parser.add_argument("--seed", type=int, default=2)
    parser.add_argument("--num_steps", type=int, default=200000)
    parser.add_argument("--w_beta", type=float, default=0.002)
    parser.add_argument("--w_probs", type=float, nargs="+", default=[5e-3])
    parser.add_argument("--lr_bg", type=float, default=1e-3)
    parser.add_argument("--shuffle_all", action="store_true", default=False)
    parser.add_argument("--crop", action="store_true", default=False)
    parser.add_argument("--background", action="store_true", default=False)
    parser.add_argument("--sprites", action="store_true", default=False)
    parser.add_argument("--no_spatial_transformer", action="store_true", default=False)
    parser.add_argument("--spatial_transformer_bg", action="store_true", default=False)
    parser.add_argument("--straight_through_probs", action="store_true", default=False)

    args = parser.parse_args()
    ttools.set_logger(args.debug)
    main(args)
