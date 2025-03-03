import os
import sys
import math
import pickle as pkl
import shutil
import argparse
from pathlib import Path

import cv2
import wget
import numpy as np
import torch
import scipy.special
import matplotlib.pyplot as plt
from beartype import beartype
from zsvision.zs_utils import BlockTimer
from tqdm import tqdm

sys.path.append("..")
import models
from utils.misc import to_torch
from utils.imutils import im_to_numpy, im_to_torch, resize_generic
from utils.transforms import color_normalize


@beartype
def load_rgb_video(video_path: Path, fps: int) -> torch.Tensor:
    """
    Load frames of a video using cv2 (fetch from provided URL if file is not found
    at given location).
    """
    cap = cv2.VideoCapture(str(video_path))
    cap_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    cap_fps = cap.get(cv2.CAP_PROP_FPS)

    # cv2 won't be able to change frame rates for all encodings, so we use ffmpeg
    if cap_fps != fps:
        tmp_video_path = f"{video_path}.tmp.{video_path.suffix}"
        shutil.move(video_path, tmp_video_path)
        cmd = (f"ffmpeg -i {tmp_video_path} -pix_fmt yuv420p "
               f"-filter:v fps=fps={fps} {video_path}")
        print(f"Generating new copy of video with frame rate {fps}")
        os.system(cmd)
        Path(tmp_video_path).unlink()
        cap = cv2.VideoCapture(str(video_path))
        cap_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        cap_fps = cap.get(cv2.CAP_PROP_FPS)
        assert cap_fps == fps, f"ffmpeg failed to produce a video at {fps}"

    f = 0
    rgb = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = frame[:, :, [2, 1, 0]]
        rgb_t = im_to_torch(frame)
        rgb.append(rgb_t)
        f += 1
    cap.release()
    rgb = torch.stack(rgb).permute(1, 0, 2, 3)
    return rgb


@beartype
def prepare_input(
    rgb: torch.Tensor,
    resize_res: int = 256,
    inp_res: int = 224,
    mean: torch.Tensor = 0.5 * torch.ones(3), std=1.0 * torch.ones(3),
):
    """
    Process the video:
    1) Resize to [resize_res x resize_res]
    2) Center crop with [inp_res x inp_res]
    3) Color normalize using mean/std
    """
    iC, iF, iH, iW = rgb.shape
    rgb_resized = np.zeros((iF, resize_res, resize_res, iC))
    for t in range(iF):
        tmp = rgb[:, t, :, :]
        tmp = resize_generic(
            im_to_numpy(tmp), resize_res, resize_res, interp="bilinear", is_flow=False
        )
        rgb_resized[t] = tmp
    rgb = np.transpose(rgb_resized, (3, 0, 1, 2))
    # Center crop coords
    ulx = int((resize_res - inp_res) / 2)
    uly = int((resize_res - inp_res) / 2)
    # Crop 256x256
    rgb = rgb[:, :, uly : uly + inp_res, ulx : ulx + inp_res]
    rgb = to_torch(rgb).float()
    assert rgb.max() <= 1
    rgb = color_normalize(rgb, mean, std)
    return rgb

@beartype
def load_model(
        checkpoint_path: Path,
        num_classes: int,
        num_in_frames: int,
) -> torch.nn.Module:
    model = models.InceptionI3d(
        num_classes=num_classes,
        spatiotemporal_squeeze=True,
        final_endpoint="Logits",
        name="inception_i3d",
        in_channels=3,
        dropout_keep_prob=0.5,
        num_in_frames=num_in_frames,
    )
    model = torch.nn.DataParallel(model).cuda()
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model


@beartype
def load_vocab(
        word_data_pkl_path: Path,
) -> dict:
    return pkl.load(open(word_data_pkl_path, "rb"))


@beartype
def sliding_windows(
        rgb: torch.Tensor,
        num_in_frames: int,
        stride: int,
) -> tuple:
    """
    Return sliding windows and corresponding (middle) timestamp
    """
    C, nFrames, H, W = rgb.shape
    # If needed, pad to the minimum clip length
    if nFrames < num_in_frames:
        rgb_ = torch.zeros(C, num_in_frames, H, W)
        rgb_[:, :nFrames] = rgb
        rgb_[:, nFrames:] = rgb[:, -1].unsqueeze(1)
        rgb = rgb_
        nFrames = rgb.shape[1]

    num_clips = math.ceil((nFrames - num_in_frames) / stride) + 1
    plural = ""
    if num_clips > 1:
        plural = "s"
    print(f"{num_clips} clip{plural} resulted from sliding window processing.")

    rgb_slided = torch.zeros(num_clips, 3, num_in_frames, H, W)
    t_mid = []
    # For each clip
    for j in range(num_clips):
        # Check if num_clips becomes 0
        actual_clip_length = min(num_in_frames, nFrames - j * stride)
        if actual_clip_length == num_in_frames:
            t_beg = j * stride
        else:
            t_beg = nFrames - num_in_frames
        t_mid.append(t_beg + num_in_frames / 2)
        rgb_slided[j] = rgb[:, t_beg : t_beg + num_in_frames, :, :]
    return rgb_slided, np.array(t_mid)


@beartype
def viz_predictions(
        rgb: torch.Tensor,
        word_topk: np.ndarray,
        prob_topk: np.ndarray,
        t_mid: np.ndarray,
        frame_dir: Path,
        confidence: float,
        gt_text: str,
):
    if gt_text != "":
        # Put linebreaks for long strings every 40 chars
        gt_text = list(gt_text)
        max_num_chars_per_line = 40
        num_linebreaks = int(len(gt_text) / max_num_chars_per_line)
        for lb in range(num_linebreaks):
            pos = (lb + 1) * max_num_chars_per_line
            gt_text.insert(pos, "\n")
        gt_text = "".join(gt_text)
        gt_text = f"GT: {gt_text}"
    print(f"Saving visualizations to {frame_dir}")
    num_frames = rgb.shape[1]
    height = rgb.shape[2]
    offset = height / 14
    vertical_sep = offset * 2
    for t in tqdm(range(num_frames)):
        t_ix = abs(t_mid - t).argmin()
        sign = word_topk[:, t_ix]
        sign_prob = prob_topk[:, t_ix]
        plt.imshow(im_to_numpy(rgb[:, t]))
        for k, s in enumerate(sign):
            if sign_prob[k] >= confidence:
                pred_text = f"Prediction: {s} \nAccuracy: {100 * sign_prob[k]:.0f}%"
                plt.text(
                    80,
                    230,
                    pred_text,
                    fontsize=14,
                    fontfamily='sans-serif',
                    fontweight="bold",
                    color="white",
                    verticalalignment="top",
                    bbox=dict(facecolor="black", alpha=0.6, boxstyle='round,pad=0.5'),
                )
        plt.axis("off")
        plt.savefig(frame_dir / f"frame_{t:03d}.png")
        plt.clf()


def main(
    video_path: Path,
    save_path: Path,
):
    checkpoint_path = Path("./weights/model.pth.tar")
    word_data_pkl_path = Path("./weights/info.pkl")
    gt_text = ""
    num_classes = 2000
    gen_gif = bool(0)
    num_in_frames = 64
    confidence = 0.2
    batch_size = 10
    stride = 8
    topk = 1
    viz = bool(1)
    fps = 25


    with BlockTimer("Loading model"):
        model = load_model(
            checkpoint_path=checkpoint_path,
            num_classes=num_classes,
            num_in_frames=num_in_frames,
        )
    with BlockTimer("Loading mapping to assign class indices to sign glosses"):
        word_data = load_vocab(
            word_data_pkl_path=word_data_pkl_path,
        )
    with BlockTimer("Loading video frames"):
        rgb_orig = load_rgb_video(
            video_path=video_path,
            fps=fps,
        )
    # Prepare: resize/crop/normalize
    rgb_input = prepare_input(rgb_orig)
    # Sliding window
    rgb_slides, t_mid = sliding_windows(
        rgb=rgb_input,
        stride=stride,
        num_in_frames=num_in_frames,
    )
    # Number of windows/clips
    num_clips = rgb_slides.shape[0]
    # Group the clips into batches
    num_batches = math.ceil(num_clips / batch_size)
    raw_scores = np.empty((0, num_classes), dtype=float)
    for b in range(num_batches):
        inp = rgb_slides[b * batch_size : (b + 1) * batch_size]
        # Forward pass
        out = model(inp)
        raw_scores = np.append(raw_scores, out["logits"].cpu().detach().numpy(), axis=0)
    prob_scores = scipy.special.softmax(raw_scores, axis=1)
    prob_sorted = np.sort(prob_scores, axis=1)[:, ::-1]
    pred_sorted = np.argsort(prob_scores, axis=1)[:, ::-1]

    word_topk = np.empty((topk, num_clips), dtype=object)
    for k in range(topk):
        for i, p in enumerate(pred_sorted[:, k]):
            word_topk[k, i] = word_data["words"][p]
    prob_topk = prob_sorted[:, :topk].transpose()
    print("Predicted signs:")
    print(word_topk)
    # Visualize predictions
    if viz:
        save_path.parent.mkdir(exist_ok=True, parents=True)
        frame_dir = save_path.parent / "frames"
        frame_dir.mkdir(exist_ok=True, parents=True)
        # Save visualization images
        viz_predictions(
            rgb=rgb_orig,
            word_topk=word_topk,
            prob_topk=prob_topk,
            t_mid=t_mid,
            frame_dir=frame_dir,
            confidence=confidence,
            gt_text=gt_text,
        )
        # Make a video from the visualization images
        ffmpeg_str = (f"ffmpeg -y -i {frame_dir}/frame_%03d.png -c:v libx264 "
                      f"-pix_fmt yuv420p -filter:v fps=fps=25 {save_path}")
        os.system(ffmpeg_str)
        # Remove the visualization images
        shutil.rmtree(str(frame_dir))

        if gen_gif:
            gif_path = save_path.with_suffix(".gif")
            cmd = f"ffmpeg -y -i {save_path} -f gif {gif_path}"
            print(f"Generating gif of output at {gif_path}")
            os.system(cmd)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Helper script to run demo.")
    p.add_argument(
        "--video_path",
        type=Path,
        default="sample_data/inputs/book.mp4",
        help="Path to test video.",
    )
    p.add_argument(
        "--save_path",
        type=Path,
        default="sample_data/demo-output.mp4",
        help="Path to save viz (if viz=1).",
    )
    main(**vars(p.parse_args()))
