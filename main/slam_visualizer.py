import os

import cv2
import matplotlib as mpl
import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from moviepy.editor import ImageSequenceClip


def read_video_from_path(path):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise IOError(f"Error opening video file: {path}")  # Improved error handling with exception
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            frames.append(np.array(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
        else:
            break
    cap.release()
    return np.stack(frames)


class SLAMVisualizer:
    def __init__(self, cfg, save_dir=None):
        self.cfg = cfg.visualizer
        self.cfg_full = cfg
        self.mode = self.cfg.mode
        self.save_dir = save_dir or self.cfg.save_dir  # Prefer explicit arg over config default

        self.color_map = mpl.colormaps.get(self.cfg.mode, mpl.colormaps["cool"])  # Fallback color map if mode not found

        self.show_first_frame = self.cfg.show_first_frame
        self.grayscale = self.cfg.grayscale
        self.tracks_leave_trace = self.cfg.tracks_leave_trace
        self.pad_value = self.cfg.pad_value
        self.linewidth = self.cfg.linewidth
        self.fps = self.cfg.fps

        self.frames = []
        self.tracks = []

    def add_frame(self, frame):
        self.frames.append(frame)

    def add_track(self, track):
        self.tracks.append(track)

    def draw_tracks_on_frames(self):
        if not self.frames:
            raise RuntimeError("No frames to visualize")  # Ensure safe drawing

        video = torch.stack(self.frames, dim=0)
        video = F.pad(
            video,
            (self.pad_value, self.pad_value, self.pad_value, self.pad_value),
            "constant",
            255,
        )
        video = video.permute(0, 2, 3, 1).detach().cpu().numpy()

        res_video_sta = []
        res_video_dyn = []

        for rgb in video:
            res_video_sta.append(rgb.copy())
            res_video_dyn.append(rgb.copy())

        T = self.fps * 2

        for t, track in enumerate(self.tracks):
            targets = track["targets"][0].long().detach().cpu().numpy() + self.pad_value  # Avoid offset bugs
            S, N, _ = targets.shape

            vis_label = track.get("vis_label")
            static_label = track.get("static_label")
            coords_vars = track.get("coords_vars")

            vis_label = vis_label[0].detach().cpu().numpy() if vis_label is not None else None
            static_label = static_label[0].detach().cpu().numpy() if static_label is not None else None
            coords_vars = coords_vars[0].detach().cpu().numpy() if coords_vars is not None else None

            for s in range(S):
                color = (
                    np.array(self.color_map(((t - S + 1 + s) % T) / T)[:3])[None] * 255
                )
                vector_colors = np.repeat(color, N, axis=0)

                for n in range(N):
                    coord = (targets[s, n, 0], targets[s, n, 1])
                    visibile = vis_label[s, n] if vis_label is not None else True
                    static = static_label[s, n] if static_label is not None else True
                    conf_scale = (
                        4 - 3 * np.exp(-coords_vars[s, n]) if coords_vars is not None else 1.0
                    )

                    if coord[0] > 0 and coord[1] > 0:  # Skip zero or invalid coordinates
                        radius = int(self.linewidth * 2)
                        circle_target = res_video_sta if static else res_video_dyn

                        cv2.circle(
                            circle_target[t],
                            coord,
                            radius,
                            vector_colors[n].tolist(),
                            thickness=-1 if visibile else 1,
                        )
                        cv2.circle(
                            circle_target[t],
                            coord,
                            int(radius * conf_scale * 3),
                            vector_colors[n].tolist(),
                            1,
                        )

        res_video = [np.concatenate([res_video_sta[i], res_video_dyn[i]], axis=0) for i in range(len(video))]

        if self.show_first_frame > 0:
            res_video = [res_video[0]] * self.show_first_frame + res_video[1:]
        return torch.from_numpy(np.stack(res_video)).permute(0, 3, 1, 2)[None].byte()

    def save_video(self, filename, writer=None, step=0):
        video = self.draw_tracks_on_frames()

        if writer:
            writer.add_video(
                f"{filename}_pred_track",
                video.to(torch.uint8),
                global_step=step,
                fps=self.fps,
            )
        else:
            os.makedirs(self.save_dir, exist_ok=True)
            wide_list = [frame[0].permute(1, 2, 0).cpu().numpy() for frame in video.unbind(1)]
            clip = ImageSequenceClip(wide_list, fps=self.fps)  # Removed [2:-1] to save entire sequence

            save_path = os.path.join(self.save_dir, f"{filename}_pred_track.mp4")
            clip.write_videofile(save_path, codec="libx264", fps=self.fps, logger=None)

            print(f"Video saved to {save_path}")