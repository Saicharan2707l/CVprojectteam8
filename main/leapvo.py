import os
import logging
import time

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange, repeat

from main.backend import altcorr, lietorch
from main.backend import projective_ops as pops
from main.backend.ba import BA
from main.backend.lietorch import SE3
from main.leap.leap_kernel import LeapKernel
from main.slam_visualizer import LEAPVisualizer

# Configure logger
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


class LEAPVO:
    def __init__(self, cfg, ht=480, wd=640):
        
        self.enable_timing = False  # <-- you can toggle this to True when needed
        

    def update(self):
        if self.enable_timing:
            start_time = time.time()

        lmbda = torch.as_tensor([1e-4], device="cuda")
        t0 = self.n - self.cfg.slam.OPTIMIZATION_WINDOW if self.is_initialized else 1
        t0 = max(t0, 1)

        ep = 10
        lmbda = 1e-4
        bounds = [0, 0, self.wd, self.ht]
        Gs = SE3(self.poses)
        patches = self.patches

        for itr in range(self.cfg.slam.ITER):
            Gs, patches = BA(
                Gs,
                patches,
                self.intrinsics.detach(),
                self.targets.detach(),
                self.weights.detach(),
                lmbda,
                self.ii,
                self.jj,
                self.kk,
                bounds,
                ep=ep,
                fixedp=t0,
                structure_only=False,
                loss=self.cfg.slam.LOSS,
            )

        self.patches_[:] = patches.reshape(self.N, self.M, 3, self.P, self.P)
        self.poses_[:] = Gs.vec().reshape(self.N, 7)

        if self.cfg.slam.USE_MAP_FILTERING:
            with torch.no_grad():
                self.map_point_filtering()

        points = pops.point_cloud(
            SE3(self.poses),
            self.patches[:, : self.m],
            self.intrinsics,
            self.ix[: self.m],
        )
        points = (
            points[..., self.P // 2, self.P // 2, :3]
            / points[..., self.P // 2, self.P // 2, 3:]
        ).reshape(-1, 3)
        self.points_[: len(points)] = points[:]

        if self.enable_timing:
            elapsed = time.time() - start_time
            logger.info(f"Update step completed in {elapsed:.3f} seconds.")

    def __call__(self, tstamp, image, intrinsics):
        if (self.n + 1) >= self.N:
            logger.error(f'Buffer overflow: {self.n+1} >= {self.N}')
            raise Exception(
                f'The buffer size is too small. You can increase it using "--buffer {self.N*2}"'
            )

        if self.viewer is not None:
            self.viewer.update_image(image)
        if self.visualizer is not None:
            self.visualizer.add_frame(image)

        self.preprocess(image, intrinsics)
        patches, clr = self.generate_patches(image)

        patches[:, :, 2] = torch.rand_like(patches[:, :, 2, 0, 0, None, None])
        if self.is_initialized:
            s = torch.median(self.patches_[self.n - 3 : self.n, :, 2])
            patches[:, :, 2] = s

        self.patches_[self.n] = patches

        if self.n % self.kf_stride == 0 and not self.is_initialized:
            self.patches_valid_[self.n] = 1

        self.init_motion()

        self.tlist.append(tstamp)
        self.tstamps_[self.n] = self.counter
        clr = clr[0]
        self.colors_[self.n] = clr.to(torch.uint8)
        self.index_[self.n] = self.n
        self.index_map_[self.n] = self.m
        self.counter += 1
        self.n += 1
        self.m += self.M

        if (self.n - 1) % self.kf_stride == 0:
            self.append_factors(*self.__edges())
            self.predict_target()

        if self.n == self.cfg.slam.num_init and not self.is_initialized:
            self.is_initialized = True
            for itr in range(12):
                self.update()

        elif self.is_initialized:
            self.update()
            self.keyframe()

        torch.cuda.empty_cache()

    def predict_target(self):
        with torch.no_grad():
            (
                trajs,
                vis_label,
                queries,
                stats,
            ) = self.get_window_trajs()

        self.last_target = trajs
        self.last_valid = vis_label
        B, S, N, C = trajs.shape
        local_target = rearrange(trajs, "b s n c -> b (n s) c")

        local_weight = torch.ones_like(local_target)
        vis_label = rearrange(vis_label, "b s n -> b (n s)")
        local_weight[~vis_label] = 0

        padding = 20
        boundary_mask = (
            (local_target[..., 0] >= padding)
            & (local_target[..., 0] < self.wd - padding)
            & (local_target[..., 1] >= padding)
            & (local_target[..., 1] < self.ht - padding)
        )
        local_weight[~boundary_mask] = 0

        if self.n >= self.cfg.slam.MIN_TRACK_LEN:
            patch_valid = (local_weight > 0).any(dim=-1)
            patch_valid = rearrange(patch_valid, "b (n s) -> b s n", s=S, n=N)
            patch_valid = patch_valid.sum(dim=1) >= self.cfg.slam.MIN_TRACK_LEN
            self.patches_valid_[self.n - S : self.n : self.kf_stride] = (
                patch_valid.reshape(-1, self.M)
            )
            track_len_mask = repeat(patch_valid, "b n -> b (n s)", s=S)
            local_weight[~track_len_mask] = 0

            if self.enable_timing:
                valid_count = patch_valid.sum().item()
                logger.info(f"Valid patches at frame {self.n}: {valid_count} / {patch_valid.numel()}")

        self.targets = torch.cat([self.targets, local_target], dim=1)
        self.weights = torch.cat([self.weights, local_weight], dim=1)

        local_target_ = rearrange(
            local_target, "b (s1 m s) c -> b s s1 m c", s=S, m=self.M
        )
        local_weight_ = rearrange(
            local_weight, "b (s1 m s) c -> b s s1 m c", s=S, m=self.M
        )

        vis_data = {
            "fid": self.n,
            "targets": local_target_,
            "weights": local_weight_,
            "queries": queries,
        }
        for key, value in stats.items():
            if value is not None:
                vis_data[key] = value

        self.visualizer.add_track(vis_data)

