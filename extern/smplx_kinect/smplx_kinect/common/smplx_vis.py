import numpy as np
import matplotlib
from copy import deepcopy
import cv2
import matplotlib.pyplot as plt

import minimal_pytorch_rasterizer as mpr
import torch

from smplx_kinect.common.body_models import SMPLXWrapper


def get_vlims(z, vmin=None, vmax=None, v_pad=.5, percentile=1, verbose_vlims=False):
    if vmin is None:
        if percentile > 0:
            res_vmin = np.percentile(z[z > 0], percentile)
        else:
            res_vmin = z[z > 0].min()
    else:
        res_vmin = vmin
    if vmax is None:
        if percentile > 0:
            res_vmax = np.percentile(z[z > 0], 100 - percentile)
        else:
            res_vmax = z.max()
    else:
        res_vmax = vmax

    v_pad = (res_vmax - res_vmin) * v_pad
    if vmin is None:
        res_vmin -= v_pad
    if vmax is None:
        res_vmax += v_pad

    if verbose_vlims:
        print(res_vmin, res_vmax)

    return res_vmin, res_vmax


def colormap_z(z, vmin=None, vmax=None, verbose_vlims=False, inverse_cmap=True, v_pad=.5, percentile=1):
    vmin, vmax = get_vlims(z, vmin=vmin, vmax=vmax, v_pad=v_pad, percentile=percentile, verbose_vlims=verbose_vlims)

    z_vis = deepcopy(z)
    if inverse_cmap:
        z_vis[z == 0] = vmax
        cm = matplotlib.cm.viridis_r
    else:
        z_vis[z == 0] = vmin
        cm = matplotlib.cm.viridis

    z_vis = (z_vis - vmin) / (vmax - vmin)

    colored = cm(z_vis)[..., :3]

    return colored


class VisualizerMeshSMPLX:
    def __init__(
        self,
        device=None,
        body_models_dp=None,
        size=1024,
        f_scale=2,
        z=5,
        normals=True,
        sides=False,
        smplx_wrapper=None,
        kinect_bones=None,
        K=None,
        scale=None
    ):
        self.size = size
        self.f_scale = f_scale
        self.z = z
        self.normals = normals
        self.sides = sides
        self.no_dist = False
        if K is not None:
            assert isinstance(size, dict)
            if scale is None:
                scale = 1
            self.pinhole2d = mpr.Pinhole2D(
                fx=scale * K[0, 0], fy=scale * K[1, 1],
                cx=scale * K[0, 2], cy=scale * K[1, 2],
                h=int(scale * size['h']), w=int(scale * size['w'])
            )
            self.no_dist = True
        else:
            self.pinhole2d = mpr.Pinhole2D(
                fx=self.f_scale * self.size, fy=self.f_scale * self.size,
                cx=self.size // 2, cy=self.size // 2,
                h=self.size, w=self.size
            )

        if smplx_wrapper is not None:
            self.smplx_wrapper = smplx_wrapper
        else:
            assert body_models_dp is not None
            assert device is not None
            self.smplx_wrapper = SMPLXWrapper(body_models_dp, device)
        self.device = torch.device(device)
        self.faces = self.smplx_wrapper.models['male'].faces_tensor.to(
            dtype=torch.int32,
            device=self.device
        )
        self.kinect_bones = kinect_bones

    def vertices2vis(self, vertices):
        assert vertices.device == self.device

        if self.sides:
            n, m = 3, 3
            coords = [
                ([0, 0, 0], -0.5, (1, 1)),
                ([0, np.pi / 3, 0], -0.5, (1, 2)),
                ([0, -np.pi / 3, 0], -0.5, (1, 0)),
                ([np.pi / 3, 0, 0], -0.3, (0, 1)),
                ([-np.pi / 3, 0, 0], -0.3, (2, 1)),
            ]
        else:
            n, m = 1, 1
            coords = [
                ([0, 0, 0], -0.5, (0, 0)),
            ]

        h, w = self.pinhole2d.h * n, self.pinhole2d.w * m
        result = np.zeros((h, w, 3), dtype=np.uint8)

        for rvec, t_up, (ax_i, ax_j) in coords:
            if self.no_dist:
                vertices_transformed = vertices
            else:
                R = torch.tensor(
                    cv2.Rodrigues(np.array(rvec, dtype=np.float32))[0],
                    dtype=torch.float32, device=self.device
                )
                t = torch.tensor([[0, -t_up, self.z]], dtype=torch.float32, device=self.device)

                vertices_transformed = vertices @ R.T + t

            if self.normals:
                coords, normals = mpr.estimate_normals(
                    vertices=vertices_transformed,
                    faces=self.faces,
                    pinhole=self.pinhole2d
                )
                vis = mpr.vis_normals(coords, normals)
                vis = cv2.merge((vis, vis, vis))  # convert gray to 3 channel img
            else:
                z_buffer = mpr.project_mesh(
                    vertices=vertices_transformed,
                    faces=self.faces,
                    vertice_values=vertices_transformed[:, [2]],
                    pinhole=self.pinhole2d
                )
                z_buffer = z_buffer[:, :, 0].cpu().numpy()
                vis = colormap_z(z_buffer, percentile=1)
                vis = (vis * 255).round().clip(0, 255).astype(np.uint8)[..., :3]

            result[
                ax_i * self.pinhole2d.h: (ax_i + 1) * self.pinhole2d.h,
                ax_j * self.pinhole2d.w: (ax_j + 1) * self.pinhole2d.w
            ] = vis

        return result

    def get_vis(
            self,
            body_pose, gender='male',
            betas=np.zeros(10),
            rvec=np.array((np.pi, 0, 0), dtype=np.float32),
            tvec=np.zeros(3, dtype=np.float32)):
        model_output = self.smplx_wrapper.get_output(
            gender=gender,
            betas=betas,
            body_pose=body_pose,
            rvec=rvec,
            tvec=tvec
        )
        vertices = model_output.vertices.detach()[0].contiguous()
        return self.vertices2vis(vertices)

    def vis_kinect_skel(self,
        joints, tmp_fp,
        show_1=True, show_2=True, title=None
    ):
        x = joints[:, 0]
        y = joints[:, 1]
        z = joints[:, 2]

        fix, axs = plt.subplots(1, 2, figsize=(30, 15))

        if show_1:
            axs[0].scatter(x, y, c='k')
            axs[0].set_aspect('equal')

            # for i in range(len(x)):
            #     axs[0].text(x[i], y[i], f'{i}', fontsize=20)

            for a, b in self.kinect_bones:
                axs[0].plot(x[[a, b]], y[[a, b]], 'c-')

            axs[0].invert_yaxis()

        if show_2:
            axs[1].scatter(x, z, c='k')
            axs[1].set_aspect('equal')

            # for i in range(len(x)):
            #     axs[1].text(x[i], z[i], f'{i}', fontsize=20)

            for a, b in self.kinect_bones:
                axs[1].plot(x[[a, b]], z[[a, b]], 'c-')

        if title is not None:
            plt.title(title)
        plt.savefig(tmp_fp)
        plt.close()

        img = cv2.imread(tmp_fp)[..., [2, 1, 0]]
        return img
