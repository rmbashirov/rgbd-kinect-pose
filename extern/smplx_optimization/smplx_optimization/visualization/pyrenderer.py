import numpy as np
import trimesh
import pyrender
from pyrender.constants import RenderFlags
from pyrender.light import DirectionalLight
from pyrender.node import Node
import cv2
from copy import deepcopy

import os
os.environ["PYOPENGL_PLATFORM"] = "egl"


def get_mesh(verts, faces):
    vert_colors = np.tile([128, 128, 128], (verts.shape[0], 1))
    mesh = trimesh.Trimesh(
        vertices=verts,
        faces=faces,
        vertex_colors=vert_colors,
        process=False
    )
    return mesh


def get_cube(ps, s=0.15):
    diffs = np.array([
        [-1, -1, -1, -1, 1, 1, 1, 1],
        [-1, -1, 1, 1, -1, -1, 1, 1],
        [-1, 1, -1, 1, -1, 1, -1, 1]
    ], dtype=np.float32) * s

    ps = ps.reshape(-1, 3)
    result = []
    for p in ps:
        result.append((diffs + p.reshape(3, -1)).T)
    result = np.concatenate(result)
    return result


def get_bbox(points):
    left = np.min(points[:, 0])
    right = np.max(points[:, 0])
    top = np.min(points[:, 1])
    bottom = np.max(points[:, 1])
    h = bottom - top
    w = right - left
    if h > w:
        cx = (left + right) / 2
        left = cx - h / 2
        right = left + h
    else:
        cy = (bottom + top) / 2
        top = cy - w / 2
        bottom = top + w
    return left, top, right, bottom


class Pyrenderer:
    def __init__(self, is_shading=True, d_light=3., scale=None):
        self.is_shading = is_shading
        self.light = (.3, .3, .3) if self.is_shading else (1., 1., 1.)

        self.scene = pyrender.Scene(bg_color=[255, 255, 255], ambient_light=self.light)

        self.size = None
        self.viewer = None

        self.T = None
        self.K_no_scale = None
        self.K = None
        self.camera = None
        self.camera_node = None

        self.d_light = d_light
        self.light_nodes = None

        self.scale = scale

    def add_raymond_light(self, s=1, d=0.25, T=np.eye(4)):
        thetas = np.pi * np.array([1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0])
        phis = np.pi * np.array([0.0, 2.0 / 3.0, 4.0 / 3.0])
        nodes = []
        for phi, theta in zip(phis, thetas):
            xp = s * np.sin(theta) * np.cos(phi)
            yp = s * np.sin(theta) * np.sin(phi)
            zp = s * np.cos(theta)

            z = np.array([xp, yp, zp])
            z = z / np.linalg.norm(z)
            x = np.array([-z[1], z[0], 0.0])
            if np.linalg.norm(x) == 0:
                x = np.array([1.0, 0.0, 0.0])
            x = x / np.linalg.norm(x)
            y = np.cross(z, x)

            matrix = np.eye(4)
            matrix[:3, :3] = np.c_[x, y, z]

            Tl = np.copy(T)

            matrix = matrix @ Tl

            nodes.append(Node(
                light=DirectionalLight(color=np.ones(3), intensity=1.0 / d),
                matrix=matrix
            ))
        return nodes

    def use_raymond_lighting(self, intensity=5.0, d=3.0, T=np.eye(4)):
        if self.light_nodes is not None:
            for n in self.light_nodes:
                self.scene.remove_node(n)

        self.light_nodes = self.add_raymond_light(T=T)
        for n in self.light_nodes:
            n.light.intensity = intensity / d
            self.scene.add_node(n)

    def set_intrinsics(self, size, K, scale=None, znear=0.1, zfar=10, rvec=None):
        self.K_no_scale = deepcopy(K)
        self.K = deepcopy(K)
        if scale is not None:
            self.K[:2] *= scale
        self.camera = pyrender.IntrinsicsCamera(
            fx=self.K[0, 0], fy=self.K[1, 1],
            cx=self.K[0, 2], cy=self.K[1, 2],
            znear=znear, zfar=zfar
        )
        if self.camera_node is not None:
            self.scene.remove_node(self.camera_node)
        self.T = np.eye(4)
        if rvec is not None:
            rotmtx = cv2.Rodrigues(np.array(rvec))[0]
            self.T[:3, :3] = rotmtx
        self.camera_node = self.scene.add(self.camera, pose=self.T, name='pc-camera')

        self.size = deepcopy(size)
        if scale is not None:
            for k in self.size:
                self.size[k] *= scale
        self.viewer = pyrender.OffscreenRenderer(self.size['w'], self.size['h'])

    def project(self, verts, scale=True):
        K = self.K if scale else self.K_no_scale
        verts_2d = verts @ K.T
        verts_2d = verts_2d[:, :2] / verts_2d[:, [2]]
        return verts_2d

    def get_bbox(self, verts, verts_filter=None, scale=True, s=.15):
        K = self.K if scale else self.K_no_scale
        # verts = verts @ self.T[:3, :3].T
        if verts_filter is not None:
            verts = verts[verts_filter]
        verts = get_cube(verts, s=s)
        verts_2d = verts @ K.T
        verts_2d = verts_2d[:, :2] / verts_2d[:, [2]]
        bbox = get_bbox(verts_2d)
        return bbox

    def render(self, verts, faces):
        for node in self.scene.get_nodes():
            if node.name is not None and 'dynamic-mesh' in node.name:
                self.scene.remove_node(node)
        mesh = get_mesh(verts, faces)
        mesh = pyrender.Mesh.from_trimesh(mesh)
        self.scene.add(mesh, 'dynamic-mesh-0')

        if self.is_shading:
            self.use_raymond_lighting(T=self.T, d=self.d_light)

        flags = RenderFlags.RGBA
        if self.is_shading:
            flags = RenderFlags.SHADOWS_DIRECTIONAL | RenderFlags.RGBA
        else:
            flags = RenderFlags.FLAT | flags

        color_img, depth_img = self.viewer.render(self.scene, flags=flags)

        return color_img