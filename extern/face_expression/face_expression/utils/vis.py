import os
import numpy as np
import cv2

import torch
import torch.nn.functional as F

os.environ['PYOPENGL_PLATFORM'] = 'egl'
import pyrender
from pyrender.constants import RenderFlags
import trimesh

from matplotlib import pylab as plt

from face_expression import utils


def fig_to_array(fig):
    fig.canvas.draw()
    fig_image = np.array(fig.canvas.renderer._renderer)

    return fig_image


def draw_bbox_in_image(image, bbox, color=(255, 0, 0), thickness=3):
    image = image.copy()
    left, top, right, down = bbox
    
    start_point = (int(left), int(top))
    end_point = (int(right), int(down))
    
    image = cv2.rectangle(image, start_point, end_point, color, thickness=thickness)
    return image


class Renderer:
    def __init__(
            self,
            canvas_shape,
            faces=None,
            bg_color=(255, 0, 0, 0),
            world_angles=(-np.pi, 0.0, 0.0),
            ambient_light=(0.3, 0.3, 0.3),
            directional_light_intensity=2.0
        ):
        
        self.canvas_shape = canvas_shape
        self.faces = faces
        
        # world transform
        world_rotation_matrix = self.rotation_matrix_from_angles(world_angles)
        world_transformation_matrix = np.eye(4)
        world_transformation_matrix[:3, :3] = world_rotation_matrix
        self.world_transformation_matrix = world_transformation_matrix
        
        self.scene = pyrender.Scene(bg_color=tuple(bg_color), ambient_light=tuple(ambient_light))
        self.viewer = pyrender.OffscreenRenderer(*canvas_shape)
        
        # light
        light_nodes = self.build_raymond_light(directional_light_intensity, T=self.world_transformation_matrix)
        for light_node in light_nodes:
            self.scene.add_node(light_node)
    
    def render(
            self,
            verts,
            faces=None,
            camera_matrix=np.eye(3),
            camera_transformation_matrix=None,
            znear=0.1,
            zfar=10.0
        ):
        
        # add mesh to the scene
        if faces is None:
            assert self.faces is not None
            faces = self.faces

        mesh = pyrender.Mesh.from_trimesh(
            trimesh.Trimesh(vertices=verts, faces=faces, process=False)
        )
        
        self._remove_nodes_by_name('mesh')
        mesh_node = self.scene.add(mesh, 'mesh')
        
        # add camera
        if camera_transformation_matrix is None:
            camera_transformation_matrix = self.world_transformation_matrix
        
        camera = pyrender.IntrinsicsCamera(
            fx=camera_matrix[0, 0], fy=camera_matrix[1, 1],
            cx=camera_matrix[0, 2], cy=camera_matrix[1, 2],
            znear=znear, zfar=zfar
        )
        
        self._remove_nodes_by_name('camera')
        camera_node = self.scene.add(camera, pose=camera_transformation_matrix, name='camera')
            
        # render
        flags = RenderFlags.RGBA | RenderFlags.SHADOWS_DIRECTIONAL
        rendered_image_rgba, rendered_image_depth = self.viewer.render(self.scene, flags=flags)
        
        return rendered_image_rgba, rendered_image_depth
    
    def _remove_nodes_by_name(self, name):
        found_nodes = self.scene.get_nodes(name=name)
        for node in found_nodes:
            self.scene.remove_node(node)    
    
    @staticmethod
    def build_raymond_light(intensity, T=np.eye(4)):
        s = 1.0

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

            nodes.append(pyrender.node.Node(
                light=pyrender.light.DirectionalLight(color=np.ones(3), intensity=intensity),
                matrix=matrix
            ))

        return nodes

    @staticmethod
    def rotation_matrix_from_angles(angles):
        cam_rot = np.eye(3)
        
        for i, theta in enumerate(angles):
            rot_row = np.eye(3).reshape(-1)
            small_rot = np.asarray([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
            rot_mask = np.ones((3, 3), dtype=bool)
            rot_mask[i, :] = 0
            rot_mask[:, i] = 0
            rot_row[rot_mask.reshape(-1)] = small_rot.reshape(-1)
            rot = rot_row.reshape(3, 3)
            cam_rot = rot @ cam_rot
            
        return cam_rot


def vis_image_with_smplx(smplx_model, renderer, expression, pose, beta, image, camera_matrix, n_samples, alpha=0.5):
    keypoints_3d, rotation_matrices, verts = utils.misc.infer_smplx(smplx_model, expression, pose, beta)

    batch_size = expression.shape[0]
    n_samples = min(n_samples, batch_size)

    verts_np = verts.detach().cpu().numpy()
    camera_matrix_np = camera_matrix.detach().cpu().numpy()

    canvases = []
    for batch_index in range(n_samples):
        image_np = (255 * image[batch_index].permute(1, 2, 0).detach().cpu().numpy()).astype(np.uint8)

        rendered_image_rgba, rendered_image_depth = renderer.render(verts_np[batch_index], camera_matrix=camera_matrix_np[batch_index])
        canvas = utils.common.blend_2_images_with_alpha(image_np, rendered_image_rgba, alpha=alpha)
        canvases.append(canvas)
    
    return canvases


def vis_image_with_smplx_keypoints_2d(smplx_model, expression, pose, beta, image, projection_matrix, n_samples, color=(255, 0, 0), antialias_factor=2):
    keypoints_2d = utils.misc.infer_smplx_keypoints_2d(smplx_model, expression, pose, beta, projection_matrix)

    # vis 2d keypoints
    canvases = vis_image_with_keypoints_2d(keypoints_2d, image, n_samples, color=color, antialias_factor=antialias_factor)  

    return canvases


def vis_image_with_keypoints_2d(keypoints_2d, image, n_samples, color=(255, 0, 0), antialias_factor=2):
    batch_size = keypoints_2d.shape[0]
    n_samples = min(n_samples, batch_size)

    min_image_size = min(image.shape[2:])

    canvases = []
    for batch_index in range(n_samples):
        keypoints_2d_np = keypoints_2d[batch_index].detach().cpu().numpy()
        keypoints_2d_np *= antialias_factor

        image_np = (255 * image[batch_index].permute(1, 2, 0).detach().cpu().numpy()).astype(np.uint8)
        image_np = utils.common.scale_image(image_np, antialias_factor)

        radius = max(1, int(1 + min_image_size / 512))
        radius = int(antialias_factor * radius)

        canvas = image_np.copy()
        for point in keypoints_2d_np:
            x, y = int(point[0]), int(point[1])
            canvas = cv2.circle(canvas, (x, y), radius, color, thickness=-1, lineType=cv2.LINE_AA)

        canvas = utils.common.scale_image(canvas, 1 / antialias_factor)

        canvases.append(canvas)

    return canvases

def vis_image_with_keypoints_2d_numpy(keypoints_2d, image, n_samples, color=(255, 0, 0), antialias_factor=2):
    keypoints_2d = keypoints_2d.copy()
    image = image.copy()

    min_image_size = min(image.shape[2:])

    keypoints_2d *= antialias_factor
    image = utils.common.scale_image(image, antialias_factor)

    radius = max(1, int(1 + min_image_size / 512))
    radius = int(antialias_factor * radius)

    canvas = image.copy()
    for point in keypoints_2d:
        x, y = int(point[0]), int(point[1])
        canvas = cv2.circle(canvas, (x, y), radius, color, thickness=-1, lineType=cv2.LINE_AA)

    canvas = utils.common.scale_image(canvas, 1 / antialias_factor)

    return canvas


def vis_triple_with_smplx_keypoints_2d(smplx_model, input_dict, output_dict, n_samples):
    # input 2d keypoints
    input_images = vis_image_with_keypoints_2d(
        input_dict['keypoints_orig'][:, :, :2],
        input_dict['image'],
        n_samples,
        color=(255, 0, 0),
    )

    # pred
    pred_images = vis_image_with_smplx_keypoints_2d(
        smplx_model,
        output_dict['expression_pred'],
        output_dict['pose_pred'],
        input_dict['beta'],
        input_dict['image'],
        input_dict['projection_matrix'],
        n_samples,
        color=(0, 0, 255)
    )

    # gt
    gt_images = vis_image_with_smplx_keypoints_2d(
        smplx_model,
        input_dict['expression'],
        input_dict['pose'],
        input_dict['beta'],
        input_dict['image'],
        input_dict['projection_matrix'],
        n_samples,
        color=(0, 255, 0)
    )

    # combined image
    n_samples = len(input_images)

    canvases = []
    for sample_index in range(n_samples):
        canvas = np.concatenate([input_images[sample_index], pred_images[sample_index], gt_images[sample_index]], axis=1)
        canvases.append(canvas)

    return canvases


def vis_triple_with_smplx(smplx_model, renderer, input_dict, output_dict, n_samples, alpha=0.5):
    # input 2d keypoints
    input_images = vis_image_with_keypoints_2d(
        input_dict['keypoints_orig'][:, :, :2],
        input_dict['image'],
        n_samples,
        color=(255, 0, 0),
    )
    
    ## add white image
    input_images = [
        np.concatenate([input_image, 255 * np.ones_like(input_image, dtype=np.uint8)], axis=0) for input_image in input_images
    ]

    # pred
    pred_images_smplx_overlay = vis_image_with_smplx(
        smplx_model,
        renderer,
        output_dict['expression_pred'],
        output_dict['pose_pred'],
        input_dict['beta'],
        input_dict['image'],
        input_dict['camera_matrix'],
        n_samples,
        alpha=alpha
    )

    pred_images_smplx_only = vis_image_with_smplx(
        smplx_model,
        renderer,
        output_dict['expression_pred'],
        output_dict['pose_pred'],
        input_dict['beta'],
        input_dict['image'],
        input_dict['camera_matrix'],
        n_samples,
        alpha=0.0
    )

    pred_images = [
        np.concatenate(image_pair, axis=0) for image_pair in zip(pred_images_smplx_overlay, pred_images_smplx_only)
    ]

    # gt
    gt_images_smplx_overlay = vis_image_with_smplx(
        smplx_model,
        renderer,
        input_dict['expression'],
        input_dict['pose'],
        input_dict['beta'],
        input_dict['image'],
        input_dict['camera_matrix'],
        n_samples,
        alpha=alpha
    )

    gt_images_smplx_only = vis_image_with_smplx(
        smplx_model,
        renderer,
        input_dict['expression'],
        input_dict['pose'],
        input_dict['beta'],
        input_dict['image'],
        input_dict['camera_matrix'],
        n_samples,
        alpha=0.0
    )

    gt_images = [
        np.concatenate(image_pair, axis=0) for image_pair in zip(gt_images_smplx_overlay, gt_images_smplx_only)
    ]

    # combined image
    n_samples = len(input_images)

    canvases = []
    for sample_index in range(n_samples):
        canvas = np.concatenate([input_images[sample_index], pred_images[sample_index], gt_images[sample_index]], axis=1)
        canvases.append(canvas)

    return canvases
