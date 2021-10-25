import pybullet as p
import numpy as np
from utils.tsdfHelper import TSDFHelper
from environment.camera import SimCameraPosition
from enum import Enum
from pathlib import Path
from subprocess import check_call
from os import devnull
from sys import platform


class SphereMarker:
    def __init__(self, position, radius=0.05, rgba_color=(1, 0, 0, 0.8), text=None, orientation=None, p_id=0):
        self.p_id = p_id
        position = np.array(position)
        vs_id = p.createVisualShape(
            p.GEOM_SPHERE, radius=radius, rgbaColor=rgba_color, physicsClientId=self.p_id)

        self.marker_id = p.createMultiBody(
            baseMass=0,
            baseInertialFramePosition=[0, 0, 0],
            baseCollisionShapeIndex=-1,
            baseVisualShapeIndex=vs_id,
            basePosition=position,
            useMaximalCoordinates=False
        )

        self.debug_item_ids = list()
        if text is not None:
            self.debug_item_ids.append(
                p.addUserDebugText(text, position + radius)
            )
        
        if orientation is not None:
            # x axis
            axis_size = 2 * radius
            rotation_mat = np.asarray(p.getMatrixFromQuaternion(orientation)).reshape(3,3)

            # x axis
            x_end = np.array([[axis_size, 0, 0]]).transpose()
            x_end = np.matmul(rotation_mat, x_end)
            x_end = position + x_end[:, 0]
            self.debug_item_ids.append(
                p.addUserDebugLine(position, x_end, lineColorRGB=(1, 0, 0))
            )
            # y axis
            y_end = np.array([[0, axis_size, 0]]).transpose()
            y_end = np.matmul(rotation_mat, y_end)
            y_end = position + y_end[:, 0]
            self.debug_item_ids.append(
                p.addUserDebugLine(position, y_end, lineColorRGB=(0, 1, 0))
            )
            # z axis
            z_end = np.array([[0, 0, axis_size]]).transpose()
            z_end = np.matmul(rotation_mat, z_end)
            z_end = position + z_end[:, 0]
            self.debug_item_ids.append(
                p.addUserDebugLine(position, z_end, lineColorRGB=(0, 0, 1))
            )

    def __del__(self):
        p.removeBody(self.marker_id, physicsClientId=self.p_id)
        for debug_item_id in self.debug_item_ids:
            p.removeUserDebugItem(debug_item_id)


class CuboidMarker:
    def __init__(self, bb_min, bb_max, wTa=np.identity(4), rgb_color=np.array([255,0,0])):
        """
        :param bb_min: bounding box min: np array of shape (3,)
        :param bb_max: bounding box max: np array of shape (3,)
        :param wTa: Translation matrix from object's frame to world frame.
        """
        self.debug_line_ids = []
        self.wTa = wTa
        rgb_color = rgb_color / 255

        f = self.translate_to_world_coordinates([bb_min[0], bb_min[1], bb_min[2]])
        t = self.translate_to_world_coordinates([bb_max[0], bb_min[1], bb_min[2]])
        self.debug_line_ids.append(p.addUserDebugLine(f, t, rgb_color))
        f = self.translate_to_world_coordinates([bb_min[0], bb_min[1], bb_min[2]])
        t = self.translate_to_world_coordinates([bb_min[0], bb_max[1], bb_min[2]])
        self.debug_line_ids.append(p.addUserDebugLine(f, t, rgb_color))
        f = self.translate_to_world_coordinates([bb_min[0], bb_min[1], bb_min[2]])
        t = self.translate_to_world_coordinates([bb_min[0], bb_min[1], bb_max[2]])
        self.debug_line_ids.append(p.addUserDebugLine(f, t, rgb_color))
        f = self.translate_to_world_coordinates([bb_min[0], bb_min[1], bb_max[2]])
        t = self.translate_to_world_coordinates([bb_min[0], bb_max[1], bb_max[2]])
        self.debug_line_ids.append(p.addUserDebugLine(f, t, rgb_color))
        f = self.translate_to_world_coordinates([bb_min[0], bb_min[1], bb_max[2]])
        t = self.translate_to_world_coordinates([bb_max[0], bb_min[1], bb_max[2]])
        self.debug_line_ids.append(p.addUserDebugLine(f, t, rgb_color))
        f = self.translate_to_world_coordinates([bb_max[0], bb_min[1], bb_min[2]])
        t = self.translate_to_world_coordinates([bb_max[0], bb_min[1], bb_max[2]])
        self.debug_line_ids.append(p.addUserDebugLine(f, t, rgb_color))
        f = self.translate_to_world_coordinates([bb_max[0], bb_min[1], bb_min[2]])
        t = self.translate_to_world_coordinates([bb_max[0], bb_max[1], bb_min[2]])
        self.debug_line_ids.append(p.addUserDebugLine(f, t, rgb_color))
        f = self.translate_to_world_coordinates([bb_max[0], bb_max[1], bb_min[2]])
        t = self.translate_to_world_coordinates([bb_min[0], bb_max[1], bb_min[2]])
        self.debug_line_ids.append(p.addUserDebugLine(f, t, rgb_color))
        f = self.translate_to_world_coordinates([bb_min[0], bb_max[1], bb_min[2]])
        t = self.translate_to_world_coordinates([bb_min[0], bb_max[1], bb_max[2]])
        self.debug_line_ids.append(p.addUserDebugLine(f, t, rgb_color))
        f = self.translate_to_world_coordinates([bb_max[0], bb_max[1], bb_max[2]])
        t = self.translate_to_world_coordinates([bb_min[0], bb_max[1], bb_max[2]])
        self.debug_line_ids.append(p.addUserDebugLine(f, t, rgb_color))
        f = self.translate_to_world_coordinates([bb_max[0], bb_max[1], bb_max[2]])
        t = self.translate_to_world_coordinates([bb_max[0], bb_min[1], bb_max[2]])
        self.debug_line_ids.append(p.addUserDebugLine(f, t, rgb_color))
        f = self.translate_to_world_coordinates([bb_max[0], bb_max[1], bb_max[2]])
        t = self.translate_to_world_coordinates([bb_max[0], bb_max[1], bb_min[2]])
        self.debug_line_ids.append(p.addUserDebugLine(f, t, rgb_color))

    def translate_to_world_coordinates(self, aP):
        wP = np.dot(self.wTa, np.array([[aP[0]], [aP[1]], [aP[2]], [1]]))
        return wP[:3]

    def __del__(self):
        for debug_line_id in self.debug_line_ids:
            p.removeUserDebugItem(debug_line_id)


def transform_point3s(t, ps):
    if len(ps.shape) != 2 or ps.shape[1] != 3:
        raise ValueError('Invalid input points p')

    # convert to homogeneous
    ps_homogeneous = np.hstack([ps, np.ones((len(ps), 1), dtype=np.float32)])
    ps_transformed = np.dot(t, ps_homogeneous.T).T

    return ps_transformed[:, :3]


def get_tableau_palette():
    """
    returns a beautiful color palette
    :return palette (np.array object): np array of rgb colors in range [0, 1]
    """
    palette = np.array(
        [
            [89, 169, 79],  # green
            [156, 117, 95],  # brown
            [237, 201, 72],  # yellow
            [78, 121, 167],  # blue
            [255, 87, 89],  # red
            [242, 142, 43],  # orange
            [176, 122, 161],  # purple
            [255, 157, 167],  # pink
            [118, 183, 178],  # cyan
            [186, 176, 172]  # gray
        ],
        dtype=np.float
    )
    return palette / 255.


def get_body_colors(body_id):
    link_colors = dict()
    for visual_info in p.getVisualShapeData(body_id):
        link_colors[visual_info[1]] = visual_info[7]
    return link_colors 


def set_visible(visual_data, visible=True):
    if visual_data is None:
        return
    for body_id, link_colors in visual_data.items():
        for link_ind, link_color in link_colors.items():
            p.changeVisualShape(
                body_id, link_ind,
                rgbaColor=link_color if visible else (0, 0, 0, 0)
            )

def change_body_color(body_id, rgbaColor):
    for link_index in range(-1, p.getNumJoints(body_id)):
        p.changeVisualShape(body_id, link_index, rgbaColor=rgbaColor)

def get_surrounding_cameras(
    bounds,
    look_at,
    image_size,
    main_camera_top_down=False,
    less_cameras: bool = False
):
    uv = [0, 0, 1]
    def get_mid(row):
        return (row[0] + row[1]) / 2

    xmin, xmax, xmid = *bounds[0], get_mid(bounds[0])
    ymin, ymax, ymid = *bounds[1], get_mid(bounds[1])
    zmin, zmax, _ = *bounds[2], get_mid(bounds[2])
    cameras = [
        SimCameraPosition([xmid, ymid, zmax], look_at, [-1, 0, 0], image_size=image_size),
        # Middle of top face edges
        SimCameraPosition([xmax, ymid, zmax], look_at, uv, image_size=image_size),  # Main camera
        SimCameraPosition([xmid, ymax, zmax], look_at, uv, image_size=image_size),  # Surrounding cameras
        SimCameraPosition([xmin, ymid, zmax], look_at, uv, image_size=image_size),
        SimCameraPosition([xmid, ymin, zmax], look_at, uv, image_size=image_size),
        # Bottom face vertices
        SimCameraPosition([xmax, ymax, zmin], look_at, uv, image_size=image_size),
        SimCameraPosition([xmax, ymin, zmin], look_at, uv, image_size=image_size),
        SimCameraPosition([xmin, ymax, zmin], look_at, uv, image_size=image_size),
        SimCameraPosition([xmin, ymin, zmin], look_at, uv, image_size=image_size),
    ]
    if less_cameras:
        cameras = [cameras[0], cameras[1], cameras[2], cameras[-1]]
        #print("=======>FIXME<=======: Using less surrounding cameras for debugging")
    return cameras


class SCENETYPE(Enum):
    KIT=0  # only kit
    OBJECTS=1  # only objects
    KIT_OBJECTS=2  # objects present on top of kit


def create_collision_mesh(
        mesh_path: Path,
        output_path: Path = None,
        timeout: int = 3000,
        high_res=False):
    if not mesh_path.exists():
        return None
    if output_path is None:
        output_path = mesh_path.parent / f"{mesh_path.stem}_cm.obj"
    vhacd = "assets/vhacd/vhacd_linux"\
        if platform == "linux" or platform == "linux2"\
        else "assets/vhacd/vhacd_osx"
    with open(devnull, 'w') as FNULL:
        cmds = [
            vhacd,
            "assets/vhacd/testVHACD",
            "--input", mesh_path,
            "--output", output_path
        ]
        if high_res:
            cmds.extend([
                "--resolution 10000000",
                "--depth 32",
                "--planeDownsampling 1",
                "--maxNumVerticesPerCH 1024",
                "--convexhullDownsampling 1",
                "--concavity 0.000001"
            ])
        else:
            cmds.append("--resolution 25000")
        try:
            check_call(
                cmds,
                stdout=FNULL,
                timeout=timeout)
            return output_path
        except Exception as e:
            print(e)
            return None


def get_scene_volume(cameras, view_bounds, voxel_size, return_first_image=False, initial_value=-1):
    views = list()
    ret_images = (None, None, None)
    for i, camera in enumerate(cameras):
        if i == 0 and return_first_image:
            rgb, d, mask = camera.get_image(seg_mask=True)
            ret_images = (rgb, d, mask)
        else:
            rgb, d, _ = camera.get_image(seg_mask=False)
        views.append((rgb, d, camera.intrinsics, camera.pose_matrix))
    
    vol = TSDFHelper.tsdf_from_camera_data(views=views, bounds=view_bounds,
        voxel_size=voxel_size, initial_value=initial_value)
    
    return vol, ret_images[0], ret_images[1], ret_images[2]