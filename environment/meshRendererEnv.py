from numpy.core.function_base import linspace
from .baseEnv import BaseEnv
from pathlib import Path
import pybullet as p
from .camera import SimCameraYawPitchRoll
import numpy as np
from utils.tsdfHelper import TSDFHelper
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot as plt
import matplotlib.cm as cm
from matplotlib import animation
from scipy.ndimage import zoom


visual_geom_template = """
        <visual>
          <origin rpy="0 0 0" xyz="0 0 0"/>
          <geometry>
            <mesh filename="{mesh_path}" scale="1 1 1"/>
          </geometry>
            <material name="{red}_{green}_{blue}_{alpha}">
                <color rgba="{red} {green} {blue} {alpha}"/>
            </material>
        </visual>
"""
collision_geom_template = """
        <collision>
          <origin rpy="0 0 0" xyz="0 0 0"/>
          <geometry>
            <mesh filename="{mesh_path}" scale="1 1 1"/>
          </geometry>
            <material>
                <color rgba="{red} {green} {blue} {alpha}"/>
            </material>
        </collision>
"""
grasp_object_template = """
    <robot name="block">
      <link name="block_base_link">
        <inertial>
            <origin rpy="0 0 0" xyz="0 0 0"/>
            <mass value="0.1"/>
            <inertia ixx="0.1" ixy="0" ixz="0" iyy="0.1" iyz="0" izz="0.1"/>
        </inertial>
        {visual_geom}
        {collision_geom}
      </link>
    </robot>
"""

class MeshRendererEnv(BaseEnv):
    def __init__(self, gui=False):
        super().__init__(gui)
        p.setGravity(0, 0, 0)
        self.body_ids = list()
        self.bb_min = np.infty * np.ones((3,))
        self.bb_max = -np.infty * np.ones((3,))

    @staticmethod
    def dump_obj_urdf(mesh_path, urdf_path=None, rgba: np.ndarray = np.ones((4,)), load_collision_mesh: bool = False):
        if urdf_path is None:
          urdf_path = mesh_path.parent / f"{mesh_path.stem}.urdf"
        with open(urdf_path, "w") as f:
            visual_geom = visual_geom_template.format(
                mesh_path=mesh_path, red=rgba[0], green=rgba[1], blue=rgba[2], alpha=rgba[3])
            if load_collision_mesh:
                collision_geom = collision_geom_template.format(
                    mesh_path=mesh_path, red=rgba[0], green=rgba[1], blue=rgba[2], alpha=rgba[3])
            else:
                collision_geom = ""
            f.write(grasp_object_template.format(visual_geom=visual_geom, collision_geom=collision_geom))
        return urdf_path

    def render(
        self,
        gif_path: Path,
        bb_min: np.ndarray = -0.08 * np.ones((3,)),
        bb_max: np.ndarray = 0.08 * np.ones((3,)),
        num_images: int = 40
    ) -> Path:
        if bb_min is None or bb_max is None:
            bb_min = self.bb_min
            bb_max = self.bb_max
        bb = np.vstack((bb_min, bb_max)).T
        tp = (bb[:, 0] + bb[:, 1]) / 2
        dist = np.sqrt(((bb[:, 1] - bb[:, 0])**2).sum())  # Put camera at a distance of diagonal away from center
        self.cameras = [
            SimCameraYawPitchRoll(tp, dist, yaw, -40, 0)
            for yaw in np.linspace(0, 360, num_images)
        ]
        self.get_scene_gif(gif_path)
        return gif_path
    
    def load_mesh(self, mesh_path: Path, pos: np.ndarray = np.zeros((3,)), ori: np.ndarray = np.array([0, 0, 0, 1]),
        rgba: np.ndarray = np.ones((4,)), load_collision_mesh: bool = False, scale: float=1, urdf_path:Path=None):
        urdf_path = self.dump_obj_urdf(mesh_path, urdf_path=urdf_path, rgba=rgba, load_collision_mesh=load_collision_mesh)
        obj_body_id = p.loadURDF(str(urdf_path), pos, ori, globalScaling=scale)
        self.body_ids.append(obj_body_id)
        bb_min, bb_max = p.getAABB(obj_body_id)
        bb_min, bb_max = np.array(bb_min), np.array(bb_max)
        if ((bb_max - bb_min) < 0.005).any():
            delta = np.ones((3,)) * 0.1
            bb_min = pos - delta
            bb_max = pos + delta
        self.bb_min = np.minimum(self.bb_min, bb_min)
        self.bb_max = np.maximum(self.bb_max, bb_max)
        return obj_body_id

    def reset(self):
        for body_id in self.body_ids:
            p.removeBody(body_id)
        self.body_ids = list()
        self.bb_min = np.infty * np.ones((3,))
        self.bb_max = -np.infty * np.ones((3,))


def dump_tsdf_vis(tsdf_in, dump_path):
    tsdf = np.copy(tsdf_in)
    tsdf_in_shape = tsdf_in.shape
    tsdf_shape = np.array([50, 50, 50])
    if np.any(tsdf_in_shape > tsdf_shape):
        zoom_factor = tsdf_shape / tsdf_in_shape
        tsdf = zoom(tsdf, zoom_factor)

    x = np.arange(tsdf.shape[0])[:, None, None]
    y = np.arange(tsdf.shape[1])[None, :, None]
    z = np.arange(tsdf.shape[2])[None, None, :]
    x, y, z = np.broadcast_arrays(x, y, z)

    c = cm.hsv((tsdf.ravel() + 1))
    alphas = (tsdf.ravel() < 0).astype(float)
    c[..., -1] = alphas

    fig = plt.figure()
    fig.tight_layout()
    ax = fig.gca(projection='3d')

    def initf():
        ax.scatter(
            (x + np.random.random(x.shape) - 0.5).ravel(),
            (y + np.random.random(y.shape) - 0.5).ravel(),
            (z + np.random.random(z.shape) - 0.5).ravel(),
            c=c, s=1
        )
        ax.w_xaxis.set_pane_color((0.0, 0.0, 0.0, 0.0))
        ax.w_yaxis.set_pane_color((0.0, 0.0, 0.0, 0.0))
        ax.w_zaxis.set_pane_color((0.0, 0.0, 0.0, 0.0))

        # Hide axes ticks
        ax.tick_params(axis='x', colors=(0.0, 0.0, 0.0, 0.0))
        ax.tick_params(axis='y', colors=(0.0, 0.0, 0.0, 0.0))
        ax.tick_params(axis='z', colors=(0.0, 0.0, 0.0, 0.0))
        return ax,

    if dump_path.suffix == ".gif":
        nframes = 30
        def animate(i):
            ax.view_init(elev=10., azim=25 + 20 * np.sin(2 * np.pi / nframes * i))
            return ax,

        anim = animation.FuncAnimation(fig, animate, init_func=initf,
                                    frames=nframes)
        anim.save(dump_path, writer='imagemagick', fps=30)
        plt.close(fig)
        return dump_path 
    else:
        initf()
        img_paths = list()
        def dump_view(elev, azim, ind):
            ax.view_init(elev, azim) 
            img_path = dump_path.parent / f"{dump_path.stem}_view{ind}.png"
            plt.savefig(str(img_path))
            img_paths.append(img_path)
            ind += 1
        dump_view(20, 45, 0)
        dump_view(20, -45, 1)
        plt.close(fig)
        return img_paths


def dump_vol_render_gif(
    v,
    mesh_path,
    voxel_size,
    visualize_mesh_gif=False,
    visualize_tsdf_gif=False,
    gif_num_images: int = 40,
):
    TSDFHelper.to_mesh(v, mesh_path, voxel_size)
    if not mesh_path.exists():
        return None, None
    vol_gif_path = None
    if visualize_mesh_gif:
        env = MeshRendererEnv()
        env.load_mesh(mesh_path)
        vol_gif_path = env.render(
            mesh_path.parent / f"{mesh_path.stem}.gif",
            num_images=gif_num_images
        )

    tsdf_vis_path = None
    if visualize_tsdf_gif:
        tsdf_vis_path = dump_tsdf_vis(v, mesh_path.parent / f"{mesh_path.stem}_tsdf.gif")
    return vol_gif_path, tsdf_vis_path 
