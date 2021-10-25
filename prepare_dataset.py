from utils import get_ray_fn, init_ray
import hydra
from omegaconf import DictConfig
from pathlib import Path
from environment.baseEnv import BaseEnv
from environment.camera import SimCameraBase
import trimesh
import numpy as np
from environment.meshRendererEnv import MeshRendererEnv
from utils.tsdfHelper import TSDFHelper
import pybullet as p
from skimage.morphology import label
import ray
from os.path import exists

def generate_kit(obj_path, obj_bounds, kit_bounds, voxel_size, image_size):
    if not exists(obj_path):
        return
    env = BaseEnv(gui=False)
    p.setGravity(0, 0, 0)
    # Setup an orthographic camera:
    # (Using an orthographic camera removes the anglular volume when captured from only one image)
    focal_length = 63e4
    z = -100
    view_matrix = p.computeViewMatrix((0, 0, z), (0, 0, 0), (1, 0, 0))
    camera = SimCameraBase(view_matrix, image_size,
                           z_near=-z-0.05, z_far=-z+0.05, focal_length=focal_length)

    kit_vol_shape = np.ceil((kit_bounds[:, 1] - kit_bounds[:, 0]) / voxel_size).astype(np.int)

    mesh = trimesh.load(obj_path)
    # Center mesh around origin
    mesh.vertices -= (mesh.vertices.max(axis=0) + mesh.vertices.min(axis=0)) / 2
    # Scael mesh to fit the desired obj_bounds precisely
    scale_factor = (obj_bounds[:,1] - obj_bounds[:, 0]) / (mesh.vertices.max(axis=0) - mesh.vertices.min(axis=0))
    mesh.vertices *= scale_factor 
    # Shift mesh such that the bottom face is at origin
    mesh.vertices[:, 2] -= mesh.vertices.min(axis=0)[2] + obj_bounds[2, 0]
    # Dump mesh
    processed_obj_path = obj_path.parent / f"obj.obj"
    mesh.export(processed_obj_path) 
    # Load mesh
    urdf_path = MeshRendererEnv.dump_obj_urdf(processed_obj_path, rgba=np.array([0, 1, 0, 1]), load_collision_mesh=True)
    p.loadURDF(str(urdf_path))

    # Adjust obj bounds to make them slightly bigger (to account for zooming logic below)
    delta = 0.01  # margin is 10 mm
    obj_bounds_zoomed = np.copy(obj_bounds)
    obj_bounds_zoomed[:2, 0] -= delta  # Only along x,y axis
    obj_bounds_zoomed[:2, 1] += delta
    color_im, depth_im, _ = camera.get_image()
    part_tsdf = TSDFHelper.tsdf_from_camera_data(
        views=[(color_im, depth_im,
                camera.intrinsics,
                camera.pose_matrix)],
        bounds=obj_bounds_zoomed,
        voxel_size=voxel_size,
    )

    # The following (zooming) logic make the object to be slightly larger while shirking the holes
    # (or preserving the location of holes)
    # The logic is as follows (thanks Shuran!!)
    # - Slide the object along +x axis by a few voxels
    # - Union it with previous object volume
    # - Repeat above for all (+-)(xyz) directions
    occ_grid = np.zeros_like(part_tsdf)
    occ_grid[part_tsdf < 0] = -1
    # now. Shift the obj volume 
    max_delta_voxels = max(1, np.ceil(delta / voxel_size).astype(np.int))
    k = np.where(occ_grid == -1)
    for x_delta_voxels in range(-max_delta_voxels + 1, max_delta_voxels + 1):
        for y_delta_voxels in range(-max_delta_voxels + 1, max_delta_voxels + 1):
            occ_grid[np.clip(k[0] + x_delta_voxels, 0, occ_grid.shape[0] - 1), np.clip(k[1] + y_delta_voxels, 0, occ_grid.shape[1] - 1), k[2]] = -1


    part_tsdf = - occ_grid
    # Ok. Now wrap this volumes inside the proper kit volume
    kit_vol = -1 * np.ones((kit_vol_shape))
    kit_x = np.ceil((0.02 - delta)/ voxel_size).astype(int)
    kit_z = np.ceil(0.02 / voxel_size).astype(int)
    kit_vol[
        kit_x: (kit_x + part_tsdf.shape[0]),
        kit_x: (kit_x + part_tsdf.shape[1]),
        -kit_z:,
    ] = part_tsdf[:, :, :kit_z]

    (obj_path.parent / "kit_parts").mkdir(exist_ok=True)
    kit_mesh_path = obj_path.parent / "kit_parts/kit.obj"
    if TSDFHelper.to_mesh(kit_vol, kit_mesh_path, voxel_size, vol_origin=[0, 0, 0.009]):
        print(obj_path)
    else:
        print(obj_path, ": kit generation failed")

    def change_color(mesh, color):
        for facet in mesh.facets:
            mesh.visual.face_colors[facet] = color
        return mesh

    # # For debugging: Load Kit
    # obj_mesh = trimesh.load_mesh(processed_obj_path)
    # obj_mesh = change_color(obj_mesh, np.array([52, 89, 23, 255]))
    # kit_mesh = trimesh.load_mesh(kit_mesh_path)
    # kit_mesh = change_color(kit_mesh, np.array([45, 93, 196, 255]))
    # scene = trimesh.Scene(obj_mesh)
    # scene.add_geometry(kit_mesh)
    # scene.show()


@hydra.main(config_path="conf", config_name="config")
def main(cfg: DictConfig):
    gpu_frac = 0.05
    adjusted_kit_width = 0.1 - 0.02
    hw = adjusted_kit_width / 2
    obj_bounds = np.array([
        [-hw, hw],
        [-hw, hw],
        [0, 0.05],
    ])

    kit_bounds = np.array([
        [-0.06, 0.06],
        [-0.06, 0.06],
        [-0.002, 0.02],
    ])
    voxel_size = cfg.env.voxel_size
     
    def get_obj_paths(split_file):
        with open(split_file, "r") as f:
            obj_paths = [Path(obj_path.strip()) for obj_path in f.readlines()]
        return obj_paths
    obj_paths = list()
    for data_paths in cfg.preprocess_paths:
        obj_paths += get_obj_paths(data_paths)

    use_ray = init_ray(cfg.ray)
    fn = get_ray_fn(cfg.ray, generate_kit, gpu_frac=gpu_frac)
    tasks = list()
    image_size = np.array(cfg.env.image_size)
    for obj_path in obj_paths:
        tasks.append(fn(obj_path, obj_bounds, kit_bounds, voxel_size, image_size))

    if use_ray:
        tasks = ray.get(tasks)

if __name__ == "__main__":
    main()