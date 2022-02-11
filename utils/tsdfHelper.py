from utils.fusion import TSDFVolume
import numpy as np
from skimage.measure import marching_cubes
from numpy import copy
from skimage.morphology import label

def extend_to_bottom(tsdf_vol):
    h, w, d = tsdf_vol.shape
    tsdf_vol_new = tsdf_vol.copy()
    for i in range(h):
        for j in range(d):
            col = np.where(tsdf_vol[i,j,:]<0)[0]
            if len(col) == 0:
                continue
            last_vol = col[-1]
            tsdf_vol_new[i,j,:last_vol] = -1
    return tsdf_vol_new

def get_single_biggest_cc_single(tsdf_vol,
                                 return_num_components=False,
                                 level=0.0):
    # Create a mask which represents presence of volume
    binary_vol_mask = np.array(tsdf_vol < level, dtype=int)
    # label connected components for mask
    labeled_vol_mask, num_components = label(binary_vol_mask, return_num=True)
    max_size = 0
    max_size_component_i = 0  # 0 represents background

    for component_i in range(1, num_components + 1):
        size = (labeled_vol_mask == component_i).sum()
        if max_size < size:
            max_size = size
            max_size_component_i = component_i
    max_size_vol = np.array(
        labeled_vol_mask == max_size_component_i, dtype=int) * tsdf_vol + \
        np.ones(tsdf_vol.shape) * np.array(labeled_vol_mask != max_size_component_i)
    if return_num_components:
        return max_size_vol, num_components
    else:
        return max_size_vol

def export_obj(filename, verts, faces, norms, colors=None):
    # Write header
    obj_file = open(filename, 'w')

    # Write vertex list
    for i in range(verts.shape[0]):
        obj_file.write("v %f %f %f\n" %
                       (verts[i, 0], verts[i, 1], verts[i, 2]))

    for i in range(norms.shape[0]):
        obj_file.write("vn %f %f %f\n" %
                       (norms[i, 0], norms[i, 1], norms[i, 2]))

    faces = copy(faces)
    faces += 1

    for i in range(faces.shape[0]):
        obj_file.write("f %d %d %d\n" %
                       (faces[i, 0], faces[i, 1], faces[i, 2]))

    obj_file.close()


class TSDFHelper:
    @staticmethod
    def tsdf_from_camera_data(views, bounds, voxel_size, initial_value=-1, trunc_margin_factor=5):
        # Initialize voxel volume
        tsdf_vol = TSDFVolume(bounds, voxel_size=voxel_size, initial_value=initial_value, trunc_margin_factor=trunc_margin_factor)
        # Fuse different views to one voxel
        for view in views:
            tsdf_vol.integrate(*view, obs_weight=1.)
        return tsdf_vol.get_volume()[0]

    @staticmethod
    def to_mesh(tsdf,
                path,
                voxel_size,
                vol_origin=[0, 0, 0],
                level=0.0,
                center_mesh=False):
        if type(tsdf) != np.array:
            tsdf = np.array(tsdf)
        # Block off sides to get valid marching cubes
        tsdf = copy(tsdf)
        tsdf[:, :, -1] = 1
        tsdf[:, :, 0] = 1
        tsdf[:, -1, :] = 1
        tsdf[:, 0, :] = 1
        tsdf[-1, :, :] = 1
        tsdf[0, :, :] = 1
        center = None
        if tsdf.min() > level or tsdf.max() < level:
            print(f"TSDF min {tsdf.min():.3f} or max {tsdf.max():.3f} is out of level :(")
            if center_mesh:
                return False, center
            else:
                return False
        try:
            verts, faces, norms, _ = marching_cubes(
                tsdf,
                level=level)

            # Shift the origin from (bottom left back)
            # vertex to the center of the volume.
            verts = verts - np.array([*tsdf.shape]) / 2
            if center_mesh:
                # Find center of all vertices
                center = verts.mean(axis=0)
                verts = verts - center

            # scale to final volume size
            verts = verts * voxel_size

            # now move the mesh origin to world's vol_origin
            verts += vol_origin
            export_obj(path, verts, faces, norms)
            if center_mesh:
                return True, center
            else:
                return True
        except Exception as e:
            print(e)
            if center_mesh:
                return False, center
            else:
                return False
