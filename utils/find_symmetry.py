import trimesh
from scipy.spatial.transform import Rotation as R
import numpy as np
from scipy.spatial.distance import cdist
from tqdm import tqdm
from os.path import exists
import h5py
from utils.tsdfHelper import TSDFHelper

def normalize_pc(pts):
    mean = np.mean(pts, axis=0)
    pts = pts - mean
    m = np.max(np.sqrt(np.sum(pts ** 2, axis=1)))
    pts = pts/m
    return pts

def get_symmetry_planes_axis(pts, cnt, cutoff, axis):
    vals = np.empty(11)
    for i in range(11):
        r = R.from_euler(axis, (i+1)*30, degrees=True).as_matrix()
        rpts = (r @ pts.T).T
        dist = cdist(pts, rpts)
        vals[i] = np.mean(np.min(dist, axis=1))
    # print(axis)
    # for i, val in enumerate(vals):
    #    print((i+1)*30, f'{val:.4f}')
    if np.all(vals<cutoff):
        return 0
    elif not np.any(vals<cutoff):
        return -1
    iso = np.where(vals<cutoff)[0]
    iso = (iso + 1) * 3
    iso = np.insert(iso, 0, 0)
    diffs = np.array([iso[i+1]-iso[i] for i in range(len(iso)-1)])
    if not np.all(diffs==diffs[0]):
        # return np.array2string(iso)
        print("wierd symmetry: ", iso)
        return -1
    else:
        return diffs[0]*10
    
def get_symmetry_planes(mesh_path, cnt=5000, cutoff=0.045, return_pts=False):
    mesh = trimesh.load(mesh_path)
    pts = mesh.sample(cnt, return_index=False)
    pts = normalize_pc(pts)
    symmetries = np.empty(3)
    for i, axis in enumerate(['x', 'y', 'z']):
        if axis=='z':
            pts_crop = pts[pts[:,2]<0, :]
            pts_crop = normalize_pc(pts_crop)
            symmetries[i] = get_symmetry_planes_axis(pts_crop, cnt, cutoff, axis)
        else:
            symmetries[i] = get_symmetry_planes_axis(pts, cnt, cutoff, axis)
        # print(symmetries[axis])
    if return_pts:
        return symmetries, pts
    return symmetries

def eval_symmetry(pts, syms, save_dir, prefix):
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    fig = plt.figure(figsize=(4,4))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(pts[:,0],pts[:,1],pts[:,2])

    s = f'x: {syms[0]}, y: {syms[1]}, z: {syms[2]}'
    ax.text2D(0.05, 0.95, s, transform=ax.transAxes)
    plt.savefig(save_dir / f'{prefix}sym.jpg')
    plt.close()