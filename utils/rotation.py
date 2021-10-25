import numpy as np
from scipy.spatial.transform import Rotation as R
import torch
from torch.nn import functional as F
from itertools import product
import pybullet as p

# def sample_directions(num_z, num_phi, max_pert_theta):
#     """
#     Sample directions in vicinity of direction v1 uniformly
#     vicinity: spherical segment defined by max_pert_theta around v1
#     granularity: is controlled by num_z, num_phi
#     Reference: https://math.stackexchange.com/q/205589
#     """
#     zs = np.linspace(np.cos(max_pert_theta), 1, num=num_z, endpoint=False)
#     phis = np.linspace(0, 2*np.pi, num=num_phi)
#     uniform_dirs = list()
#     for z in zs:
#         for phi in phis:
#             hyp = np.sqrt(1 - z**2)
#             uniform_dirs.append(np.array([hyp*np.cos(phi), hyp*np.sin(phi), z]))
#     return uniform_dirs

def rotation_matrix_from_vectors(vec1, vec2):
    """ Find the rotation matrix that aligns vec1 to vec2
    :param vec1: A 3d "source" vector
    :param vec2: A 3d "destination" vector
    :return mat: A transform matrix (3x3) which when applied to vec1, aligns it with vec2.
    """
    if np.allclose(vec1, vec2):
        return np.eye(3)
    a, b = (vec1 / np.linalg.norm(vec1)).reshape(3), (vec2 / np.linalg.norm(vec2)).reshape(3)
    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)
    kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    rotation_matrix = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))
    return rotation_matrix

# def v_theta_to_quaternion(quat_v: np.ndarray, quat_theta: float):
#     return np.array([
#         quat_v[0] * np.sin(quat_theta / 2),
#         quat_v[1] * np.sin(quat_theta / 2),
#         quat_v[2] * np.sin(quat_theta / 2),
#         np.cos(quat_theta / 2),
#     ])

# def sample_rot(max_pert_theta, max_pert_rot, num_z, num_phi, num_rots):
#     """
#     Given a quaternion, sample quaternions uniformly in its vicinity
#     - vicinity is defined by max_pert_theta
#     - sampling granularity is defined by num_*
#     Checkout scripts/visualize_6DoF_angle_sampling.py for visualization
#     """
#     v2s = sample_directions(num_z, num_phi, max_pert_theta)

#     rot_angles = np.linspace(-max_pert_rot, max_pert_rot, num=num_rots)
#     v2s = np.array([v_theta_to_quaternion(v2, angle) for v2 in v2s for angle in rot_angles])
#     rot_mat = np.array([R.from_quat(v2).as_matrix() for v2 in v2s])
#     return v2s, rot_mat

# def sample_rot(rot_config):
#     xconf, yconf, zconf = rot_config
#     def gen_angles(rg, n):
#         if rg == 0:
#             return [0]
#         return np.arange(-rg, rg+0.1, rg/n)
#     angles_x = gen_angles(*xconf)
#     angles_y = gen_angles(*yconf)
#     angles_z = gen_angles(*zconf)
#     euler_angles = list(product(angles_x,angles_y,angles_z))
#     rot_mats = np.array([R.from_euler('xyz', euler, degrees=True).as_matrix() for euler in euler_angles]) # (nx*ny*nz, 3, 3)
#     rot_quats = np.array([R.from_euler('xyz', euler, degrees=True).as_quat() for euler in euler_angles]) # (nx*ny*nz, 4)

#     return rot_quats, rot_mats

def multiply_quat(quat1, quat2):
    return np.array(p.multiplyTransforms((0,0,0), quat1, (0,0,0), quat2)[1])

def normal_to_mat(normal):
    return rotation_matrix_from_vectors(np.array([0,0,1]), normal)

def normal_to_quat(normal):
    return mat_to_quat(normal_to_mat(normal))

def normal_to_euler(normal):
    return mat_to_euler(normal_to_mat(normal))

def quat_to_normal(quat):
    mat = quat_to_mat(quat)
    return mat @ np.array([0,0,1])

def quat_to_mat(quat):
    return R.from_quat(quat).as_matrix()

def mat_to_quat(mat):
    return R.from_matrix(mat).as_quat()

def mat_to_euler(mat):
    return quat_to_euler(mat_to_quat(mat))

def euler_to_mat(euler):
    return quat_to_mat(euler_to_quat(euler))

def quat_to_euler(quat, degrees=True):
    if degrees:
        return np.array(p.getEulerFromQuaternion(quat)) * 180/np.pi
    else:
        return np.array(p.getEulerFromQuaternion(quat))

def euler_to_quat(euler, degrees=False):
    if degrees:
        euler = np.array(euler) * np.pi/180
    return np.array(p.getQuaternionFromEuler(euler))

def invert_quat(quat):
    return mat_to_quat(inverse(quat_to_mat(quat)))

def invert_euler(euler):
    return mat_to_euler(inverse(euler_to_mat(euler)))

def inverse(mat):
    return np.linalg.inv(mat)

def uniform_sample_quaternion():
    u1, u2, u3 = np.random.random(3)
    quat = [
        np.sqrt(1-u1)*np.sin(2*np.pi*u2),
        np.sqrt(1-u1)*np.cos(2*np.pi*u2),
        np.sqrt(u1)*np.sin(2*np.pi*u3),
        np.sqrt(u1)*np.cos(2*np.pi*u3),
    ]
    return quat

#quaternion batch*4
def torch_quat_to_mat(quaternion, device):
    '''
    Acknowledgement: Zhou Yi
    https://github.com/papagina/RotationContinuity/blob/master/sanity_test/code/tools.py
    Args:
        quaternion: batch*4, (x, y, z, w)
    '''

    def normalize_vector(v, return_mag=False):
        batch=v.shape[0]
        v_mag = torch.sqrt(v.pow(2).sum(1))# batch
        v_mag = torch.max(v_mag, torch.autograd.Variable(torch.FloatTensor([1e-8]).to(device)))
        v_mag = v_mag.view(batch,1).expand(batch,v.shape[1])
        v = v/v_mag
        if(return_mag==True):
            return v, v_mag[:,0]
        else:
            return v

    batch= quaternion.shape[0]
    quat = normalize_vector(quaternion).contiguous()
    
    qx = quat[...,0].contiguous().view(batch, 1)
    qy = quat[...,1].contiguous().view(batch, 1)
    qz = quat[...,2].contiguous().view(batch, 1)
    qw = quat[...,3].contiguous().view(batch, 1)

    # Unit quaternion rotation matrices computatation
    xx = qx*qx
    yy = qy*qy
    zz = qz*qz
    xy = qx*qy
    xz = qx*qz
    yz = qy*qz
    xw = qx*qw
    yw = qy*qw
    zw = qz*qw

    row0 = torch.cat((1-2*yy-2*zz, 2*xy - 2*zw, 2*xz + 2*yw), 1) #batch*3
    row1 = torch.cat((2*xy+ 2*zw,  1-2*xx-2*zz, 2*yz-2*xw  ), 1) #batch*3
    row2 = torch.cat((2*xz-2*yw,   2*yz+2*xw,   1-2*xx-2*yy), 1) #batch*3

    matrix = torch.cat((row0.view(batch, 1, 3), row1.view(batch,1,3), row2.view(batch,1,3)),1) #batch*3*3

    return matrix.to(device)

def sample_with_diff(theta, delta):
    vec = np.array([0,-np.sin(theta),np.cos(theta)])
    rotm = np.array([[np.cos(delta), -np.sin(delta), 0],
                     [np.sin(delta), np.cos(delta), 0],
                     [0,0,1]])
    vecs = []
    for _ in range(int(np.floor(2*np.pi/delta))):
        vec = rotm @ vec
        vecs.append(vec)
    return vecs

def sample_rot(max_theta, delta_theta):
    thetas = np.linspace(0,max_theta,np.int(max_theta/delta_theta))
    vecs = []
    for theta in thetas:
        if theta == 0:
            vecs.append(np.array([0,0,1]))
            continue
        vecs += sample_with_diff(theta, delta_theta)
    mats = np.array([normal_to_mat(vec) for vec in vecs])
    quats = np.array([R.from_matrix(mat).as_quat() for mat in mats])
    return quats, mats

def sample_rot_roll(max_yaw_pitch, delta_theta, delta_diff):
    thetas = np.linspace(delta_theta,np.pi,np.int(np.pi/delta_theta))
    vecs = []
    for theta in thetas:
        vecs += sample_with_diff(theta, delta_theta)
    
    quats = [np.array([0,0,0,1])]
    for vec in vecs:
        for theta in np.linspace(0.1,3.4,34):
            quat = np.zeros(4)
            quat[:3] = vec * np.sin(theta/2)
            quat[3] = np.cos(theta/2)
            euler = quat_to_euler(quat, degrees=False)
            if abs(euler[0]) <= max_yaw_pitch and abs(euler[1]) <= max_yaw_pitch:
                quats.append(quat)
    for theta in np.linspace(0.1,3.4,34):
        quats.append(euler_to_quat([0,0,theta], degrees=False))
    processed_quats = [quats[0]]
    for quat in quats:
        if (get_quat_diff(np.array(processed_quats), quat) > delta_diff).all():
            processed_quats.append(quat)
    quats = np.array(processed_quats)
    quats = np.array(quats)
    mats = np.array([quat_to_mat(quat) for quat in quats])
    return quats, mats

def get_quat_diff(q1, q2):
    """
    q1: (n,4) or (4)
    q1: (n,4) or (4)
    """
    def normalize(q):
        if len(q.shape) == 1:
            return q / np.linalg.norm(q)
        elif len(q1.shape) == 2:
            return q / np.sqrt(np.sum(q**2, axis=1, keepdims=True))
    q1 = normalize(q1)
    q2 = normalize(q2)
    if len(q1.shape) == 1:
        return np.arccos(2 * ((q1@q2.T)**2) -1)
    elif len(q1.shape) == 2:
        dot = np.sum(q1 * q2, axis=1)
        return np.arccos(2 * (dot**2) -1)

def get_quat_diff_sym_normal(quat_gt, quat_pred, normal, sym_val):
    ori_diff = get_quat_diff(quat_gt, quat_pred)
    if sym_val == 0:
        for theta in range(3,360,3):
            sym_ori = np.empty(4)
            sym_ori[:3] = normal * np.sin(theta/2)
            sym_ori[3] = np.cos(theta/2)
            p1_ori_sym = multiply_quat(sym_ori, quat_gt)
            ori_diff = min( ori_diff, get_quat_diff(p1_ori_sym, quat_pred) )
    elif sym_val != -1:
        for theta in range(0,360,int(sym_val)):
            sym_ori = np.empty(4)
            sym_ori[:3] = normal * np.sin(theta/2)
            sym_ori[3] = np.cos(theta/2)
            p1_ori_sym = multiply_quat(sym_ori, quat_gt)
            ori_diff = min( ori_diff, get_quat_diff(p1_ori_sym, quat_pred) )
    return ori_diff

def get_quat_diff_sym(quat_gt, quat_pred, concav_ori, sym):
    ori_diff = get_quat_diff(quat_gt, quat_pred)
    mat = quat_to_mat(concav_ori)
    normal_x = mat @ np.array([1,0,0])
    normal_y = mat @ np.array([0,1,0])
    normal_z = mat @ np.array([0,0,1])
    ori_diff = min(get_quat_diff_sym_normal(quat_gt, quat_pred, normal_x, sym[0]), ori_diff)
    ori_diff = min(get_quat_diff_sym_normal(quat_gt, quat_pred, normal_y, sym[1]), ori_diff)
    ori_diff = min(get_quat_diff_sym_normal(quat_gt, quat_pred, normal_z, sym[2]), ori_diff)
    ori_diff =  ori_diff * 180/np.pi
    return ori_diff
    