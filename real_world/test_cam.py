import numpy as np
from utils.tsdfHelper import TSDFHelper
import matplotlib.pyplot as plt
from real_world.pyphoxi import PhoXiSensor
import cv2

tcp_ip = "127.0.0.1"
tcp_port = 50200
bin_cam = PhoXiSensor(tcp_ip, tcp_port)
bin_cam.start()
camera_pose = bin_cam._extr
camera_color_intr = bin_cam._intr

_, gray, d = bin_cam.get_frame(True)
rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
# plt.imshow(rgb)
# plt.savefig('test_rgb.jpg')
# plt.imshow(d, cmap='jet')
# plt.savefig('test_d.jpg')

cv2.morphologyEx(
            d, cv2.MORPH_CLOSE, cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (10, 10)), iterations=5)
views = [(rgb, d, camera_color_intr, camera_pose)]
voxel_size = 0.002
crop_bounds = np.array([
            [-0.3, 0.3],
            [-0.7, -0.4],
            [0, 0.1]
        ])
sc_inp = TSDFHelper.tsdf_from_camera_data(views, crop_bounds, voxel_size)
print(sc_inp.shape)
TSDFHelper.to_mesh(sc_inp, 'real_world/test_multi.obj', voxel_size)