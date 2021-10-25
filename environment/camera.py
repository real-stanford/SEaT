import numpy as np
import pybullet as p


class SimCameraBase:
    def __init__(self, view_matrix, image_size=(720, 1280), z_near=0.01, z_far=10, focal_length=925.26886):
        # focal_length = 925.26886 came from RealSense.color_intr
        # RealSense.color_intr = (925.26886, 0, 621.80005, 0, 925.1762, 340.30853, 0, 0, 1)
        self.view_matrix = view_matrix
        self.image_size = image_size
        self.z_near = z_near
        self.z_far = z_far
        self.focal_length = focal_length

        fovh = (self.image_size[0] / 2) / self.focal_length
        fovh = 180 * np.arctan(fovh) * 2 / np.pi
        # Notes: 1) FOV is vertical FOV 2) aspect must be float
        aspect_ratio = self.image_size[1] / self.image_size[0]
        self.proj_matrix = p.computeProjectionMatrixFOV(fovh, aspect_ratio, self.z_near, self.z_far)

        self.intrinsics = np.array([
            [self.focal_length, 0, float(self.image_size[1]) / 2],
            [0, self.focal_length, float(self.image_size[0]) / 2],
            [0, 0, 1]
        ])

        self.pose_matrix = np.linalg.inv(np.array(self.view_matrix).reshape(4, 4).T)
        self.pose_matrix[:, 1:3] = -self.pose_matrix[:, 1:3]

    def get_image(self, seg_mask=False, shadows=True):
        if seg_mask:
            img_arr = p.getCameraImage(
                width=self.image_size[1],
                height=self.image_size[0],
                viewMatrix=self.view_matrix,
                projectionMatrix=self.proj_matrix,
                shadow=int(shadows),
                renderer=p.ER_TINY_RENDERER
            )
        else:
            img_arr = p.getCameraImage(
                width=self.image_size[1],
                height=self.image_size[0],
                viewMatrix=self.view_matrix,
                projectionMatrix=self.proj_matrix,
                shadow=int(shadows),
                renderer=p.ER_TINY_RENDERER,
                flags=p.ER_NO_SEGMENTATION_MASK
            )
        w = img_arr[0]
        h = img_arr[1]
        rgb = img_arr[2]
        rgb_arr = np.array(rgb, dtype=np.uint8).reshape([h, w, 4])
        rgb = rgb_arr[:, :, 0:3]

        d = img_arr[3]
        d = np.array(d).reshape([h, w])
        d = (2.0 * self.z_near * self.z_far) \
            / (self.z_far + self.z_near - (2.0 * d - 1.0)
               * (self.z_far - self.z_near))
        
        if seg_mask:
            mask = np.array(img_arr[4]).reshape(h, w)
        else:
            mask = None

        return rgb, d, mask
    
    def get_config(self):
        return {"pose": self.pose_matrix, "intrinsics": self.intrinsics}

class SimCameraYawPitchRoll(SimCameraBase):
    def __init__(
        self,
        target_position, distance, yaw, pitch, roll,
        image_size=(720, 1280), z_near=0.01, z_far=10
    ):
        # setup camera
        view_matrix = p.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=target_position,
            distance=distance,
            yaw=yaw,  # in degrees
            pitch=pitch,
            roll=roll,
            upAxisIndex=2,
        )
        super().__init__(view_matrix, image_size, z_near, z_far)

class SimCameraPosition(SimCameraBase):
    def __init__(
        self,
        eyePosition, targetPosition, upVector=[0,0,1],
        image_size=(720, 1280), z_near=0.01, z_far=10
    ):
        view_matrix = p.computeViewMatrix(eyePosition, targetPosition, upVector)
        super().__init__(view_matrix, image_size=image_size, z_near=z_near, z_far=z_far)