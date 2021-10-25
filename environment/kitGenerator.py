import random
from pathlib import Path
from utils import get_split_obj_roots, get_split_file
import numpy as np
import pybullet as p


class KitGenerator:
    urdf_template = """
        <robot name="block">
            <material name="blue">
            <color rgba="0.50 0.50 0.50 1.0"/>
            </material>

            <link name="base_link">
                <inertial>
                    <origin rpy="0 0 0" xyz="0 0 0"/>
                    <mass value="1"/>
                    <inertia ixx="1" ixy="0" ixz="0" iyy="1" iyz="0" izz="1"/>
                </inertial>
            </link>
            {links}
        </robot>
    """

    @staticmethod
    def get_link_string(name, r, p, yaw, x, y, z, vm, cm=None):
        link_template = """
                <link name="{name}">
                <inertial>
                    <origin rpy="{r} {p} {yaw}" xyz="{x} {y} {z}"/>
                    <mass value="1"/>
                    <inertia ixx="1" ixy="0" ixz="0" iyy="1" iyz="0" izz="1"/>
                </inertial>
                <visual>
                    <origin rpy="{r} {p} {yaw}" xyz="{x} {y} {z}"/>
                    <geometry>
                    <mesh filename="{vm}" scale="1 1 1"/>
                    </geometry>
                    <material name="blue"/>
                </visual>
        """.format(name=name, r=r, p=p, yaw=yaw, x=x, y=y, z=z, vm=vm)

        if cm is not None:
            link_template += """
                <collision>
                    <origin rpy="{r} {p} {yaw}" xyz="{x} {y} {z}"/>
                    <geometry>
                    <mesh filename="{cm}" scale="1 1 1"/>
                    </geometry>
                </collision>
            """.format(r=r, p=p, yaw=yaw, x=x, y=y, z=z, cm=cm)

        link_template += """
                </link>

                <joint name="{name}_base_link" type="fixed">
                <origin rpy="0 0 0" xyz="0 0 0"/>
                <parent link="base_link"/>
                <child link="{name}"/>
                </joint>
        """.format(name=name)
        return link_template

    def __init__(self, kit_width = 0.1, dataset_split:str="train") -> None:
        # Generate a kit and a set of objects.
        # What is their relative position and orientation with respect to the kit
        self.kit_width = kit_width

        self.kits_root = Path("assets/kits")
        self.kit_base_path = self.kits_root / "base_0.8.obj"

        self.kit_paths = get_split_obj_roots(get_split_file(dataset_split))
        self.orientations = [0, np.pi / 2, np.pi, 3 * np.pi / 2]

    @staticmethod
    def dump_urdf_from_links(urdf_path:Path, links:str):
        urdf = KitGenerator.urdf_template.format(links=links)
        if not urdf_path.parent.exists():
            urdf_path.parent.mkdir(parents=True, exist_ok=True)
        with open(urdf_path, "w") as f:
            f.write(urdf)
        
    def create_link(self, kit_path, x, y, z, roll, pitch, yaw, link_name:str, collision_mesh:bool=True):
        links = str()
        links += KitGenerator.get_link_string(
            name=f"{link_name}__base",
            r=roll, p=pitch, yaw=yaw,
            x=x, y=y, z=z,
            vm = self.kit_base_path,
            cm = self.kit_base_path if collision_mesh else None
        )
        for i, kit_part_path in enumerate((kit_path / "kit_parts").glob("*.obj")):
            links += KitGenerator.get_link_string(
                name=f"{link_name}__{i}",
                r=roll, p=pitch, yaw=yaw,
                x=x, y=y, z=z,
                vm = kit_part_path,
                cm = kit_part_path if collision_mesh else None
            )
        obj_detail = dict()
        obj_detail["obj_path"] = kit_path / "obj.obj"
        obj_detail["path"] = kit_path / "obj.urdf"
        obj_detail["position"] = np.array([x, y, z])
        obj_detail["orientation"] = np.array([roll, pitch, yaw])
        return links, obj_detail

    def generate_random_kit(self, urdf_path: Path, nrows=2, ncols=2):
        tl = (- self.kit_width * ncols / 2, self.kit_width * nrows / 2)
        links = ""
        obj_details = list()
        for row in range(nrows):
            for col in range(ncols):
                x = tl[0] + self.kit_width / 2 + col * self.kit_width
                y = tl[1] - self.kit_width / 2 - row * self.kit_width
                yaw = random.sample(self.orientations, 1)[0]
                # Randomly sample a mesh
                kit_path = random.sample(self.kit_paths, 1)[0]
                link, obj_detail = self.create_link(kit_path, x, y, 0, 0, 0, yaw, f"{row}_{col}")
                links += link
                obj_details.append(obj_detail)
        self.dump_urdf_from_links(urdf_path, links)
        return obj_details

    def generate_random_kit_3d_one_plate(self, urdf_path: Path):
        """
        one unit kit randomly oriented in 3D
        """
        l = 0.2
        kit_path = random.sample(self.kit_paths, 1)[0]
        # Random quaternion in upper hemisphere
        xy_theta = random.random() * 2 * np.pi
        x = 2 * (random.random() - 0.5) * np.cos(xy_theta)
        y = 2 * (random.random() - 0.5) * np.sin(xy_theta)
        z = np.sqrt(1 - x**2 - y**2)
        theta = random.random() * 2 * np.pi
        kit_ori = np.array([
            x * np.sin(theta / 2),
            y * np.sin(theta / 2),
            z * np.sin(theta / 2),
            np.cos(theta / 2),
        ])
        kit_ori = p.getEulerFromQuaternion(kit_ori)
        links = ""
        obj_details = list()
        kit_position = np.array([0, 0, (l / np.sqrt(2))])
        link, obj_detail = self.create_link(kit_path, *kit_position, *kit_ori, f"kit")
        links += link
        obj_details.append(obj_detail)
        self.dump_urdf_from_links(urdf_path, links)
        return obj_details

    def generate_random_kit_3d_five_plates(self, urdf_path: Path, collision_mesh:bool=True):
        obj_details = list()
        links = ""

        # Choose a base plate randomly. Origin will be centered at it.
        base_kit = random.sample(self.kit_paths, 1)[0]
        link, obj_detail = self.create_link(base_kit, 0, 0, 0, 0, 0, 0, "baseKit", collision_mesh=collision_mesh)
        links += link
        obj_details.append(obj_detail)
        # Choose number of sides between [1, 4]. 
        n_plates = random.randrange(1, 5)
        side_indices = random.sample(range(4), n_plates)
        sides = np.array([[1, 0], [0, 1], [-1, 0], [0, -1]])
        for side_index in side_indices:
            side = sides[side_index]
            # For each side, choose an angle.
            theta = random.uniform(np.pi / 4, np.pi / 2)
            z = -self.kit_width * np.cos(theta) / 2
            x, y = side * (self.kit_width / 2 + self.kit_width * np.sin(theta) / 2)
            yaw = 0 
            pitch, roll = side * (np.pi / 2 - theta)
            roll *= -1
            kit = random.sample(self.kit_paths, 1)[0]
            link, obj_detail = self.create_link(kit, x, y, z, roll, pitch, yaw, f"kit_{side_index}", collision_mesh=collision_mesh)
            links += link
            obj_details.append(obj_detail)
        self.dump_urdf_from_links(urdf_path, links)
        return obj_details

if __name__ == "__main__":
    import sys
    from pathlib import Path
    root_path = Path(__file__).parent.parent.absolute()
    # print("Adding to python path: ", root_path)
    sys.path = [str(root_path)] + sys.path
    from time import sleep
    import pybullet as p
    import pybullet_data
    p.connect(p.GUI)
    p.setRealTimeSimulation(1)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.8)
    p.setPhysicsEngineParameter(enableFileCaching=0)
    p.loadURDF("plane.urdf")

    kg = KitGenerator()
    kit_path = Path("/tmp/random_kit")
    one_plate_kit_path = Path("/tmp/random_kit_one")
    five_plate_kit_path = Path("/tmp/random_kit_five")
    while True:
        ids = list()
        # random kit 2d
        kg.generate_random_kit(kit_path)
        ids.append(p.loadURDF(str(kit_path), basePosition=(-0.3, 0, 0)))
        # random kit 3d one plate
        kg.generate_random_kit_3d_one_plate(one_plate_kit_path)
        ids.append(p.loadURDF(str(one_plate_kit_path), basePosition=(0, 0, 0)))
        # random kit 3d five plates
        kg.generate_random_kit_3d_five_plates(five_plate_kit_path)
        ids.append(p.loadURDF(str(five_plate_kit_path), basePosition=(0.3, 0, 0)))

        sleep(3)        
        for id in ids:
            p.removeBody(id)
