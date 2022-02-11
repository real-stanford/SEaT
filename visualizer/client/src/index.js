import * as THREE from "three";
import {
    OrbitControls
} from "three/examples/jsm/controls/OrbitControls";
import {
    OBJLoader
} from "three/examples/jsm/loaders/OBJLoader";
import {
    TransformControls
} from "three/examples/jsm/controls/TransformControls.js";
import {
    PLYLoader
} from "three/examples/jsm/loaders/PLYLoader";

function resizeRendererToDisplaySize(renderer) {
    const canvas = renderer.domElement;
    const width = canvas.clientWidth;
    const height = canvas.clientHeight;
    const needResize = canvas.width != width || canvas.height != height;
    if (needResize) {
        renderer.setSize(width, height, false);
    }
    return needResize;
}

const DEMO = true;

function main() {
    THREE.Object3D.DefaultUp = new THREE.Vector3(0, 0, 1);

    const canvas = document.querySelector("#three_canvas");
    const renderer = new THREE.WebGLRenderer({ canvas });

    const fov = 45;
    const aspect = 2
    const near = 0.1
    const far = 100;
    const camera = new THREE.PerspectiveCamera(fov, aspect, near, far);
    camera.up.set(0, 0, 1);
    camera.position.set(-1.21, -0.21, 0.66);
    camera.setRotationFromEuler(new THREE.Euler(0.12271172332001311, -0.6687206034262742, -1.374432550111837));

    const orbitControl = new OrbitControls(camera, canvas);
    orbitControl.target.set(0, 0, 0);
    orbitControl.update();

    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x66645d);

    // transform orbitControl
    const transformControls = new TransformControls(camera, canvas);
    transformControls.addEventListener("dragging-changed", function (event) {
        orbitControl.enabled = !event.value;
    });

    // const ROTATION_DELTA = MATH.PI / 2;
    // function ensureAngle(angle) {
    //     // bring the angle between -pi and pi
    //     return ((angle % TPI) + TPI) % TPI - MATH.PI
    // }

    transformControls.addEventListener("change", function (event) {
        var object = transformControls.object;
        if (workspace_bounds_min && workspace_bounds_max && object) {
            object.position.clamp(workspace_bounds_min, workspace_bounds_max);
        }
        // if (object) {
        //     const initial_yaw = 0;
        //     console.log(object.rotation);
        //     const yaw = ensureAngle(object.rotation.z);
        //     if (yaw > initial_yaw + ROTATION_DELTA) {
        //         object.rotation.z = ROTATION_MIN;
        //     }
        //     if (object.rotation.z < 0 && object.rotation.z < ROTATION_MAX) {
        //         object.rotation.z = ROTATION_MAX;
        //     }
        // }
    })
    scene.add(transformControls);
    let workspace_bounds_min;
    let workspace_bounds_max;


    const ambiLight = new THREE.AmbientLight(0xf0f0f0)
    scene.add(ambiLight);
    const light = new THREE.SpotLight(0xffffff, 0.5);
    light.position.set(0, 0, 200);
    light.angle = Math.PI * 0.2;
    light.castShadow = true;
    light.shadow.camera.near = 200;
    light.shadow.camera.far = 2000;
    light.shadow.bias = - 0.000222;
    light.shadow.mapSize.width = 1024;
    light.shadow.mapSize.height = 1024;
    scene.add(light);

    let objects_meshes = Array();
    let objects_metada = Array();

    let colors = ['gray', 'brown', 'orange', 'blue'];
    const materials = []
    for (var i = 0; i < colors.length; i++) {
        materials.push(
            new THREE.MeshPhongMaterial({
                color: colors[i],
            })
        );
    }
    let color_index = 0;
    function obj_laoder_callback(obj_mesh, obj_data) {
        console.log("object mesh recieved: ", obj_mesh, obj_data, typeof (obj_mesh));
        const orientation = new THREE.Quaternion(
            obj_data["orientation"][0],
            obj_data["orientation"][1],
            obj_data["orientation"][2],
            obj_data["orientation"][3],
        );
        obj_mesh.setRotationFromQuaternion(orientation);
        if ("color" in obj_data) {
            const color = new THREE.Color(obj_data["color"][0], obj_data["color"][1], obj_data["color"][2]);
            const material = new THREE.MeshPhongMaterial({ color: color });
            obj_mesh.traverse(function (child) {
                if (child.isMesh) {
                    child.material = material;
                }
            });

        }
        const p = obj_data["position"]
        obj_mesh.position.set(p[0], p[1], p[2]);

        let scale = [1, 1, 1];
        if ("scale" in obj_data) {
            scale = obj_data.scale;
        }
        obj_mesh.scale.x = scale[0];
        obj_mesh.scale.y = scale[1];
        obj_mesh.scale.z = scale[2];

        // obj_mesh.
        console.log(`Applied transformation: ${p}, ${orientation} to object ${obj_data["name"]}`);

        scene.add(obj_mesh);

        color_index = (color_index + 1) % materials.length;
        if (obj_data["name"] != "kit") {
            // This check is to avoid transformControls for kit
            objects_metada.push(obj_data);
            objects_meshes.push(obj_mesh);
            transformControls.attach(obj_mesh);
            last_active_object = obj_mesh;
        }
    }

    // load the scene json file
    let base_url = "scenes/json_scene/"
    let scene_json_url = `${base_url}scene.json`
    fetch(scene_json_url)
        .then(response => response.json())
        .then(data => {
            let objects = data["objects"];
            if (objects) {
                for (var i = 0; i < objects.length; i++) {
                    // if (i == objects.length - 1)
                    //     break;
                    let obj_data = objects[i]
                    let objectPath = `scenes/json_scene/${obj_data["path"]}`
                    console.log(`Loading object from ${objectPath}`);
                    const loader = new OBJLoader();
                    loader.load(
                        objectPath,
                        (obj_mesh) => {
                            obj_laoder_callback(obj_mesh, obj_data)
                        },
                        null, null, null
                    )
                }
            }
            if (data["bounds"]) {
                let translation_bounds = data["bounds"]["translation"];
                workspace_bounds_min = new THREE.Vector3(
                    Math.min(translation_bounds[0][0], translation_bounds[1][0]),
                    Math.min(translation_bounds[0][1], translation_bounds[1][1]),
                    Math.min(translation_bounds[0][2], translation_bounds[1][2])
                );
                workspace_bounds_max = new THREE.Vector3(
                    Math.max(translation_bounds[0][0], translation_bounds[1][0]),
                    Math.max(translation_bounds[0][1], translation_bounds[1][1]),
                    Math.max(translation_bounds[0][2], translation_bounds[1][2])
                );
            }
        });

    const loader = new PLYLoader();
    let obj_pc;
    let kit_pc;
    loader.load('scenes/json_scene/scene_KIT_pcl.ply', function (geometry) {
        const material = new THREE.PointsMaterial({ size: 0.01, vertexColors: true });
        obj_pc = new THREE.Points(geometry, material);
        scene.add(obj_pc);
    });
    loader.load('scenes/json_scene/scene_OBJECTS_pcl.ply', function (geometry) {
        const material = new THREE.PointsMaterial({ size: 0.01, vertexColors: true });
        kit_pc = new THREE.Points(geometry, material);
        scene.add(kit_pc);
    });

    function transformControlSetRotate() {
        transformControls.setMode("rotate");
        // if (DEMO) {
        //     transformControls.showX = false;
        //     transformControls.showY = false;
        // }
    }

    function transformControlSetTranslate() {
        transformControls.setMode("translate");
        // if (DEMO) {
        //     transformControls.showX = true;
        //     transformControls.showY = true;
        // }
    }
    function keyDownHandler(event) {
        switch (event.keyCode) {
            case 81: // Q
                if (!DEMO) {
                    transformControls.setSpace(transformControls.space === "local" ? "world" : "local");
                }
                break;
            case 87: // W
                transformControlSetTranslate();
                if (setTransformButton) {
                    setTransformButton.innerHTML = rotation_str;
                }
                break;
            case 69: // E
                transformControlSetRotate();
                if (setTransformButton) {
                    setTransformButton.innerHTML = translation_str;
                }
                break;
            case 187:
            case 107: // +, =, num+
                transformControls.setSize(transformControls.size + 0.1);
                break;

            case 189:
            case 109: // -, _, num-
                transformControls.setSize(Math.max(transformControls.size - 0.1, 0.1));
                break;
        }
    }


    let last_active_object = null;
    function clickHandler(event) {
        let mouse = new THREE.Vector2(
            (event.clientX / window.innerWidth) * 2 - 1,
            -(event.clientY / window.innerHeight) * 2 + 1
        );
        raycaster.setFromCamera(mouse, camera);
        let intersections = raycaster.intersectObjects(objects_meshes, true);
        if (intersections.length > 0) {
            last_active_object = intersections[0].object.parent;
            transformControls.attach(last_active_object);
        }
    };

    function onPointerDown(event) {
        onDownPosition.x = event.clientX;
        onDownPosition.y = event.clientY;
    }

    function onPointerUp(event) {
        onUpPosition.x = event.clientX;
        onUpPosition.y = event.clientY;
        if (onDownPosition.distanceTo(onUpPosition) === 0) {
            console.log("Detaching")
            transformControls.detach();
        }
    }

    function onPointerMove(event) {
        pointer.x = (event.clientX / window.innerWidth) * 2 - 1;
        pointer.y = - (event.clientY / window.innerHeight) * 2 + 1;
        raycaster.setFromCamera(pointer, camera);
        const intersects = raycaster.intersectObjects(objects_meshes);
        if (intersects.length > 0) {
            const object = intersects[0].object;
            if (object !== transformControls.object) {
                transformControls.attach(object);
            }
        }
    }
    const raycaster = new THREE.Raycaster();
    const pointer = new THREE.Vector2();
    const onUpPosition = new THREE.Vector2();
    const onDownPosition = new THREE.Vector2();
    document.addEventListener('pointerdown', onPointerDown, false);
    document.addEventListener('pointerup', onPointerUp, false);
    document.addEventListener('pointermove', onPointerMove, false);
    window.addEventListener("keydown", keyDownHandler);
    document.addEventListener("click", clickHandler);

    function get_scene_json() {
        let scene_json = new Object();
        scene_json.objects = new Array();
        for (var i = 0; i < objects_meshes.length; i++) {
            let obj_mesh = objects_meshes[i];
            let obj_json = new Object();
            obj_json = Object.assign(obj_json, objects_metada[i]);
            let position = obj_mesh.children[0].getWorldPosition(new THREE.Vector3());
            obj_json.position = [position.x, position.y, position.z];
            let quaternion = obj_mesh.children[0].getWorldQuaternion(new THREE.Quaternion());
            obj_json.orientation = [quaternion.x, quaternion.y, quaternion.z, quaternion.w];
            scene_json.objects.push(obj_json);
        }
        return JSON.stringify(scene_json);
    }

    function send_json_request(url, jsonStr) {
        let myHeaders = new Headers();
        myHeaders.append("Content-Type", "application/json");
        let requestOptions = {
            method: 'POST',
            headers: myHeaders,
            body: jsonStr,
            redirect: 'follow'
        };
        return fetch(url, requestOptions).then(response => response.text());

    }
    let button = document.querySelector("#upload_scene")
    if (button) {
        button.onclick = function () {
            if (confirm("Do you want to upload the scene?")) {
                let raw = get_scene_json();
                send_json_request("http://localhost:52000/upload_scene", raw)
                    .then(result => {
                        console.log(result);
                        button.parentNode.removeChild(button);
                        alert("Scene uploaded!");
                    })
                    .catch(error => console.log('error', error));
            }
        };

    }

    // let refreshButton = document.querySelector("#refresh_scene");
    // refreshButton.onclick = function() {
    //     let jsonStr = JSON.stringify({"refresh": "true"})
    //     send_json_request("http://128.59.17.212:5000/refresh_scene", jsonStr)
    //     .then(result => console.log(result))
    //     .catch(error => console.log('error', error));
    // };

    let togglePcButton = document.querySelector("#toggle_point_cloud");
    function togglePC(pc) {
        if (pc == undefined) {
            return;
        }

        pc.traverse(function (child) {
            if (child instanceof THREE.Points) {
                if (child.visible)
                    child.visible = false;
                else
                    child.visible = true;
            }
        });
    }
    if (togglePcButton) {
        togglePcButton.onclick = function () {
            console.log("Toggling pointclouds");
            togglePC(kit_pc);
            togglePC(obj_pc);
        };
    }

    let setTransformButton = document.querySelector("#set_transform");
    let translation_str = "Turn translation ON"
    let rotation_str = "Turn rotation ON"
    setTransformButton.onclick = function () {
        console.log("Toggling transformations");
        if (setTransformButton.innerHTML == translation_str) {
            // turn on translation
            transformControlSetTranslate();
            setTransformButton.innerHTML = rotation_str;
        } else {
            // turn on rotation
            transformControlSetRotate();
            setTransformButton.innerHTML = translation_str;
        }
        if (last_active_object != null) {
            transformControls.attach(last_active_object);
        }
    };

    let setFrameButton = document.querySelector("#set_transform_frame");
    let obj_frame_str = "Turn local frame ON"
    let world_frame_str = "Turn world frame ON"
    setFrameButton.onclick = function () {
        console.log("Toggling transform frame");
        // transformControls.setSpace(transformControls.space === "local" ? "world" : "local");
        if (setFrameButton.innerHTML == obj_frame_str) {
            // turn on obj frame
            transformControls.setSpace("local");
            setFrameButton.innerHTML = world_frame_str;
        } else {
            // turn on world frame
            transformControls.setSpace("world");
            setFrameButton.innerHTML = obj_frame_str;
        }
        if (last_active_object != null) {
            transformControls.attach(last_active_object);
        }
    };

    function render() {
        if (resizeRendererToDisplaySize(renderer)) {
            const canvas = renderer.domElement;
            camera.aspect = canvas.clientWidth / canvas.clientHeight;
            camera.updateProjectionMatrix();
        }

        renderer.render(scene, camera);
        requestAnimationFrame(render);
    }
    requestAnimationFrame(render);
}
main();
