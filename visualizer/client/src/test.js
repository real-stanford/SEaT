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

function main() {
    // renderer
    const canvas = document.querySelector("#three_canvas")
    const renderer = new THREE.WebGLRenderer({
        canvas
    });

    // camera
    const camera = new THREE.PerspectiveCamera(75, 2, 0.1, 100);
    camera.position.set(0, 3, 3);

    // control
    const orbitControl = new OrbitControls(camera, canvas);
    orbitControl.target.set(0, 0, 0);
    orbitControl.update();

    // scene
    const scene = new THREE.Scene();
    scene.add(new THREE.GridHelper(5, 50));

    // transform controls
    const transformControls = new TransformControls(camera, canvas);
    transformControls.addEventListener("dragging-changed", function (event) {
        orbitControl.enabled = !event.value;
        // console.log(transformControls.object, transformControls.object.position);
    });
    scene.add(transformControls);

    // lights
    {
        const light = new THREE.HemisphereLight();
        scene.add(light);
    } {
        const light = new THREE.DirectionalLight();
        light.position.set(0, 10, 0);
        light.target.position.set(0, 0, 0);
        scene.add(light);
        scene.add(light.target);
    }

    // objects
    let objects = [];
    let colors = ["red", "green", "blue"];
    let positions = [
        [0, 0, -0.2],
        [0, 0, 0],
        [0, 0, 0.2],
    ]
    let mesh_index = 0;
    let files = [
        "static/scenes/json_scene/toteA.L.obj",
        "static/scenes/json_scene/4.obj",
        "static/scenes/json_scene/5.obj"
    ]

    function callbackOnLoad(obj) {
        let material = new THREE.MeshBasicMaterial({color:colors[mesh_index]});
        obj.traverse( function ( child ) {
            if (child.isMesh) {
                child.material = material;
            }
        });
        const p = positions[mesh_index];
        obj.position.set(p[0], p[1], p[2]);

        scene.add(obj);
        objects.push(obj);
        mesh_index++;
    }

    for (var i=0; i<files.length; i++) {
        const objLoader = new OBJLoader();
        objLoader.load(files[i], callbackOnLoad, null, null, null);
    }

    document.addEventListener("click", function(event) {
        let mouse = new THREE.Vector2(
            (event.clientX / window.innerWidth) * 2 - 1,
            -(event.clientY / window.innerHeight) * 2 + 1
        );
        let raycaster = new THREE.Raycaster();
        raycaster.setFromCamera(mouse, camera);
        let intersections = raycaster.intersectObjects(objects, true);
        if (intersections.length > 0) {
            let object = intersections[0].object;
            transformControls.attach(object);
        } 
    });

    function render() {
        if (resizeRendererToDisplaySize(renderer)) {
            const canvas = renderer.domElement;
            camera.aspect = canvas.clientWidth / canvas.clientHeight;
            camera.updateProjectionMatrix();
        }

        for (var i=0; i<objects.length; i++) {
            let p = objects[i].children[0].position; // xxx: hacky but works
            console.log(`-- ${p.x} ${p.y} ${p.z}`);
        }
        console.log("======");
        renderer.render(scene, camera);
        requestAnimationFrame(render);
    }
    requestAnimationFrame(render);

}
main();