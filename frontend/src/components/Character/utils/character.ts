import * as THREE from "three";
import { DRACOLoader, GLTFLoader } from "three-stdlib";
import type { GLTF } from "three-stdlib";
import { setCharTimeline } from "../../utils/GsapScroll";

const setCharacter = (
  renderer: THREE.WebGLRenderer,
  scene: THREE.Scene,
  camera: THREE.PerspectiveCamera
) => {
  const loader = new GLTFLoader();
  const dracoLoader = new DRACOLoader();
  dracoLoader.setDecoderPath("/draco/");
  loader.setDRACOLoader(dracoLoader);

  const loadCharacter = () => {
    return new Promise<GLTF | null>((resolve, reject) => {
      let character: THREE.Object3D;
      loader.load(
        "/models/character.glb",
        async (gltf) => {
          character = gltf.scene;
          await renderer.compileAsync(character, camera, scene);
          character.traverse((child: any) => {
            if (child.isMesh) {
              const mesh = child as THREE.Mesh;
              child.castShadow = false;
              child.receiveShadow = false;
              mesh.frustumCulled = true;
              if (mesh.material && !Array.isArray(mesh.material)) {
                (mesh.material as THREE.ShaderMaterial).precision = "mediump";
              }
            }
          });
          resolve(gltf);
          setCharTimeline(character, camera);
          character!.getObjectByName("footR")!.position.y = 3.36;
          character!.getObjectByName("footL")!.position.y = 3.36;
          dracoLoader.dispose();
        },
        undefined,
        (error) => {
          console.error("Error loading GLTF model:", error);
          reject(error);
        }
      );
    });
  };

  return { loadCharacter };
};

export default setCharacter;
