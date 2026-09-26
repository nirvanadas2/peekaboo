import * as THREE from "three";
import { ScrollTrigger } from "gsap/ScrollTrigger";

export default function handleResize(
  renderer: THREE.WebGLRenderer,
  camera: THREE.PerspectiveCamera,
  canvasDiv: React.RefObject<HTMLDivElement | null>
) {
  if (!canvasDiv.current) return;
  const rect = canvasDiv.current.getBoundingClientRect();
  renderer.setSize(rect.width, rect.height);
  camera.aspect = rect.width / rect.height;
  camera.updateProjectionMatrix();
  // Recalculates trigger start/end positions against the new layout without
  // destroying and recreating every ScrollTrigger (which was re-triggering
  // the "target not found" warnings on every resize).
  ScrollTrigger.refresh();
}
