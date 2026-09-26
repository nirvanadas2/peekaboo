import * as THREE from "three";
import gsap from "gsap";

// Character choreography across Hero + Problem Statement ("The Gap"),
// mirroring the source rig's side-by-side pattern: the character (position:
// fixed) rotates/pulls back out of the hero's centered pose and settles into
// the right-hand column reserved by .problem-character-space, staying
// visible next to the Problem Statement text, then fades out (opacity +
// pointer-events:none) near the end of that section's own scroll range so it
// never reaches the sections below. Two scroll-scrubbed timelines, handing off
// rather than one continuous one, keeps each independently correct against
// its own trigger element.
export function setCharTimeline(
  character: THREE.Object3D<THREE.Object3DEventMap> | null,
  camera: THREE.PerspectiveCamera
) {
  if (!character || window.innerWidth <= 1024) return;

  // The rig's monitor/screen-prop meshes are visible by default in the GLB.
  // The source site only ever reveals them via a scroll-tied tween (removed
  // here along with tl2/tl3, since that choreography belonged to a section
  // structure we no longer have) - without this, they render as an opaque
  // plane sitting in front of the character from frame one. Keep them
  // permanently hidden instead, matching how the character rests off-screen
  // of any monitor reveal in this simplified hero-only choreography.
  character.children.forEach((object: any) => {
    if (object.name === "Plane004") {
      object.children.forEach((child: any) => {
        child.material.transparent = true;
        child.material.opacity = 0;
      });
    }
    if (object.name === "screenlight") {
      object.material.transparent = true;
      object.material.opacity = 0;
    }
  });

  const tl1 = gsap.timeline({
    scrollTrigger: {
      trigger: ".pk-hero",
      start: "top top",
      end: "bottom top",
      scrub: true,
      invalidateOnRefresh: true,
    },
  });

  tl1
    .fromTo(character.rotation, { y: 0 }, { y: 0.7, duration: 1 }, 0)
    .to(camera.position, { z: 22 }, 0)
    .fromTo(".character-model", { x: 0 }, { x: "32%", duration: 1 }, 0)
    .to(".hero-container", { opacity: 0, duration: 0.4 }, 0)
    .to(".hero-container", { y: "40%", duration: 0.8 }, 0)
    .fromTo(".pk-problem", { y: "-50%" }, { y: "0%" }, 0);

  // Holds the character in the reserved right-hand column through Problem
  // Statement's own scroll range, then fades it out before the next
  // section begins.
  const tl2 = gsap.timeline({
    scrollTrigger: {
      trigger: ".pk-problem",
      start: "top bottom",
      end: "bottom top",
      scrub: true,
      invalidateOnRefresh: true,
    },
  });

  tl2.to(
    ".character-model",
    { opacity: 0, pointerEvents: "none", duration: 0.25 },
    0.75
  );
}

// Reveal timelines for the data-heavy sections (pillars, architecture
// diagram) - same stagger/scrub technique as the source rig's
// career-timeline reveal, applied to Peekaboo's own content blocks. Runs
// independently of the character rig, so it works on every device.
export function setSectionTimelines() {
  const pillarsTimeline = gsap.timeline({
    scrollTrigger: {
      trigger: ".pk-pillars",
      start: "top 65%",
      end: "bottom 45%",
      scrub: 1.2,
      invalidateOnRefresh: true,
    },
  });
  pillarsTimeline.fromTo(
    ".pillar-card",
    { opacity: 0, y: 60, scale: 0.94 },
    { opacity: 1, y: 0, scale: 1, stagger: 0.2, duration: 1, ease: "power2.out" },
    0
  );

  // One-shot reveal rather than scrubbed: the diagram is interactive, so
  // every box must be fully visible whenever it's on screen. A scrubbed
  // stagger left later boxes half-faded until the page bottom.
  const archTimeline = gsap.timeline({
    scrollTrigger: {
      trigger: ".arch-canvas",
      start: "top 80%",
      toggleActions: "play none none none",
    },
  });
  archTimeline
    .fromTo(
      ".arch-node",
      { opacity: 0.15 },
      { opacity: 1, stagger: 0.06, duration: 0.5, ease: "power1.out" },
      0
    )
    // Edges use pathLength=1, so a dashoffset of 1 hides the whole curve.
    .fromTo(
      ".arch-edge",
      { strokeDashoffset: 1 },
      { strokeDashoffset: 0, stagger: 0.04, duration: 0.6, ease: "power1.out" },
      0
    );
}
