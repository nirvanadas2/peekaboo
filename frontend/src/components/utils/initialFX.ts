import { TextSplitter } from "./textSplitter";
import gsap from "gsap";
import { lenis } from "../Navbar";

export function initialFX() {
  document.body.style.overflowY = "auto";
  if (lenis) {
    lenis.start();
  }
  document.getElementsByTagName("main")[0]?.classList.add("main-active");
  gsap.to("body", {
    backgroundColor: "#07070a",
    duration: 0.5,
    delay: 1,
  });

  const selectors = [".hero-eyebrow", ".hero-title", ".hero-tagline"];
  const elements = selectors.flatMap((selector) =>
    Array.from(document.querySelectorAll(selector))
  );
  const heroText = new TextSplitter(elements, {
    type: "chars,lines",
    linesClass: "split-line",
  });
  gsap.fromTo(
    heroText.chars,
    { opacity: 0, y: 80, filter: "blur(5px)" },
    {
      opacity: 1,
      duration: 1.2,
      filter: "blur(0px)",
      ease: "power3.inOut",
      y: 0,
      stagger: 0.02,
      delay: 0.3,
    }
  );

  gsap.fromTo(
    ".hero-cta",
    { opacity: 0, y: 30 },
    {
      opacity: 1,
      duration: 1.2,
      ease: "power1.inOut",
      y: 0,
      delay: 0.9,
    }
  );

  gsap.fromTo(
    [".header", ".nav-fade"],
    { opacity: 0 },
    {
      opacity: 1,
      duration: 1.2,
      ease: "power1.inOut",
      delay: 0.1,
    }
  );
}
