import { useEffect } from "react";
import { ScrollTrigger } from "gsap/ScrollTrigger";
import { gsap } from "gsap";
import Lenis from "lenis";
import { Link } from "react-router-dom";
import HoverLinks from "./HoverLinks";
import "./styles/Navbar.css";

gsap.registerPlugin(ScrollTrigger);
export let lenis: Lenis | null = null;

const NAV_LINKS = [
  { label: "THE GAP", href: "#pk-problem" },
  { label: "PILLARS", href: "#pk-pillars" },
  { label: "ARCHITECTURE", href: "#pk-architecture" },
];

const Navbar = () => {
  useEffect(() => {
    lenis = new Lenis({
      duration: 1.7,
      easing: (t) => Math.min(1, 1.001 - Math.pow(2, -10 * t)),
      orientation: "vertical",
      gestureOrientation: "vertical",
      smoothWheel: true,
      wheelMultiplier: 1.7,
      touchMultiplier: 2,
      infinite: false,
    });

    // Start paused - initialFX() releases it once the intro reveal begins.
    lenis.stop();

    function raf(time: number) {
      lenis?.raf(time);
      requestAnimationFrame(raf);
    }
    requestAnimationFrame(raf);

    const links = document.querySelectorAll(".header ul a");
    links.forEach((elem) => {
      const element = elem as HTMLAnchorElement;
      element.addEventListener("click", (e) => {
        if (window.innerWidth > 1024) {
          e.preventDefault();
          const section = element.getAttribute("data-href");
          if (section && lenis) {
            const target = document.querySelector(section) as HTMLElement;
            if (target) {
              lenis.scrollTo(target, {
                offset: 0,
                duration: 1.5,
              });
            }
          }
        }
      });
    });

    window.addEventListener("resize", () => {
      lenis?.resize();
    });

    return () => {
      lenis?.destroy();
    };
  }, []);

  return (
    <>
      <div className="header">
        <Link to="/" className="navbar-title">
          PEEKABOO
        </Link>
        <ul>
          {NAV_LINKS.map((link) => (
            <li key={link.href}>
              <a data-href={link.href} href={link.href}>
                <HoverLinks text={link.label} />
              </a>
            </li>
          ))}
          <li>
            <Link to="/dashboard" className="navbar-cta">
              Open Scanner
            </Link>
          </li>
        </ul>
      </div>

      <div className="nav-fade"></div>
    </>
  );
};

export default Navbar;
