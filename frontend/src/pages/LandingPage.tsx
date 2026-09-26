import { useEffect, useState } from "react";
import Navbar from "../components/Navbar";
import Hero from "../components/sections/Hero";
import ProblemStatement from "../components/sections/ProblemStatement";
import CoverageGapTable from "../components/sections/CoverageGapTable";
import DetectionPillars from "../components/sections/DetectionPillars";
import ArchitectureDiagram from "../components/sections/ArchitectureDiagram";
import ClosingCTA from "../components/sections/ClosingCTA";
import setSplitText from "../components/utils/splitText";
import { setSectionTimelines } from "../components/utils/GsapScroll";
import CharacterModel from "../components/Character";

const LandingPage = () => {
  const [shouldRenderCharacter, setShouldRenderCharacter] = useState(false);
  const [isDesktopView, setIsDesktopView] = useState(
    () => typeof window !== "undefined" && window.innerWidth > 1024
  );

  useEffect(() => {
    const resizeHandler = () => {
      setSplitText();
      setIsDesktopView(window.innerWidth > 1024);
    };
    resizeHandler();
    window.addEventListener("resize", resizeHandler);
    return () => window.removeEventListener("resize", resizeHandler);
  }, []);

  useEffect(() => {
    if (window.innerWidth <= 1024) return;

    let timeoutId: ReturnType<typeof setTimeout> | undefined;
    let idleId: number | undefined;
    const win = window as Window & {
      requestIdleCallback?: (
        callback: IdleRequestCallback,
        options?: IdleRequestOptions
      ) => number;
      cancelIdleCallback?: (handle: number) => void;
    };

    const mountCharacter = () => setShouldRenderCharacter(true);

    if (typeof win.requestIdleCallback === "function") {
      idleId = win.requestIdleCallback(mountCharacter, { timeout: 1500 });
    } else {
      timeoutId = setTimeout(mountCharacter, 1200);
    }

    return () => {
      if (idleId !== undefined && typeof win.cancelIdleCallback === "function") {
        win.cancelIdleCallback(idleId);
      }
      if (timeoutId !== undefined) clearTimeout(timeoutId);
    };
  }, []);

  useEffect(() => {
    const timer = setTimeout(() => {
      import("../components/utils/initialFX").then(({ initialFX }) => {
        initialFX();
      });
    }, 300);
    return () => clearTimeout(timer);
  }, []);

  // Independent of the character rig / device, so coverage table, pillar,
  // and architecture-diagram reveals still run on mobile or if the 3D
  // character never mounts.
  useEffect(() => {
    setSectionTimelines();
  }, []);

  return (
    <div className="container-main">
      <Navbar />
      {isDesktopView && shouldRenderCharacter && <CharacterModel />}
      <Hero />
      <ProblemStatement />
      <CoverageGapTable />
      <DetectionPillars />
      <ArchitectureDiagram />
      <ClosingCTA />
    </div>
  );
};

export default LandingPage;
