import type { PropsWithChildren } from "react";
import "./styles/Hero.css";

const Hero = ({ children }: PropsWithChildren) => {
  return (
    <section className="pk-hero landing-section" id="pk-hero">
      <div className="hero-container">
        <div className="hero-intro">
          <h2 className="hero-eyebrow">AI Model Security Scanner</h2>
          <h1 className="hero-title">
            PEEKABOO
          </h1>
        </div>
        <div className="hero-info">
          <h3 className="hero-tagline">
            Catching what hides in the weights.
          </h3>
        </div>
        <div className="hero-cta">
          <span>Scroll to see the gap</span>
          <div className="hero-cta-line" />
        </div>
      </div>
      {children}
    </section>
  );
};

export default Hero;
