import type { PillarScore } from "../types/report";

// A pillar "drives" the verdict when its score equals the overall score --
// true by construction of fusion's max-over-pillars (peekaboo/pipeline/fusion.py).
// Excluded when overall_score is 0: with every scored pillar clean, nothing
// is "driving" a HIGH score there is to explain.
export function isDrivingPillar(pillar: PillarScore, overallScore: number): boolean {
  return pillar.status !== "not_run" && overallScore > 0 && pillar.score === overallScore;
}
