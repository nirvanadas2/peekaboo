# Final validation suite (seeds 20-29)

This suite was registered in PHASE6.md **before** Stage 3 was made
report-only and before Stage 7 was built. It was then run exactly once
on the complete, frozen system (Stages 1-7). The results are locked into
`tests/test_fusion.py::TestFinalSuite`. File hashes at that run:

| File | sha256 (prefix) |
|---|---|
| `statistical_check.py` | `451d373f…` |
| `stego_check.py` | `ec955199…` |
| `behavioral_probe.py` | `1d794457…` |
| `fusion.py` | `67336ad5…` |
| `gate.py` | `2ba5ff5f…` |
| `report.py` | `349f7587…` |

`report.py` changed after the run in one respect only: the figures in
its `LIMITATIONS` text were updated to cite these results. No detection
or scoring code changed.

| Seed | Trigger (3×3, value 6.0) | Target class |
|---|---|---|
| 20 | rows 0-2, cols 5-7 | 2 |
| 21 | rows 5-7, cols 0-2 | 1 |
| 22 | rows 13-15, cols 5-7 | 0 |
| 23 | rows 5-7, cols 13-15 | 2 |
| 24 | rows 2-4, cols 2-4 | 3 |
| 25 | rows 2-4, cols 11-13 | 0 |
| 26 | rows 11-13, cols 2-4 | 3 |
| 27 | rows 11-13, cols 11-13 | 1 |
| 28 | rows 5-7, cols 1-3 | 1 |
| 29 | rows 9-11, cols 13-15 | 0 |

Seeds 20-23 were first written one pixel off, straddling a quadrant
boundary. `TriggerSpec`'s guard rejected them *before* any model was
generated, and they were moved inside the quadrant. No results existed
at that point.

Only `clean`, `noisy`, and `backdoored` are kept. The run's
`steganographic` and `combined` results matched `clean` and `backdoored`
respectively.

**This suite is now spent.**
