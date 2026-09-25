# Fresh validation suite (seeds 10-19)

This suite was pre-registered in PHASE5.md ("Fixing Stages 3 and 4")
**before** the Stage 3 (noise-aware z) and Stage 4 (init-lattice) fixes
were designed. It was generated in the same environment as
`../benchmark/PROVENANCE.md`.

It was run exactly once, after those fixes and the already-frozen Stage 5
were frozen. File hashes at that run:

| File | sha256 (prefix) |
|---|---|
| `statistical_check.py` | `451d373f…` |
| `stego_check.py` | `ec955199…` |
| `behavioral_probe.py` | `1d794457…` |
| `fusion.py` | `abd49772…` |

Results are locked into `tests/test_fusion.py::TestFreshSuite`.

| Seed | Trigger (3×3, value 6.0) | Target class |
|---|---|---|
| 10 | top-left corner | 1 |
| 11 | top-right corner | 3 |
| 12 | bottom-right corner | 2 |
| 13 | bottom-left corner | 0 |
| 14 | rows 4-6, cols 4-6 | 2 |
| 15 | rows 4-6, cols 9-11 | 0 |
| 16 | rows 9-11, cols 4-6 | 1 |
| 17 | rows 10-12, cols 10-12 | 0 |
| 18 | rows 1-3, cols 5-7 | 3 |
| 19 | rows 12-14, cols 9-11 | 1 |

Only `clean`, `noisy`, and `backdoored` are kept. The run measured
`steganographic` and `combined` too, and their results matched `clean` and
`backdoored` respectively, because the 67-byte payload went undetected
(PHASE3.md §5).

**This suite is now spent.** Any further change to Stages 3-6 must be
validated on a new pre-registered suite.
