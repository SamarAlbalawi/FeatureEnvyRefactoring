# Selected aggregate research results

These figures summarize existing results; they do not represent a new run.
The outcome unit below is an applied nonempty patch in a particular round.
There were 35 eligible cases, 105 matched pairs per round, two rounds, and
66 applied nonempty patches: 30 in Round 1 and 36 in Round 2.

[EVALUATION_EVIDENCE.csv](EVALUATION_EVIDENCE.csv) contains the existing
patch-level classifications, with rounds kept separate. The classifications
cannot be independently verified from the published CSV alone.

| Outcome | PASS | FAIL | UNKNOWN | Denominator |
|---|---:|---:|---:|---:|
| Target Feature Envy resolved | 10 | 3 | 53 | 66 applied patches |
| No new relevant Feature Envy | 0 | 1 | 65 | 66 applied patches |
| No dependency regression, inspected scope | 1 | 0 | 65 | 66 applied patches |
| Verified safe resolution | 0 | 4 | 62 | 66 applied patches |

PASS for the dependency row is limited to one ResourceManager project
comparison. It checked the changed class, introduced helper, and three
project-local direct calls using before/after semantic evidence; it does not
establish that every downstream consumer or runtime behavior is safe. The
new-smell FAIL is an introduced helper, `RelativeDateCalculator.Calculate`,
with ATFD 13, LAA 0, and FDP 3 under this project's evaluator. Its use of
framework properties and enum members affects interpretation. UNKNOWN is
neither PASS nor FAIL.

The later focused analysis added one target-level FAIL relative to the earlier
safety summary. The earlier saved aggregate's all-UNKNOWN dependency field was
not synchronized with the later one-case scoped comparison; the dependency
row above uses the later scoped assessment, not that stale field.

Previously recorded build results for the 66 patches were 3 PASS, 22 FAIL,
19 blocked by missing assets, 3 otherwise blocked, and 19 NOT_RUN. Recorded
test results were 1 PASS, 2 FAIL and 63 NOT_RUN. Failures were not necessarily
attributable to the patch, and missing validation does not establish safety.

The matched evidence is too incomplete to establish superiority for either
condition. No safe-refactoring comparison should be inferred from target smell
resolution or patch application alone. The raw per-patch artifacts are excluded
from this privacy-first release; these aggregates are not independently
auditable from the release alone.
