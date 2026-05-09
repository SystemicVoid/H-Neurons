# Keep Claim-Bearing Modules Domain-Named

As of 2026-05-09, the architecture work is intentionally split around
domain-named Modules such as Run Contract, Baseline Binding, Benchmark Sweep,
CSV2 Measurement, SIMID Open-Correctness Evidence, SIMID Run Family, Artifact
IO, Evidence Pack, Review Package, and Model Runtime. Future refactors should
preserve current claim-bearing JSON shapes and scientific semantics first, and
only then deepen shared mechanics behind smaller Interfaces. Generic helper
Modules are acceptable for mechanics such as hashing, JSONL policy, alpha-panel
paths, or mismatch formatting, but they must not absorb the domain meaning owned
by the claim-bearing Modules.
