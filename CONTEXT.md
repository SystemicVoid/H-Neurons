# Project Context

## Domain Vocabulary

### Run Contract

A Run Contract is the claim-bearing record that decides whether existing run outputs
may be reused or compared. It binds the measurement benchmark, model identity,
localization artifact identity, sample selection, alpha schedule, generation policy,
and benchmark-specific inputs needed for a reproducible comparison.

Run Contracts are stricter than ordinary provenance: a mismatch is a claim-integrity
blocker, not just metadata drift. Intervention runs and negative-control runs may
store different contract documents, but they share the same responsibility: refuse
to resume or compare outputs when the stored contract no longer matches the run
being requested.

### Baseline Binding

A Baseline Binding connects a control run to the intervention baseline it is meant
to test against. It includes the baseline summary identity, the baseline Run
Contract, and hashes for the files used as evidence. Baseline Binding mismatches are
control-claim blockers.
