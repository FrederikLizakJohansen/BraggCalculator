# External crystallographic explanation review

## Purpose

This gate tests whether the generated explanations are scientifically correct
and useful to practicing crystallographers. It is not a software unit test,
and it cannot be signed by the software author on behalf of external experts.

## Blinded procedure

1. Make one copy of `expert_review_packet.json` per reviewer.
2. Do not provide `expert_review_key.json` until all initial scores and comments
   have been returned.
3. Give reviewers the plotted numerical evidence, metric definitions and
   synthetic structures, but not the intended information-loss labels.
4. Each reviewer scores scientific correctness and diagnostic usefulness from
   1 to 5, decides whether the claimed mechanism follows from the evidence,
   and records counterexamples or missing qualifications.
5. Freeze returned packets without editing their scores. Revisions to software
   explanations must be recorded as a new benchmark version and reviewed
   again.

## Proposed publication acceptance rule

- at least two independent reviewers with crystallographic diffraction
  experience;
- median scientific-correctness score of at least 4 for every case;
- median usefulness score of at least 4 for every case;
- no unresolved `mechanism_supported=false` judgment;
- all requested qualifications either incorporated or explicitly rebutted in
  a response log;
- signed reviewer name, affiliation and date retained in the private review
  archive, with consent determining whether identities appear publicly.

## Current status

No packet is signed. `results.json` therefore records
`external_expert_review: null` and the package release status remains
`pending_external_review`.

This open gate is intentional evidence of what the software cannot establish
by testing itself.
