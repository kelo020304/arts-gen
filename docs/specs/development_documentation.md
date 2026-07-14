# Development Documentation Spec

## Scope

This specification defines the default documentation workflow for code changes
in this repository.

## Required Documentation

- Update the relevant specification under `docs/specs/` when a change affects
  behavior, data contracts, APIs, interfaces, or operational guarantees.
- Update the relevant code update log under `TRELLIS-arts/code_update/` after an
  implementation change. Preserve an existing topic's file and format; use
  `<topic>.md` for a new topic.

## Plans

- Do not create or update a plan document by default.
- Create a plan document only when the user explicitly requests a plan, such as
  by saying `计划` or `plan`.

## Excluded Workflow Artifacts

Do not create brainstorming, milestone, UAT, summary, or Superpowers artifacts
unless the user explicitly requests that specific deliverable. Superpowers is
not part of the project workflow and must not be invoked or reinstalled.
