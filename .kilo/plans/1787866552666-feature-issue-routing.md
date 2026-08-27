# Plan: Feature Command Issue Routing

## Context

`/issue` already researches, drafts, confirms, and publishes one issue without implementation. `/feature` currently requires an issue for significant work but does not define how its argument selects an existing issue versus a new idea.

## Decisions

- `/feature #<number>` and `/feature <GitHub issue URL>` select an existing issue.
- Any other non-empty `/feature` argument is a new work description and must run the `/issue` workflow automatically, then stop without implementation.
- The automatic issue branch preserves `/issue`'s research, duplicate detection, one-question-at-a-time clarification, draft confirmation, and publication rules.
- An existing issue must be read and open. A closed issue stops the command before diagnostics or edits.
- Existing issue scope must still match the requested work; a mismatch stops and directs the user to `/issue` for a new or revised issue.

## Implementation Tasks

1. Rewrite `.kilo/command/feature.md` around an explicit input-routing section using `$ARGUMENTS`.
2. Define the new-description branch: delegate to the `/issue` preparation workflow using the supplied description and terminate after that workflow, including after optional issue publication.
3. Define the issue-reference branch: accept only `#<number>` or an issue URL for `Fleynaro/blender-cad`, retrieve the issue, reject unreadable, invalid, cross-repository, or closed references, and confirm scope before continuing.
4. Retain the existing exemption behavior from `issue-first.md` for work that does not require an issue, while making clear that an explicit issue reference is still validated when supplied.
5. Keep the existing Blender implementation phases and their authority links unchanged after the routing and gate succeed.
6. Ensure the command never starts diagnostics, changes files, or creates tests in the new-description or rejected-reference branches.

## Validation

1. Review the rendered command instructions for three representative invocations: `/feature add a selector`, `/feature #123`, and `/feature https://github.com/Fleynaro/blender-cad/issues/123`.
2. Confirm the first invocation reaches only `/issue` behavior; both valid open references reach issue validation and then the existing phases.
3. Confirm closed, malformed, foreign-repository, and scope-mismatched references terminate before implementation.

## Risks and Boundaries

- This is command-workflow documentation/configuration only; no Blender code, tests, or GitHub records are changed.
- The automatic branch must not silently create an issue: `/issue`'s explicit later-message confirmation remains mandatory.
