# Documentation Synchronization Rule

## Keep Documentation Current

Every project change must include a documentation impact check. Before finishing any implementation, inspect the changed behavior, public API, configuration, workflow, tests, and examples and identify the corresponding documentation in `README.md` and `docs/`.

If the change affects documented behavior, update the relevant documentation in the same task and in the same change set. Documentation updates must describe the current implementation, use valid APIs and examples, and preserve working relative links. Do not defer documentation synchronization to a later task.

Changes that add, remove, rename, or alter public functionality must update at least the relevant guide and any README overview, feature list, workflow choice, or example affected by the change. Changes to configuration or developer workflow must update the applicable contributor or project instructions when such documentation exists.

## Verification

Before declaring the task complete:

1. Review the final diff for stale names, behavior descriptions, examples, and links in `README.md` and `docs/`.
2. Check that documentation examples match the current public API and project conventions.
3. If no documentation update is needed, state why the change has no user-facing or contributor-facing impact; do not silently skip the check.

Documentation changes belong in the same pull request or commit as the implementation they document.

## Documentation-First Answers

When answering a question about the codebase, first search the relevant content in `README.md` and `docs/`. Use the source code only after checking the documentation, either to fill a documented gap or to verify details that the documentation does not cover.

Answers must distinguish between behavior confirmed by documentation and behavior inferred or verified from the implementation. When the documentation is missing or outdated, identify that gap and apply the documentation synchronization rule if the task includes a relevant code change.
