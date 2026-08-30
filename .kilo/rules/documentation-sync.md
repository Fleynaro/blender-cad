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
