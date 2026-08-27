# Issue-First Workflow Rule

## Significant Changes Require an Issue

Before making a significant change, require an existing GitHub issue reference in the user's request or available context, unless the user explicitly states that an issue is not needed. A valid reference is an issue URL for `Fleynaro/blender-cad` or `#<number>`.

Significant changes include behavior changes, public API changes, test changes, and Blender-geometry changes. Do not edit source, configuration, or tests for significant work without a valid existing issue reference unless the user explicitly says an issue is not needed. That explicit instruction is sufficient and must be honored.

The following are exempt from the issue requirement: typo-only changes, formatting-only changes, comments, development-configuration changes, work explicitly tied to an existing GitHub issue, and work where the user explicitly states that an issue is not needed.

Before an issue exists, read-only repository research, GitHub issue searches, and clarifying questions are allowed. For significant work without a valid reference, do not begin implementation; follow the `/issue` workflow to analyze and prepare an issue instead, except when the user explicitly states that an issue is not needed.

## Implementation From an Issue

When a significant implementation request cites an issue, read that issue before making diagnostics or edits. Confirm that the requested scope matches the issue. If it does not match, stop and direct the request through `/issue` to revise or prepare the correct issue.

Creating an issue does not authorize implementation. After publication, stop and wait for a separate implementation request that cites the issue.
