# Issue-First Workflow

## Goal

Require a confirmed GitHub issue before any significant change in this repository. The workflow is a backlog-management gate, not an automatic implementation trigger: after an issue is created, the agent stops and waits for a separate implementation request that cites the issue.

## Confirmed Decisions

- Significant work includes behavior, public API, test, and Blender-geometry changes.
- Exemptions: typo-only, formatting-only, comments, development-configuration changes, and work explicitly tied to an existing GitHub issue.
- Drafts are displayed only in the chat. They are never written to the repository.
- The agent must not create the GitHub issue until the user explicitly confirms the displayed draft (for example, `создай issue` or `подтверждаю`).
- Use the repository's existing `bug` label for fixes and `enhancement` for new capabilities.
- Search open issues for duplicates. When an open issue already covers the request, show it and stop instead of drafting or creating a duplicate unless the user explicitly requests a separate issue.
- If analysis shows the change is harmful, unnecessary, or underspecified, recommend against creating it; explain the reason and offer the smallest viable alternative rather than forcing an issue.

## Implementation

1. Add `.kilo/rules/issue-first.md` as a project-wide instruction.
   - Define the significant-change boundary and the agreed exemptions.
   - For a significant change without an existing issue URL or `#<number>` in the request, prohibit source/config/test edits and direct the agent into the issue-analysis workflow.
   - Permit read-only repository research, GitHub issue searches, and clarifying questions before an issue exists.
   - Require an implementation request to cite an existing issue after the issue is created; the agent must read that issue and confirm that the requested scope matches it before editing.
   - Make an explicit user instruction to bypass the workflow insufficient for significant work unless it includes a pre-existing issue reference; this preserves the backlog rule consistently.

2. Add `.kilo/command/issue.md` with frontmatter routing it to the `code` agent and expose it as `/issue <request>`.
   - State that the command prepares an issue, does not implement the requested change, and accepts the whole request through `$ARGUMENTS`.
   - Research phase: inspect the relevant code, tests, project rules, existing APIs, and likely integration/test risks. Use GitHub search to find related open issues before producing a draft.
   - Clarification phase: ask only questions that materially affect scope, behavior, acceptance criteria, or whether the change should be rejected. Ask one question at a time and make a recommendation.
   - Duplicate/rejection branch: if an open duplicate covers the request, report its link and rationale, then stop. If the proposed work should not proceed, explain the technical reason, safer alternative, and stop unless the user asks to pursue a distinct scope.
   - Draft phase: present exactly one readable draft in chat with `Title`, `Type`, `Problem and context`, `Evidence / affected code`, `Proposed scope`, `Out of scope`, `Acceptance criteria`, `Risks and constraints`, and `Verification plan` sections. Include concrete paths/symbols and observed constraints where available; do not invent implementation details or acceptance criteria.
   - Confirmation phase: wait for an explicit confirmation in a later user message. If the user changes the draft, revise and display the complete draft again; do not publish a stale version.
   - Publication phase: call `github_issue_write` with `method: create`, repository `Fleynaro/blender-cad`, the confirmed title/body, and exactly one label: `bug` for a defect or `enhancement` for a feature. Report the created issue URL/number and stop without modifying code.

3. Update `.kilo/command/feature.md` to add an `Issue gate` before its current `Understand` phase.
   - For non-exempt work, require an issue reference in the command invocation/context and require that the issue be read before diagnostics or code changes start.
   - If no reference exists, stop the feature workflow and invoke the behavior described by `/issue` rather than beginning the Blender diagnostic cycle.
   - Preserve the current Blender 5.0 diagnostic, test, and cleanup phases unchanged once the gate passes.

4. Keep GitHub behavior safe and predictable.
   - Use read/search GitHub operations during analysis only; the sole write operation is issue creation after confirmation.
   - Use the verified existing labels `bug` and `enhancement`; do not introduce label creation or depend on the unavailable `gh` executable.
   - Do not auto-close issues, assign people, alter milestones, or create pull requests as part of the workflow.

## Validation

1. Review the rendered command and rules for correct relative links and loaded `.kilo` paths.
2. In a Kilo session, invoke `/issue` with a hypothetical feature and verify the agent researches and displays a draft without calling `github_issue_write`.
3. Confirm a revised draft and verify exactly one GitHub issue is created with `enhancement`, then verify the agent stops without editing files.
4. Repeat with a defect and verify `bug` is used.
5. Invoke `/issue` with wording matching an existing open issue and verify the agent links the duplicate without creating a new issue.
6. Ask for a significant implementation without an issue reference and verify the project rule routes it to issue analysis; then retry with a valid `#<number>` and verify the existing feature-development workflow proceeds.

## Out Of Scope

- GitHub label creation or normalization; the required labels already exist.
- GitHub Projects, priorities, milestones, assignees, automatic issue closing, and automatic implementation after issue creation.
