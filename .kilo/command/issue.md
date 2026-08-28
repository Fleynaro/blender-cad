---
description: "Research and prepare a GitHub issue without implementing the change"
agent: "code"
subtask: false
---

# Issue Preparation Workflow

Prepare an issue for this request: `$ARGUMENTS`.

This command prepares a GitHub issue and never implements the requested change. Do not edit repository files, run implementation diagnostics, or create tests while using this command.

## Research

1. Inspect the relevant implementation, tests, project rules, existing APIs, and likely integration or verification risks using read-only repository research.
2. Search open GitHub issues in `Fleynaro/blender-cad` for related work before drafting. Use GitHub read/search operations only during analysis; do not use `gh`.
3. If an open issue already covers the request, report its URL and why it is a duplicate, then stop. Do not draft or create another issue unless the user explicitly requests a distinct issue.
4. If the proposed work is harmful, unnecessary, or underspecified, recommend against creating it. State the technical reason and smallest viable alternative, then stop unless the user asks to pursue a distinct scope.

## Clarification

Ask only questions that materially affect scope, behavior, acceptance criteria, or whether the change should be rejected. Ask one question at a time and include a recommendation.

## Draft

After research and any necessary clarification, display exactly one complete issue draft in chat. Do not write the draft to the repository. Use these sections exactly:

```text
Title
Type
Problem and context
Evidence / affected code
Proposed scope
Out of scope
Acceptance criteria
Risks and constraints
Verification plan
```

The entire issue draft must be written in English, including the title and every section and detail. Translate the user's request and any relevant evidence into English when the request is written in another language. This English-only requirement applies even when the displayed draft is not yet confirmed.

Use concrete paths, symbols, and observed constraints where available. Do not invent implementation details or acceptance criteria.

## Confirmation and Publication

Wait for explicit confirmation in a later user message before publication, such as `create issue` or `confirm`. If the user changes the draft, revise it and display the entire updated draft again; never publish a stale draft.

After explicit confirmation, create exactly one issue with `github_issue_write` using:

- `method: create`
- Repository: `Fleynaro/blender-cad`
- The confirmed title and body
- Exactly one label: `bug` for a defect or `enhancement` for a new capability

Before publishing, verify that both the title and body are entirely in English. Never publish an untranslated issue, even when the original request or confirmation is in another language.

Do not create labels, pull requests, milestones, assignees, Project items, or perform any other GitHub write. Report the created issue URL and number, then stop without modifying code.
