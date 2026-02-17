Generate release notes for the next version.

The user optionally provides a version as $ARGUMENTS (e.g., `0.2.0`). If not provided, read the current version from `pyproject.toml` and suggest the next version based on the changes (patch for fixes only, minor for new features, major for breaking changes).

## Steps

1. **Find the latest git tag**: `git tag --sort=-v:refname | head -1`
2. **List commits since that tag**: `git log --oneline <last_tag>..HEAD`
3. **Read commit messages** and categorize them using conventional commit prefixes:
   - `feat` → **Features**
   - `fix` → **Bug Fixes**
   - `refactor` → **Refactoring**
   - `perf` → **Performance**
   - `docs` → **Documentation**
   - `test` → **Tests**
   - `build`, `ci`, `chore` → **Maintenance** (group these together)
   - Breaking changes (any commit with `!` after type, e.g., `feat!:`) → **Breaking Changes** section at the top
4. **Skip**: Merge commits, commits that only touch `.claude/` or `docs/plans/`
5. **Check for version bump**: Read version in `pyproject.toml` — warn if it hasn't been updated from the last release

## Output Format

Generate a title and body suitable for `gh release create`. Output them clearly so the user can review before posting.

**Title format**: `v<version>` (e.g., `v0.2.0`)

**Body format**:
```markdown
## What's New

### Features
- Brief description of feature ([#PR](url) if applicable)

### Bug Fixes
- Brief description of fix

### Refactoring
- Brief description of change

### Maintenance
- Brief description

**Full Changelog**: https://github.com/horizon-rl/strands-env/compare/<last_tag>...v<new_version>
```

## Guidelines

- Write descriptions from the **user's perspective**, not the commit message verbatim. For example, "Add dual tool limits (max iterations + max calls)" is better than "feat(tool_limiter): add dual tool limits".
- Omit empty sections.
- Group related commits into a single bullet point when they're part of the same change.
- Keep it concise — release notes are for users, not a git log dump.

## Output

Write the release notes to a file at `docs/releases/v<version>.md` so the user can easily copy the content. The file should contain the title as an H1 heading followed by the body. Create the `docs/releases/` directory if it doesn't exist.

After writing, tell the user the file path so they can review and copy from it.

## Important

- Do NOT create the release or push tags — the user does this manually via GitHub web UI (see CLAUDE.md).
- Do NOT modify `pyproject.toml` unless the user asks.
