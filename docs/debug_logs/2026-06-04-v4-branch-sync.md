# V4 Branch Sync Debug Note

Date: 2026-06-04

## Goal

Unify the v4 development branch across local, GitHub, and the requested KML machine, then resolve the previous KML dirty working tree without losing useful code.

## Current Evidence

- Branch: `codex/v4-skill-framework`
- Implementation sync baseline: `aa35b1aafa29caeabdf37b15ef74ebf5f95c68e7`
- Before this documentation sync, local HEAD, GitHub `origin/codex/v4-skill-framework`, KML HEAD, and KML `origin/codex/v4-skill-framework` all matched that baseline.
- KML worktree: clean on `codex/v4-skill-framework`

## KML Access

- KML URL: `https://kml-dtmachine-23666-prod-0.kmlhb2az1l3-2.corp.kuaishou.com`
- Remote repo: `/home/xuboshen/zgw/visual-coding-agent-harness`
- Remote Python: `/home/xuboshen/Anaconda/envs/visual-agent-harness/bin/python`
- Proxy for GitHub access on KML:

```bash
export http_proxy=http://oversea-squid2.ko.txyun:11080
export https_proxy=http://oversea-squid2.ko.txyun:11080
export no_proxy=localhost,127.0.0.1,localaddress,localdomain.com,internal,corp.kuaishou.com,test.gifshow.com,staging.kuaishou.com
```

The KML machine can reach GitHub through this proxy. A temporary `GIT_ASKPASS` credential helper was used for one GitHub fetch and was removed afterward. The token was not persisted in `origin` or git config.

## Dirty Worktree Resolution

The original KML dirty worktree was saved as:

```text
stash@{0}: On main: codex-v4-align-backup
```

Stash inspection showed 18 tracked files plus 7 untracked v4 files. The untracked files were compared against current HEAD and all matched exactly:

```text
src/visual_coding_agent_harness/agents/skills/__init__.py
src/visual_coding_agent_harness/agents/skills/predicates.py
src/visual_coding_agent_harness/agents/skills/specs.py
src/visual_coding_agent_harness/schemas.py
src/visual_coding_agent_harness/tools/global_view.py
tests/test_global_view.py
tests/test_v4_foundation.py
```

The tracked code/test changes from the stash are also absorbed by the current v4 branch. The remaining stash-vs-HEAD differences are documentation/debug-note updates, not missing implementation logic.

## Verification

Local:

```text
PYTHONPATH=src python -m unittest discover -s tests -p 'test_*.py'
Ran 113 tests OK.
```

KML:

```text
PYTHONPATH=src /home/xuboshen/Anaconda/envs/visual-agent-harness/bin/python -m unittest discover -s tests -p 'test_*.py'
Ran 113 tests OK.
```

## Next Actions

1. Keep development on `codex/v4-skill-framework`.
2. Continue v4 skill execution wiring from this clean branch.
3. Rotate or revoke the GitHub token used during this sync because it was shared in chat.
