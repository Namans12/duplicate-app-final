# Recovered from local folder (Desktop/duplicate_sort/Duplicate Identifier App/App)

Files that existed only on the local machine and were in no repository branch when the
local folder was retired (2026-10-06):

- `.github/agents/audio-dedup-e2e-optimizer.agent.md` - custom agent definition (was hidden by .gitignore).
- `main.py.wip` / `gpu_accelerator.py.wip` - unreferenced intermediate versions found as dangling
  git objects in the local .git (never committed; superseded by later commits on `final`).

Not preserved (deliberately): `.venv-app/` (rebuild with setup_app_env.ps1), `__pycache__/`, and
~10 dangling MP3 test blobs that were once staged by accident.
