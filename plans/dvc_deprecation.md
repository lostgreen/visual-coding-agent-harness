# DVC / Scene-Index Deprecation Note

Current status:

- The active multi_v3 path builds and consumes `VideoWorkspace` with `Chapter`/`Beat`.
- The old public `Scene` / `Shot` / `VideoIndex` hierarchy has been removed from `video/index.py`.
- `source_segments` inheritance and duration-based scene aggregation have been removed from the active builder path.
- The VideoMME DVC-style `SceneIndex` remains only as a flat cache/input artifact for root captions, ASR hints, and existing eval cache compatibility.

Operational rule:

- New agent execution must use `build_video_workspace`, `VideoWorkspace.timeline_text`, and playbook programs.
- DVC root captions may be used as optional construction hints until VideoMME ablations confirm they can be removed from cache generation entirely.
