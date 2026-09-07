# datasets

Shared by `Playground/` and `centerpoint/`, which is why these sit at repo root rather
than inside either project.

- `examples/` — five hand-picked target photos, **tracked in git**. The qualitative check
  every approach runs against. Unlabelled: visual inspection only, never metrics.
  Chosen to span the failure modes — mid-grey backing (`easy`), low-contrast wood
  (`wood-background`), perspective plus watermark (`angle`), white-on-white
  (`white-background`), overlapping shots (`close-shots`).
- `bullet_rchsr/`, `bullet_holes/` — Roboflow downloads. **Gitignored and regenerable**
  via `Playground/src/prepare_bullet_*.py`, run from repo root. Labelled; all quantitative
  evaluation uses these splits.

Keep `examples/` free of non-image files — the Playground scripts iterate it with
`os.listdir` and feed every entry straight to an image loader.
