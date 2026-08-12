# Publishing the video gallery

The local gallery is ready, but it does not publish or upload anything by
itself. It expects these three public assets on a GitHub Release tagged
`video-v1`:

- `h3-standard-fp8.mp4`
- `h3-turbo-v4-fp8.mp4`
- `ltx-2.5-distilled-int8.mp4`

## One-time GitHub setup

1. Push the repository after reviewing the staged files.
2. Create a release with tag `video-v1` and upload the three MP4s from the
   canonical local artifact paths in `data/artifacts.tsv`.
3. In repository Settings → Pages, publish from the `main` branch and `/docs`
   folder.
4. The page will appear at `https://boxwrench.github.io/R9700/` after the Pages
   build completes.

The README poster frames are tracked because they are small. The MP4s remain
release assets so they do not enter normal Git history. If the release tag or
asset filenames change, update `docs/index.html` and the README links together.
