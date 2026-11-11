# Releasing

Update the package version and merge the change. Push a `v<version>` tag or
publish a GitHub Release for that tag. `.github/workflows/release.yml` runs the
shared CI workflow, builds the distributions, and publishes them to PyPI after
all checks pass. A manual run of **Release** against a version tag can retry a
publication. A manual run against a branch skips publication.

Tag pushes and GitHub Release publication use the same concurrency group for a
tag. Publication runs serially, and files already uploaded to PyPI are skipped.

## PyPI Trusted Publisher

Before switching to this workflow, add a GitHub Trusted Publisher in the
[project publishing settings](https://pypi.org/manage/project/onnx-shape-inference/settings/publishing/):

- Owner: `onnx`
- Repository: `onnx-shape-inference`
- Workflow filename: `release.yml`
- Environment: `release`

PyPI binds a publisher to its workflow filename. Keep the existing `main.yml`
publisher until the new configuration is ready, then remove it after the first
successful publication through `release.yml`.
