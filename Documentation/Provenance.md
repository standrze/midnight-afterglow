# Source provenance

The quantization and model support libraries were imported from Wick; decision
training was adapted from the independent Training project. Those original local
projects and their experiments are preserved. Afterglow builds independently.

`import-manifest.json` records retained imported files. Its `sha256` identifies
the original source at import; `current_sha256` identifies the cleaned, adapted
file in this repository. Renamed model-support paths preserve their origin.
Pinned third-party dependency overlays are in `Patches` and are applied by
`prepare-dependencies.sh`; upstream licenses and source notices remain intact.

Dependency identities and revisions are recorded in `Package.resolved`. Preserve
the matching dependency LICENSE and NOTICE files with binary distributions.
