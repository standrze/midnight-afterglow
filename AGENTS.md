# Afterglow

Keep this repository focused on the `midnight-afterglow` command-line app, its
required libraries, tests, build overlays, and documentation. Serving belongs in
Midnight; inspection belongs in Studio. Do not upload weights, generated outputs,
web interfaces, unrelated tools, or research campaigns here.

Follow `Documentation/SwiftProjectStyle.md`, adapted from `~/asslayer`.
Use `Scripts/format-swift.sh` and `Scripts/check-swift-format.sh` for Swift edits.
Keep loom and weft lowercase. Run `./test.sh`; for terminal changes also run
`python3 Tests/terminal-smoke.py .build/debug/midnight-afterglow`.
