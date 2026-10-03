# Swift project style

These conventions follow the Swift guide and formatter configuration in
`~/asslayer`. This package uses Swift 6.4 and SwiftPM's standard source and test
layout, with a single executable and libraries separated by responsibility.

Use four spaces, UTF-8, LF, a final newline, and a 120-column code target. Keep one
statement per line, expand complex control flow, separate related steps, and
preserve string contents and terminal escapes. Documentation begins with a short
summary followed by a separate explanatory paragraph when needed.

Keep the executable entrypoint short. Put command implementations in focused
files; reusable libraries must not import the executable. Prefer Apple's
ArgumentParser for commands and Foundation for networking, Codable, and files.
Do not introduce dependencies or actors merely for style. Keep loom and weft
lowercase. Put detailed explanations in `Documentation` and keep the README brief.

Run `Scripts/format-swift.sh`, `Scripts/check-swift-format.sh`, and `./test.sh`.
Exercise the PTY smoke test when terminal behavior changes. Pinned dependency
patches retain upstream spelling and layout; the formatter applies to this
package's manifest, Sources, and Tests rather than third-party patches.
