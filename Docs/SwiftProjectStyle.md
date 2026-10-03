# Swift project style

Use this guide when creating or reorganizing Stephen's Swift projects. It is a
practical default for readable code and a navigable package, informed by Swift
project repositories. Explicit user instructions and an existing project's local
conventions take precedence. Do not reformat unrelated code to impose this guide.

This policy was established on 2026-09-22 after a survey of Swift project
conventions. It is a default for Midnight where the repository has no stronger
local convention.

## Chosen conventions

There is no universal formatting style across Swift's repositories. SwiftPM uses
four-space indentation and a 120-column limit in its [.editorconfig](https://github.com/swiftlang/swift-package-manager/blob/de46f21fbf317cedc7a6ae18893cc968a7126c63/.editorconfig)
and [.swiftformat](https://github.com/swiftlang/swift-package-manager/blob/de46f21fbf317cedc7a6ae18893cc968a7126c63/.swiftformat).
Swift Syntax and SourceKit-LSP use two spaces; other projects differ on line and
documentation wrapping. For new projects, use this deliberately selected default:

- Four spaces, no tabs, UTF-8, LF, and a final newline.
- A 120-column code target. Wrap earlier when it makes an expression easier to follow.
- One statement per line. Expand initializers and control flow instead of packing them with semicolons.
- Separate related steps with a blank line. Do not add blank lines immediately inside braces.
- Keep an obvious computed property or empty initializer compact when it remains clear.
- Put complex closure bodies and switch cases on separate lines. Name intermediate values when they clarify units or intent.
- Commit `.editorconfig` and `.swift-format`. The toolchain's `swift format` is our enforcement tool; SwiftPM itself uses the different SwiftFormat tool.

Do not fight the formatter over alignment. Preserve string literal content, terminal
escape sequences, and fixtures exactly during formatting-only changes.

## Start with a small package

Use SwiftPM's standard source and test discovery. Create directories only when
there is content to put in them.

```text
Package.swift
README.md
CONTRIBUTING.md
AGENTS.md
.editorconfig
.swift-format
Sources/
    library/
        Feature/
            PrimaryType.swift
    application/
        main.swift
Tests/
    libraryTests/
        PrimaryTypeTests.swift
Docs/
    Architecture.md
Scripts/
    check-swift-format.sh
```

List products, dependencies, and targets clearly in the manifest. Introduce a
module boundary when it expresses ownership or dependency direction, not simply
because a file is long. Keep demos and application policy outside reusable
libraries. Avoid generic `Utils`, `Managers`, and `Helpers` directories.

For Midnight, the main dependency direction is:

```text
Midnight → ModelRunnerCore → ModelRunnerProtocol
```

`Midnight` owns the CLI and HTTP listener. `ModelRunnerCore` owns model loading,
generation, caches, and MLX execution. `ModelRunnerProtocol` owns shared wire
types, settings, and resource limits. The optional Vision package and shared
ModelFiles package have their own manifests. Keep application policy out of the
reusable libraries.

Keep the serving path Swift-first. Use platform APIs only at narrow runtime
and process boundaries, and do not add a foreign-language implementation when
Swift can reasonably provide the behavior.

## Make the source easy to navigate

Name a file after its primary type or focused responsibility. Small related types
can share a file when that improves understanding. Group larger targets by feature,
such as `Rendering`, `Text`, and `Widgets`; avoid arbitrary depth.

Within a type, group public state, private storage, initialization, public behavior,
and implementation helpers. Keep related operations close together. Use extensions
for a coherent capability, such as `ChatApplication+Rendering.swift`, rather than
splitting every method into its own file. Cross-file extensions should not force
public access to application internals.

Keep executable entrypoints short: parse options, construct the application, run it,
and report an error. Give event handling and rendering named methods so their flow
can be read independently. Avoid a giant function that owns every local variable.

The [Foundation contribution guide](https://github.com/swiftlang/swift-foundation/blob/b898bbce74a46ce590791e0a4e3ff181a8b8af25/CONTRIBUTION_GUIDELINE.md)
is a useful reference for focused files, descriptive names, and explaining decisions.

## Write readable Swift

Follow the [Swift API Design Guidelines](https://www.swift.org/documentation/api-design-guidelines/):
make a call site understandable and prefer clarity over squeezing code into fewer
characters. Preserve published signatures during a style cleanup.

- Use `UpperCamelCase` for types and `lowerCamelCase` for members and locals.
- Preserve the module and product names in `Package.swift`; the `midnight` command remains lowercase. This does not lowercase Swift type names.
- Name values by their role: `visibleArea`, `entryStart`, `cursorColumn`, and `remainingBytes`.
- Short coordinate names such as `x` and `y` are appropriate when their meaning is obvious.
- Use parameter labels that explain operations. Give sizes, positions, durations, and limits clear units.
- Give state changes their own lines. Prefer explicit branches over nested ternaries when the reader must decode several decisions.
- Use `let` by default, keep mutation local, and expose only the API callers need.
- Use `self` when required for disambiguation or closure capture; otherwise follow the surrounding file consistently.

Prefer this:

```swift
public mutating func resize(width: Int, height: Int) {
    guard width != front.width || height != front.height else {
        return
    }

    front = Buffer(width: width, height: height)
    frame = Frame(width: width, height: height)
    presented = false
}
```

Avoid compressing several assignments, a guard, and a return onto the same line.
Whitespace should expose the algorithm rather than replace meaningful names.

## Document contracts and decisions

Add a short `///` summary to public types and operations. Explain what a caller
needs to know: units, coordinate system, side effects, ownership, errors, complexity,
and limits. For private code, explain a non-obvious invariant or reason, rather than
restating an assignment. Simple stored properties do not need boilerplate prose.

In model-serving code, document these distinctions explicitly:

- Prompt tokens versus emitted tokens, cache positions, and UTF-8 bytes.
- Who owns model and cache state, including load, cancellation, and error paths.
- Whether a request changes the loaded model, writes a report, or only reads state.
- Why a platform-specific MLX or process workaround exists.
- Which backends and model configurations were actually tested.

Use Markdown links to detailed architecture or protocol notes. Keep the README
short and put substantial explanations under `Docs/`. Use a brief
`AGENTS.md` to point future work at this guide and the verification commands.

Do not copy upstream copyright headers, licenses, private underscored APIs, ABI
attributes, or compatibility scaffolding merely to imitate the appearance of an
official repository. Independently written code needs its own accurate provenance.

## Keep boundaries explicit

Confine unsafe pointers and platform APIs to small helpers with clear ownership.
Separate HTTP request decoding from listener I/O so request validation can be
tested without a live socket. Keep report construction separate from file writes
so tests can inspect structured results.

Do not add actors, tasks, services, dependencies, or background threads solely for
style. If concurrency is required, state who owns mutable state and how shutdown
works. A readability refactor should preserve event ordering, output bytes, and
cleanup behavior.

## Make checks repeatable

Tests mirror module and feature boundaries. Use focused filenames, behavior-based
test names, and shared fixtures only when they remove actual duplication. Keep live
server smoke scripts separate from unit tests. A style change normally needs the existing
checks, not new tests that merely assert filenames or formatting choices.

For Midnight, from its root:

```sh
Scripts/format-swift.sh
Scripts/check-swift-format.sh
swift test
swift test --package-path Optional/Vision
swift build --product midnight
```

Build touched benchmark products and check their CLI help when changing those
entrypoints. Use release mode when validating or installing a release, not as a
prerequisite for every edit. Do not run competing SwiftPM builds against the
same build directory.

Before finishing, inspect the change for lost files, public API changes, stale
links, accidental literal changes, and edits outside the intended scope. Report
formatting, tests, and interactive checks separately. Do not call a prototype a
complete replacement for a mature upstream project.

## Applying this to the next project

1. Read local instructions and inspect the manifest and source tree.
2. Preserve established conventions; use this guide where a choice is still open.
3. Create the smallest useful targets and folders, with formatter configuration.
4. Keep application behavior, reusable logic, and platform boundaries distinct.
5. Write one representative API and its call site before expanding the design.
6. Add documentation and verification commands alongside the implementation.
7. Record deliberate exceptions instead of inventing a new convention on each task.
