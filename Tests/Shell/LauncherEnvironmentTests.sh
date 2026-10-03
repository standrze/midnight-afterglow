#!/usr/bin/env bash
set -euo pipefail

PACKAGE_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"

# Mock compiler/build processes; exercise the real launchers without MLX or GPU work.
python3 - "$PACKAGE_ROOT" <<'PY'
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

source = Path(sys.argv[1])
launchers = {
    "quantize-laguna-q4r8.sh": "wick-laguna-quantize",
    "rescore-laguna-q4r8.sh": "wick-laguna-q4r8-rescore",
    "verify-laguna-q4r8.sh": "wick-laguna-q4r8-verify",
    "audit-laguna-q4-scale-search.sh": "wick-q4-scale-search-audit",
    "benchmark-metal-quantization.sh": "wick-metal-quant-bench",
}


def executable(path, text):
    path.write_text(text)
    path.chmod(0o755)


with tempfile.TemporaryDirectory(prefix="wick-launchers-") as temporary:
    root = Path(temporary).resolve()
    scripts, mocks = root / "Scripts", root / "mocks"
    scripts.mkdir()
    mocks.mkdir()
    for name in [*launchers, "swiftpm-scratch-path.sh"]:
        shutil.copy2(source / "Scripts" / name, scripts)

    # Model the documented build boundary: Wick settings take precedence over
    # Facet compatibility settings and the original model-runner settings.
    # Linux honors the selected scratch directory, while Metal uses .build.
    build = '''#!/usr/bin/env python3
import json, os
from pathlib import Path
root = Path(os.environ["TEST_ROOT"])
product = os.environ.get("WICK_BUILD_PRODUCT") or os.environ.get("FACET_BUILD_PRODUCT") or os.environ.get("MODEL_RUNNER_BUILD_PRODUCT") or "wick"
configuration = os.environ.get("WICK_BUILD_CONFIGURATION") or os.environ.get("FACET_BUILD_CONFIGURATION") or os.environ.get("MODEL_RUNNER_BUILD_CONFIGURATION") or "release"
scratch = root / ".build"
if os.environ["TEST_HOST"] == "Linux":
    selected = os.environ.get("WICK_SCRATCH_PATH") or os.environ.get("FACET_SCRATCH_PATH") or os.environ.get("MODEL_RUNNER_SCRATCH_PATH")
    if selected:
        scratch = root / selected
destination = scratch / configuration
destination.mkdir(parents=True, exist_ok=True)
binary = destination / product
binary.write_text("#!/usr/bin/env python3\\nimport json, os, sys\\nfrom pathlib import Path\\nPath(os.environ['TEST_LAUNCH_RECORD']).write_text(json.dumps({'binary': sys.argv[0], 'arguments': sys.argv[1:]}))\\n")
binary.chmod(0o755)
Path(os.environ["TEST_BUILD_RECORD"]).write_text(json.dumps({"product": product, "configuration": configuration, "binary": str(binary)}))
'''
    executable(root / "build.sh", build)
    executable(root / "build-metal.sh", build)
    executable(mocks / "uname", '#!/usr/bin/env bash\nprintf "%s\\n" "$TEST_HOST"\n')
    executable(mocks / "swift", '''#!/usr/bin/env python3
import os, sys
from pathlib import Path
arguments = sys.argv[1:]
assert arguments[0] == "build" and "--show-bin-path" in arguments, arguments
configuration = arguments[arguments.index("--configuration") + 1]
scratch = Path(os.environ["TEST_ROOT"]) / ".build"
if "--scratch-path" in arguments:
    scratch = Path(arguments[arguments.index("--scratch-path") + 1])
print(scratch / configuration)
''')

    count = 0
    for host in ("Darwin", "Linux"):
        for scratch_selection in ("wick", "facet", "legacy"):
            for launcher, expected_product in launchers.items():
                if host == "Linux" and launcher == "benchmark-metal-quantization.sh":
                    continue
                for path in (root / ".build", root / "selected scratch", root / "legacy scratch"):
                    shutil.rmtree(path, ignore_errors=True)
                build_record, launch_record = root / "build.json", root / "launch.json"
                build_record.unlink(missing_ok=True)
                launch_record.unlink(missing_ok=True)
                environment = {
                    **os.environ,
                    "PATH": str(mocks) + os.pathsep + os.environ["PATH"],
                    "TEST_ROOT": str(root), "TEST_HOST": host,
                    "TEST_BUILD_RECORD": str(build_record),
                    "TEST_LAUNCH_RECORD": str(launch_record),
                    "WICK_BUILD_PRODUCT": "wick",
                    "WICK_BUILD_CONFIGURATION": "debug",
                    "FACET_BUILD_PRODUCT": "facet",
                    "FACET_BUILD_CONFIGURATION": "release",
                    "MODEL_RUNNER_BUILD_PRODUCT": "wick",
                    "MODEL_RUNNER_BUILD_CONFIGURATION": "debug",
                    "MODEL_RUNNER_SCRATCH_PATH": "legacy scratch",
                    "SPM_CUDA": "0",
                }
                environment.pop("WICK_SCRATCH_PATH", None)
                environment.pop("FACET_SCRATCH_PATH", None)
                if scratch_selection == "wick":
                    environment["WICK_SCRATCH_PATH"] = "selected scratch"
                    environment["FACET_SCRATCH_PATH"] = "ignored facet scratch"
                elif scratch_selection == "facet":
                    environment["FACET_SCRATCH_PATH"] = "facet scratch"
                arguments = ["source with spaces", "destination with spaces"]
                result = subprocess.run(
                    ["bash", str(scripts / launcher), *arguments],
                    cwd=root, env=environment, text=True, capture_output=True, timeout=10,
                )
                context = (host, scratch_selection, launcher, result.stderr)
                assert result.returncode == 0, context
                built, launched = json.loads(build_record.read_text()), json.loads(launch_record.read_text())
                assert built["product"] == expected_product, (context, built)
                assert built["configuration"] == "release", (context, built)
                assert launched["binary"] == built["binary"], (context, built, launched)
                assert launched["arguments"] == arguments, (context, launched)
                expected_scratch = ".build" if host == "Darwin" else (
                    "selected scratch" if scratch_selection == "wick" else
                    "facet scratch" if scratch_selection == "facet" else "legacy scratch")
                assert Path(built["binary"]).parent == root / expected_scratch / "release", (context, built)
                count += 1
    print(f"Launcher environment checks passed ({count} mock builds; no compiler or GPU workloads).")
PY
