#!/usr/bin/env python3
# Copyright 2026 Zachary Brooks
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Validate that all Python source files have the required Apache 2.0 license header."""

import re
import sys
from pathlib import Path

# Expected Apache 2.0 license header pattern (flexible on year attribution)
LICENSE_HEADER_PATTERN = re.compile(
    r"^# Copyright \d{4}.*\n"
    r"#\n"
    r"# Licensed under the Apache License, Version 2\.0 \(the \"License\"\);\n"
    r"# you may not use this file except in compliance with the License\.\n"
    r"# You may obtain a copy of the License at\n"
    r"#\n"
    r"#\s+http://www\.apache\.org/licenses/LICENSE-2\.0\n"
    r"#\n"
    r"# Unless required by applicable law or agreed to in writing, software\n"
    r"# distributed under the License is distributed on an \"AS IS\" BASIS,\n"
    r"# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied\.\n"
    r"# See the License for the specific language governing permissions and\n"
    r"# limitations under the License\.\n",
    re.MULTILINE,
)


# Files to exclude from checking
EXCLUDED_FILES = {"__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache"}


def find_python_files(directory: Path) -> list[Path]:
    """Recursively find all Python files in the given directory."""
    python_files = []
    for path in directory.rglob("*.py"):
        # Skip excluded directories
        if any(part in EXCLUDED_FILES for part in path.parts):
            continue
        python_files.append(path)
    return sorted(python_files)


def has_valid_license_header(file_path: Path) -> bool:
    """Check if a file has a valid Apache 2.0 license header."""
    try:
        content = file_path.read_text(encoding="utf-8")
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return False

    # Check if the license header pattern matches at the start of the file
    match = LICENSE_HEADER_PATTERN.match(content)
    return match is not None


def main() -> int:
    """Main entry point."""
    # Get the project root (parent of the scripts directory)
    script_dir = Path(__file__).parent.resolve()
    project_root = script_dir.parent

    # Check both src and tests directories
    check_dirs = [project_root / "src", project_root / "tests"]

    missing_headers = []
    checked_files = []

    for check_dir in check_dirs:
        if not check_dir.exists():
            continue

        for py_file in find_python_files(check_dir):
            checked_files.append(py_file)
            if not has_valid_license_header(py_file):
                missing_headers.append(py_file)

    # Print results
    print(f"Checked {len(checked_files)} Python files.")

    if missing_headers:
        print(
            f"\n❌ {len(missing_headers)} file(s) missing valid "
            "Apache 2.0 license header:\n"
        )
        for file_path in missing_headers:
            rel_path = file_path.relative_to(project_root)
            print(f"  - {rel_path}")
        print("\nExpected header format:")
        print("""
# Copyright YYYY [Author]
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""")
        return 1
    else:
        print("✅ All Python files have valid Apache 2.0 license headers.")
        return 0


if __name__ == "__main__":
    sys.exit(main())
