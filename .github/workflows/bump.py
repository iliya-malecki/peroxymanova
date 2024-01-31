import sys
import tomllib
import os


def get_project_version():
    try:
        with open("pyproject.toml", "rb") as toml_file:
            data = tomllib.load(toml_file)
        version = data["project"]["version"]
        return version
    except Exception as e:
        print(f"Error reading pyproject.toml: {e}")
        sys.exit(1)


def compare_versions(old_version, project_version):
    old_version_parts = [int(part) for part in old_version.split(".")]
    project_version_parts = [int(part) for part in project_version.split(".")]
    try:
        for old, new in zip(old_version_parts, project_version_parts, strict=True):
            if new < old:
                return ""
            if new > old:
                return project_version
        return ""
    except ValueError:
        print(f"version formats incompatible: {old_version = }, {project_version = }")
        sys.exit(1)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python version_check.py <latest_tag>")
        sys.exit(1)

    old_version = sys.argv[1]
    if old_version.startswith("v"):
        old_version = old_version[1:]

    project_version = get_project_version()

    new_version = compare_versions(old_version, project_version)

    print(f"{new_version = }")
    with open(os.environ["GITHUB_ENV"], "a") as f:
        f.write("new_version=" + new_version)
