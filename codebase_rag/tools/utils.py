from pathlib import Path


class PathResolutionError(ValueError):
    """Custom exception for path resolution errors."""


def safe_resolve_path(
    project_root: Path, requested_path: str
) -> Path | PathResolutionError:
    """
    Safely resolves a requested path against the project root, handling various
    common formats and ensuring the final path is within the project boundaries.
    """
    project_root = project_root.resolve()
    clean_path_str = requested_path.strip().replace("\\", "/")

    # Handle cases where the AI includes the project's root folder name in the path
    project_folder_name = project_root.name
    if clean_path_str.startswith(f"{project_folder_name}/"):
        clean_path_str = clean_path_str[len(project_folder_name) + 1 :]

    path_obj = Path(clean_path_str)

    # Determine the target path based on whether the input is absolute or relative
    if path_obj.is_absolute():
        target_path = path_obj.resolve()
    else:
        target_path = (project_root / path_obj).resolve()

    # Final, critical security check: ensure the resolved path is within the project root
    if not str(target_path).startswith(str(project_root)):
        return PathResolutionError(
            "Access denied: Cannot access paths outside the project root."
        )

    return target_path

