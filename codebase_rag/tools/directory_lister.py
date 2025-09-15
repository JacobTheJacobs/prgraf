import os
from pathlib import Path

from loguru import logger
from pydantic_ai import Tool
from ..schemas import DirectoryListing
from .utils import PathResolutionError, safe_resolve_path


class DirectoryLister:
    def __init__(self, project_root: str):
        self.project_root = Path(project_root).resolve()

    def list_contents(self, directory_path: str) -> DirectoryListing:
        """
        Lists the contents of a specified directory, separating files and directories.
        """
        logger.info(f"Attempting to list directory: {directory_path}")
        try:
            target_path = safe_resolve_path(self.project_root, directory_path)
            if isinstance(target_path, PathResolutionError):
                logger.warning(str(target_path))
                return DirectoryListing(
                    directory_path=directory_path,
                    directories=[],
                    files=[],
                    error=str(target_path),
                )

            if not target_path.is_dir():
                error_msg = f"Error: '{directory_path}' is not a valid directory."
                logger.warning(error_msg)
                return DirectoryListing(
                    directory_path=directory_path,
                    directories=[],
                    files=[],
                    error=error_msg,
                )

            directories = []
            files = []
            for item in os.listdir(target_path):
                if (target_path / item).is_dir():
                    directories.append(item)
                else:
                    files.append(item)

            return DirectoryListing(
                directory_path=directory_path,
                directories=sorted(directories),
                files=sorted(files),
            )

        except Exception as e:
            logger.error(f"Error listing directory {directory_path}: {e}")
            return DirectoryListing(
                directory_path=directory_path,
                directories=[],
                files=[],
                error=f"Error: Could not list contents of '{directory_path}'. Reason: {e}",
            )


def create_directory_lister_tool(project_root: str) -> Tool:
    lister = DirectoryLister(project_root=project_root)
    return Tool(
        function=lister.list_contents,
        name="list_directory_contents",
        description="Lists the contents of a directory, separating files and subdirectories. Useful for exploring the codebase.",
    )
