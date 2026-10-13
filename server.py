import base64
import os
import re

import requests
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP

# Load environment variables from the .env file
load_dotenv()

mcp = FastMCP("GitHub Repository Assistant")

API_BASE = "https://api.github.com"

GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GITHUB_TOKEN:
    raise RuntimeError(
        "GITHUB_TOKEN is missing. Add it to your .env file."
    )

HEADERS = {
    "Accept": "application/vnd.github+json",
    "X-GitHub-Api-Version": "2022-11-28",
    "Authorization": f"Bearer {GITHUB_TOKEN}",
}


def validate_repo(owner: str, repo: str):
    """Validate GitHub owner and repository names."""
    pattern = r"^[A-Za-z0-9_.-]+$"

    if (
        not owner
        or not repo
        or not re.fullmatch(pattern, owner)
        or not re.fullmatch(pattern, repo)
    ):
        raise ValueError(
            "Invalid GitHub owner or repository name."
        )


def github_get(path: str, params=None):
    """Make a read-only GitHub API request."""
    response = requests.get(
        f"{API_BASE}{path}",
        headers=HEADERS,
        params=params,
        timeout=15,
    )

    if response.status_code == 404:
        raise ValueError(
            "Repository or resource not found, "
            "or access is unavailable."
        )

    if response.status_code == 401:
        raise RuntimeError(
            "GitHub authentication failed. Check GITHUB_TOKEN."
        )

    if response.status_code == 403:
        remaining = response.headers.get(
            "X-RateLimit-Remaining"
        )

        if remaining == "0":
            raise RuntimeError(
                "GitHub API rate limit exceeded."
            )

        raise RuntimeError(
            "GitHub API access forbidden. Check token permissions "
            "or API rate limits."
        )

    response.raise_for_status()
    return response.json()


@mcp.tool()
def get_repository_info(owner: str, repo: str) -> dict:
    """Get repository description, stars, language, and default branch."""
    validate_repo(owner, repo)

    data = github_get(f"/repos/{owner}/{repo}")

    return {
        "name": data["full_name"],
        "description": data.get("description"),
        "stars": data["stargazers_count"],
        "language": data.get("language"),
        "default_branch": data["default_branch"],
        "url": data["html_url"],
    }


@mcp.tool()
def list_recent_commits(
    owner: str, repo: str, limit: int = 5
) -> list:
    """List recent commits with SHA, message, author, and date."""
    validate_repo(owner, repo)

    if not 1 <= limit <= 10:
        raise ValueError("Limit must be between 1 and 10.")

    data = github_get(
        f"/repos/{owner}/{repo}/commits",
        params={"per_page": limit},
    )

    return [
        {
            "sha": item["sha"][:7],
            "message": item["commit"]["message"].splitlines()[0],
            "author": (
                item["commit"]["author"] or {}
            ).get("name"),
            "date": (
                item["commit"]["author"] or {}
            ).get("date"),
            "url": item["html_url"],
        }
        for item in data
    ]


@mcp.tool()
def list_open_issues(
    owner: str, repo: str, limit: int = 5
) -> list:
    """List open issues, excluding pull requests."""
    validate_repo(owner, repo)

    if not 1 <= limit <= 10:
        raise ValueError("Limit must be between 1 and 10.")

    data = github_get(
        f"/repos/{owner}/{repo}/issues",
        params={"state": "open", "per_page": limit},
    )

    return [
        {
            "number": item["number"],
            "title": item["title"],
            "labels": [
                label["name"]
                for label in item.get("labels", [])
            ],
            "url": item["html_url"],
        }
        for item in data
        if "pull_request" not in item
    ][:limit]


@mcp.tool()
def read_repository_file(
    owner: str, repo: str, path: str
) -> dict:
    """Read a text file from a GitHub repository."""
    validate_repo(owner, repo)

    if (
        not path
        or path.startswith("/")
        or "\\" in path
        or ".." in path.split("/")
    ):
        raise ValueError("Invalid repository file path.")

    data = github_get(
        f"/repos/{owner}/{repo}/contents/{path}"
    )

    if isinstance(data, list):
        raise ValueError("The requested path is a directory.")

    if data.get("type") != "file":
        raise ValueError("The requested resource is not a file.")

    if data.get("size", 0) > 100_000:
        raise ValueError(
            "File is too large for this demo (100 KB limit)."
        )

    if data.get("encoding") != "base64":
        raise ValueError("Unsupported file encoding.")

    try:
        content = base64.b64decode(
            data["content"], validate=True
        ).decode("utf-8", errors="replace")
    except (ValueError, KeyError) as exc:
        raise ValueError(
            "Could not decode the repository file."
        ) from exc

    return {
        "path": data["path"],
        "content": content,
        "url": data["html_url"],
    }


if __name__ == "__main__":
    mcp.run()
