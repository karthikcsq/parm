from __future__ import annotations

import copy
from typing import Any

from .environment import (
    ToolInvocationError,
    ToolResult,
    ToolSpec,
    TrajectoryStep,
    _TrajectoryRecorder,
    register_environment,
)


ADAPTER_NAME = "github_fixture"
ADAPTER_VERSION = "github_fixture_v1"


def _object(properties: dict[str, Any], required: list[str]) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": properties,
        "required": required,
        "additionalProperties": False,
    }


_STRING = {"type": "string"}
_INTEGER = {"type": "integer"}

TOOLS: tuple[ToolSpec, ...] = (
    ToolSpec(
        "list_issues",
        "List repository issues with their number, title, state, and labels.",
        _object({"state": {"type": "string", "enum": ["open", "closed", "all"]}}, ["state"]),
    ),
    ToolSpec(
        "get_issue",
        "Read one issue including its full body and comments.",
        _object({"issue_number": _INTEGER}, ["issue_number"]),
    ),
    ToolSpec(
        "create_issue",
        "Open a new issue.",
        _object({"title": _STRING, "body": _STRING}, ["title", "body"]),
        mutating=True,
    ),
    ToolSpec(
        "add_issue_comment",
        "Comment on an issue or pull request.",
        _object({"issue_number": _INTEGER, "body": _STRING}, ["issue_number", "body"]),
        mutating=True,
    ),
    ToolSpec(
        "close_issue",
        "Close an issue.",
        _object({"issue_number": _INTEGER}, ["issue_number"]),
        mutating=True,
    ),
    ToolSpec(
        "list_pull_requests",
        "List pull requests with their number, title, state, head, and base.",
        _object({"state": {"type": "string", "enum": ["open", "closed", "all"]}}, ["state"]),
    ),
    ToolSpec(
        "get_pull_request",
        "Read one pull request including its full body, changed files, checks, "
        "requested reviewers, and comments.",
        _object({"pull_number": _INTEGER}, ["pull_number"]),
    ),
    ToolSpec(
        "create_pull_request",
        "Open a pull request from an existing branch.",
        _object(
            {"title": _STRING, "body": _STRING, "head": _STRING, "base": _STRING},
            ["title", "body", "head", "base"],
        ),
        mutating=True,
    ),
    ToolSpec(
        "merge_pull_request",
        "Merge an open pull request into its base branch.",
        _object({"pull_number": _INTEGER}, ["pull_number"]),
        mutating=True,
    ),
    ToolSpec(
        "request_reviewers",
        "Request review on a pull request from one or more collaborators.",
        _object(
            {
                "pull_number": _INTEGER,
                "reviewers": {"type": "array", "items": _STRING},
            },
            ["pull_number", "reviewers"],
        ),
        mutating=True,
    ),
    ToolSpec(
        "list_branches",
        "List branch names.",
        _object({}, []),
    ),
    ToolSpec(
        "create_branch",
        "Create a branch from an existing branch.",
        _object({"branch": _STRING, "from_branch": _STRING}, ["branch", "from_branch"]),
        mutating=True,
    ),
    ToolSpec(
        "list_files",
        "List every file path on a branch, optionally under a directory prefix.",
        _object({"branch": _STRING, "prefix": _STRING}, ["branch"]),
    ),
    ToolSpec(
        "get_file_contents",
        "Read a file on a branch.",
        _object({"branch": _STRING, "path": _STRING}, ["branch", "path"]),
    ),
    ToolSpec(
        "create_or_update_file",
        "Write a file on a branch.",
        _object(
            {
                "branch": _STRING,
                "path": _STRING,
                "content": _STRING,
                "message": _STRING,
            },
            ["branch", "path", "content", "message"],
        ),
        mutating=True,
    ),
    ToolSpec(
        "list_collaborators",
        "List repository collaborators who can be requested as reviewers.",
        _object({}, []),
    ),
)

_TOOLS_BY_NAME = {tool.name: tool for tool in TOOLS}


class GitHubFixtureEnvironment:
    """An isolated, in-process environment with GitHub's artifact shapes.

    MCPMark's own GitHub service duplicates a seed repository into a real
    private evaluation org and drives it over the REST API, which needs live
    credentials, mutates account-visible state, and cannot give a positive and
    its cue-ablated twin independent resets inside one parallel run. This
    adapter keeps the artifact shapes, the tool surface, and MCPMark's
    final-state verification style while staying local and reproducible. The
    fixture records the upstream MCPMark task and revision it was derived from.

    What it deliberately does not reproduce: API pagination, rate limits, and
    real error taxonomies. A future ``mcpmark_live`` adapter can register under
    the same protocol when running against a real org is worth the cost.
    """

    adapter_name = ADAPTER_NAME
    adapter_version = ADAPTER_VERSION

    def __init__(self, fixture: dict[str, Any]) -> None:
        self._fixture = copy.deepcopy(fixture)
        repository = self._fixture.get("repository")
        if not isinstance(repository, str) or not repository.strip():
            raise ValueError("github fixture needs a repository name")
        self.repository = repository
        self.default_branch = str(self._fixture.get("default_branch", "main"))
        self._branches: dict[str, dict[str, str]] = {
            str(name): dict(files)
            for name, files in self._fixture.get("branches", {}).items()
        }
        if self.default_branch not in self._branches:
            raise ValueError("github fixture default branch is missing from branches")
        self._issues: dict[int, dict[str, Any]] = {
            int(issue["number"]): _normalize_issue(issue)
            for issue in self._fixture.get("issues", [])
        }
        self._pulls: dict[int, dict[str, Any]] = {
            int(pull["number"]): _normalize_pull(pull)
            for pull in self._fixture.get("pull_requests", [])
        }
        self._collaborators = [
            str(name) for name in self._fixture.get("collaborators", [])
        ]
        self._next_number = max(
            [*self._issues, *self._pulls, 0]
        ) + 1
        self._recorder = _TrajectoryRecorder()
        self._actor = str(self._fixture.get("actor", "agent"))

    # -- protocol ---------------------------------------------------------

    def tools(self) -> tuple[ToolSpec, ...]:
        return TOOLS

    @property
    def trajectory(self) -> tuple[TrajectoryStep, ...]:
        return tuple(self._recorder.steps)

    def mutations(self) -> tuple[dict[str, Any], ...]:
        return self._recorder.mutations

    def invoke(self, name: str, arguments: dict[str, Any]) -> ToolResult:
        spec = _TOOLS_BY_NAME.get(name)
        step_index = self._recorder.next_step_index
        if spec is None:
            result = ToolResult(False, f"error: unknown tool {name!r}")
            self._recorder.record(name, arguments, result, mutating=False)
            return result
        handler = getattr(self, f"_tool_{name}")
        try:
            result = handler(step_index, dict(arguments))
        except ToolInvocationError as exc:
            result = ToolResult(False, f"error: {exc}")
        self._recorder.record(name, arguments, result, mutating=spec.mutating)
        return result

    def state(self) -> dict[str, Any]:
        return {
            "repository": self.repository,
            "default_branch": self.default_branch,
            "branches": {
                name: sorted(files) for name, files in self._branches.items()
            },
            "files": copy.deepcopy(self._branches),
            "issues": copy.deepcopy(self._issues),
            "pull_requests": copy.deepcopy(self._pulls),
            "collaborators": list(self._collaborators),
        }

    # -- read tools -------------------------------------------------------

    def _tool_list_issues(self, step: int, args: dict[str, Any]) -> ToolResult:
        state = _required_str(args, "state")
        rows = [
            issue
            for issue in self._issues.values()
            if state == "all" or issue["state"] == state
        ]
        if not rows:
            return ToolResult(True, "No issues match that state.")
        lines = [f"{len(rows)} issue(s) in {self.repository}:"]
        for issue in sorted(rows, key=lambda row: row["number"]):
            labels = ", ".join(issue["labels"]) or "none"
            lines.append(
                f"#{issue['number']} [{issue['state']}] {issue['title']} "
                f"(labels: {labels})"
            )
        return ToolResult(True, "\n".join(lines))

    def _tool_get_issue(self, step: int, args: dict[str, Any]) -> ToolResult:
        issue = self._issue(_required_int(args, "issue_number"))
        return ToolResult(True, _render_issue(issue))

    def _tool_list_pull_requests(self, step: int, args: dict[str, Any]) -> ToolResult:
        state = _required_str(args, "state")
        rows = [
            pull
            for pull in self._pulls.values()
            if state == "all" or pull["state"] == state
        ]
        if not rows:
            return ToolResult(True, "No pull requests match that state.")
        lines = [f"{len(rows)} pull request(s) in {self.repository}:"]
        for pull in sorted(rows, key=lambda row: row["number"]):
            lines.append(
                f"#{pull['number']} [{pull['state']}] {pull['title']} "
                f"({pull['head']} -> {pull['base']})"
            )
        return ToolResult(True, "\n".join(lines))

    def _tool_get_pull_request(self, step: int, args: dict[str, Any]) -> ToolResult:
        pull = self._pull(_required_int(args, "pull_number"))
        return ToolResult(True, _render_pull(pull))

    def _tool_list_branches(self, step: int, args: dict[str, Any]) -> ToolResult:
        return ToolResult(True, "Branches: " + ", ".join(sorted(self._branches)))

    def _tool_list_files(self, step: int, args: dict[str, Any]) -> ToolResult:
        branch_name = _required_str(args, "branch")
        branch = self._branch(branch_name)
        prefix = str(args.get("prefix") or "")
        paths = sorted(path for path in branch if path.startswith(prefix))
        if not paths:
            return ToolResult(True, f"No files on {branch_name} under {prefix!r}.")
        return ToolResult(
            True, f"{len(paths)} file(s) on {branch_name}:\n" + "\n".join(paths)
        )

    def _tool_get_file_contents(self, step: int, args: dict[str, Any]) -> ToolResult:
        branch_name = _required_str(args, "branch")
        branch = self._branch(branch_name)
        path = _required_str(args, "path")
        if path not in branch:
            raise ToolInvocationError(
                f"file not found: {path}. Files on {branch_name}: "
                + ", ".join(sorted(branch))
            )
        return ToolResult(True, f"{path}:\n{branch[path]}")

    def _tool_list_collaborators(self, step: int, args: dict[str, Any]) -> ToolResult:
        if not self._collaborators:
            return ToolResult(True, "No collaborators are configured.")
        return ToolResult(True, "Collaborators: " + ", ".join(self._collaborators))

    # -- write tools ------------------------------------------------------

    def _tool_create_issue(self, step: int, args: dict[str, Any]) -> ToolResult:
        number = self._next_number
        self._next_number += 1
        self._issues[number] = _normalize_issue(
            {
                "number": number,
                "title": _required_str(args, "title"),
                "body": _required_str(args, "body"),
                "state": "open",
                "author": self._actor,
            }
        )
        self._recorder.record_mutation(
            step, "create_issue", {"issue_number": number}
        )
        return ToolResult(True, f"Created issue #{number}.")

    def _tool_add_issue_comment(self, step: int, args: dict[str, Any]) -> ToolResult:
        number = _required_int(args, "issue_number")
        body = _required_str(args, "body")
        target = self._issues.get(number) or self._pulls.get(number)
        if target is None:
            raise ToolInvocationError(f"no issue or pull request #{number}")
        target["comments"].append({"author": self._actor, "body": body})
        self._recorder.record_mutation(
            step, "add_comment", {"number": number}
        )
        return ToolResult(True, f"Commented on #{number}.")

    def _tool_close_issue(self, step: int, args: dict[str, Any]) -> ToolResult:
        issue = self._issue(_required_int(args, "issue_number"))
        issue["state"] = "closed"
        self._recorder.record_mutation(
            step, "close_issue", {"issue_number": issue["number"]}
        )
        return ToolResult(True, f"Closed issue #{issue['number']}.")

    def _tool_create_pull_request(self, step: int, args: dict[str, Any]) -> ToolResult:
        head = _required_str(args, "head")
        base = _required_str(args, "base")
        self._branch(head)
        self._branch(base)
        number = self._next_number
        self._next_number += 1
        self._pulls[number] = _normalize_pull(
            {
                "number": number,
                "title": _required_str(args, "title"),
                "body": _required_str(args, "body"),
                "state": "open",
                "head": head,
                "base": base,
                "author": self._actor,
            }
        )
        self._recorder.record_mutation(
            step, "create_pull_request", {"pull_number": number}
        )
        return ToolResult(True, f"Opened pull request #{number} ({head} -> {base}).")

    def _tool_merge_pull_request(self, step: int, args: dict[str, Any]) -> ToolResult:
        pull = self._pull(_required_int(args, "pull_number"))
        if pull["state"] != "open":
            raise ToolInvocationError(
                f"pull request #{pull['number']} is {pull['state']}"
            )
        base = self._branch(pull["base"])
        head = self._branches.get(pull["head"], {})
        base.update(head)
        base.update(pull.get("changed_files", {}))
        pull["state"] = "merged"
        pull["merged"] = True
        self._recorder.record_mutation(
            step, "merge_pull_request", {"pull_number": pull["number"]}
        )
        return ToolResult(True, f"Merged pull request #{pull['number']} into {pull['base']}.")

    def _tool_request_reviewers(self, step: int, args: dict[str, Any]) -> ToolResult:
        pull = self._pull(_required_int(args, "pull_number"))
        reviewers = args.get("reviewers")
        if not isinstance(reviewers, list) or not reviewers:
            raise ToolInvocationError("reviewers must be a non-empty list")
        unknown = [
            str(name)
            for name in reviewers
            if str(name) not in self._collaborators
        ]
        if unknown:
            raise ToolInvocationError(
                "not a collaborator: " + ", ".join(unknown)
            )
        for name in reviewers:
            if str(name) not in pull["requested_reviewers"]:
                pull["requested_reviewers"].append(str(name))
        self._recorder.record_mutation(
            step,
            "request_reviewers",
            {"pull_number": pull["number"], "reviewers": [str(n) for n in reviewers]},
        )
        return ToolResult(
            True,
            f"Requested review on #{pull['number']} from "
            + ", ".join(str(name) for name in reviewers)
            + ".",
        )

    def _tool_create_branch(self, step: int, args: dict[str, Any]) -> ToolResult:
        branch = _required_str(args, "branch")
        source = self._branch(_required_str(args, "from_branch"))
        if branch in self._branches:
            raise ToolInvocationError(f"branch already exists: {branch}")
        self._branches[branch] = dict(source)
        self._recorder.record_mutation(step, "create_branch", {"branch": branch})
        return ToolResult(True, f"Created branch {branch}.")

    def _tool_create_or_update_file(
        self, step: int, args: dict[str, Any]
    ) -> ToolResult:
        branch_name = _required_str(args, "branch")
        branch = self._branch(branch_name)
        path = _required_str(args, "path")
        branch[path] = _required_str(args, "content")
        self._recorder.record_mutation(
            step, "write_file", {"branch": branch_name, "path": path}
        )
        return ToolResult(True, f"Wrote {path} on {branch_name}.")

    # -- helpers ----------------------------------------------------------

    def _issue(self, number: int) -> dict[str, Any]:
        issue = self._issues.get(number)
        if issue is None:
            raise ToolInvocationError(f"no issue #{number}")
        return issue

    def _pull(self, number: int) -> dict[str, Any]:
        pull = self._pulls.get(number)
        if pull is None:
            raise ToolInvocationError(f"no pull request #{number}")
        return pull

    def _branch(self, name: str) -> dict[str, str]:
        branch = self._branches.get(name)
        if branch is None:
            raise ToolInvocationError(f"no branch named {name}")
        return branch


def _normalize_issue(issue: dict[str, Any]) -> dict[str, Any]:
    return {
        "number": int(issue["number"]),
        "title": str(issue["title"]),
        "body": str(issue.get("body", "")),
        "state": str(issue.get("state", "open")),
        "labels": [str(label) for label in issue.get("labels", [])],
        "author": str(issue.get("author", "unknown")),
        "comments": [
            {"author": str(row.get("author", "unknown")), "body": str(row["body"])}
            for row in issue.get("comments", [])
        ],
    }


def _normalize_pull(pull: dict[str, Any]) -> dict[str, Any]:
    return {
        "number": int(pull["number"]),
        "title": str(pull["title"]),
        "body": str(pull.get("body", "")),
        "state": str(pull.get("state", "open")),
        "merged": bool(pull.get("merged", False)),
        "head": str(pull.get("head", "")),
        "base": str(pull.get("base", "main")),
        "author": str(pull.get("author", "unknown")),
        "checks": str(pull.get("checks", "unknown")),
        "changed_files": {
            str(path): str(content)
            for path, content in pull.get("changed_files", {}).items()
        },
        "requested_reviewers": [
            str(name) for name in pull.get("requested_reviewers", [])
        ],
        "comments": [
            {"author": str(row.get("author", "unknown")), "body": str(row["body"])}
            for row in pull.get("comments", [])
        ],
    }


def _render_issue(issue: dict[str, Any]) -> str:
    lines = [
        f"Issue #{issue['number']}: {issue['title']}",
        f"State: {issue['state']}",
        f"Author: {issue['author']}",
        f"Labels: {', '.join(issue['labels']) or 'none'}",
        "",
        "Body:",
        issue["body"],
    ]
    if issue["comments"]:
        lines.append("")
        lines.append("Comments:")
        for comment in issue["comments"]:
            lines.append(f"- {comment['author']}: {comment['body']}")
    return "\n".join(lines)


def _render_pull(pull: dict[str, Any]) -> str:
    lines = [
        f"Pull request #{pull['number']}: {pull['title']}",
        f"State: {pull['state']}",
        f"Author: {pull['author']}",
        f"Branches: {pull['head']} -> {pull['base']}",
        f"Checks: {pull['checks']}",
        "Requested reviewers: "
        + (", ".join(pull["requested_reviewers"]) or "none"),
        "Changed files: "
        + (", ".join(sorted(pull["changed_files"])) or "none"),
        "",
        "Body:",
        pull["body"],
    ]
    if pull["comments"]:
        lines.append("")
        lines.append("Comments:")
        for comment in pull["comments"]:
            lines.append(f"- {comment['author']}: {comment['body']}")
    return "\n".join(lines)


def _required_str(args: dict[str, Any], key: str) -> str:
    value = args.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ToolInvocationError(f"{key} must be a non-empty string")
    return value


def _required_int(args: dict[str, Any], key: str) -> int:
    value = args.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, str)):
        raise ToolInvocationError(f"{key} must be an integer")
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ToolInvocationError(f"{key} must be an integer") from exc


register_environment(ADAPTER_NAME, GitHubFixtureEnvironment)
