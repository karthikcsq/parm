from __future__ import annotations

from typing import Any, Callable


class AssertionKindError(LookupError):
    pass


def evaluate_assertions(
    assertions: list[dict[str, Any]],
    *,
    state: dict[str, Any],
    mutations: tuple[dict[str, Any], ...],
    trajectory: tuple[Any, ...],
) -> list[dict[str, Any]]:
    """Score every declared assertion against the final environment state.

    Each row keeps its ``role`` so the scorer can separate the decision the
    memory was supposed to change from ordinary workflow competence.
    """

    context = {"state": state, "mutations": mutations, "trajectory": trajectory}
    rows = []
    for assertion in assertions:
        kind = str(assertion.get("kind", ""))
        checker = _CHECKS.get(kind)
        if checker is None:
            raise AssertionKindError(f"unknown assertion kind: {kind!r}")
        passed, detail = checker(assertion, context)
        rows.append(
            {
                "id": str(assertion.get("id", "")),
                "kind": kind,
                "role": str(assertion.get("role", "")),
                "passed": bool(passed),
                "detail": detail,
            }
        )
    return rows


# -- individual checks ----------------------------------------------------


def _branch_exists(
    assertion: dict[str, Any], context: dict[str, Any]
) -> tuple[bool, str]:
    branch = str(assertion["branch"])
    branches = context["state"]["branches"]
    if branch in branches:
        return True, f"branch {branch} exists"
    matches = [
        name for name in branches if name.startswith(str(assertion.get("prefix", "")))
    ]
    if assertion.get("prefix") and matches:
        return True, f"branch matching prefix exists: {', '.join(sorted(matches))}"
    return False, f"no branch named {branch}"


def _file_exists(
    assertion: dict[str, Any], context: dict[str, Any]
) -> tuple[bool, str]:
    path = str(assertion["path"])
    files = context["state"]["files"]
    branch = assertion.get("branch")
    candidates = (
        {str(branch): files.get(str(branch), {})}
        if branch is not None
        else files
    )
    for name, contents in candidates.items():
        if path not in contents:
            continue
        missing = _missing_keywords(contents[path], assertion.get("body_keywords", []))
        if missing:
            return False, f"{path} on {name} is missing: {', '.join(missing)}"
        return True, f"{path} exists on {name}"
    return False, f"no branch contains {path}"


def _issue_exists(
    assertion: dict[str, Any], context: dict[str, Any]
) -> tuple[bool, str]:
    for issue in context["state"]["issues"].values():
        if assertion.get("state") and issue["state"] != assertion["state"]:
            continue
        if _missing_keywords(issue["title"], assertion.get("title_keywords", [])):
            continue
        if _missing_keywords(issue["body"], assertion.get("body_keywords", [])):
            continue
        return True, f"issue #{issue['number']} matches"
    return False, "no issue matches the required title and body keywords"


def _pull_request_exists(
    assertion: dict[str, Any], context: dict[str, Any]
) -> tuple[bool, str]:
    for pull in context["state"]["pull_requests"].values():
        head_prefix = str(assertion.get("head_prefix", ""))
        if head_prefix and not pull["head"].startswith(head_prefix):
            continue
        if assertion.get("base") and pull["base"] != assertion["base"]:
            continue
        if assertion.get("state") and pull["state"] != assertion["state"]:
            continue
        if _missing_keywords(pull["body"], assertion.get("body_keywords", [])):
            continue
        return True, f"pull request #{pull['number']} matches"
    return False, "no pull request matches the required shape"


def _pull_request_state(
    assertion: dict[str, Any], context: dict[str, Any]
) -> tuple[bool, str]:
    number = int(assertion["pull_number"])
    pull = context["state"]["pull_requests"].get(number)
    if pull is None:
        return False, f"no pull request #{number}"
    expected = str(assertion["expected"])
    if pull["state"] == expected:
        return True, f"#{number} is {expected}"
    return False, f"#{number} is {pull['state']}, expected {expected}"


def _reviewer_requested(
    assertion: dict[str, Any], context: dict[str, Any]
) -> tuple[bool, str]:
    number = int(assertion["pull_number"])
    pull = context["state"]["pull_requests"].get(number)
    if pull is None:
        return False, f"no pull request #{number}"
    reviewers = [name.casefold() for name in pull["requested_reviewers"]]
    wanted = [str(name).casefold() for name in assertion.get("reviewers", [])]
    missing = [name for name in wanted if name not in reviewers]
    if missing:
        return False, f"#{number} has no review request for {', '.join(missing)}"
    return True, f"#{number} requests review from {', '.join(wanted)}"


def _no_reviewer_requested(
    assertion: dict[str, Any], context: dict[str, Any]
) -> tuple[bool, str]:
    number = int(assertion["pull_number"])
    pull = context["state"]["pull_requests"].get(number)
    if pull is None:
        return False, f"no pull request #{number}"
    if pull["requested_reviewers"]:
        return False, (
            f"#{number} unexpectedly requests review from "
            + ", ".join(pull["requested_reviewers"])
        )
    return True, f"#{number} has no review request"


def _comment_contains(
    assertion: dict[str, Any], context: dict[str, Any]
) -> tuple[bool, str]:
    number = int(assertion["number"])
    state = context["state"]
    target = state["issues"].get(number) or state["pull_requests"].get(number)
    if target is None:
        return False, f"no issue or pull request #{number}"
    keywords = assertion.get("keywords", [])
    # "all" is the default because most checks name one specific thing. Use
    # "any" when several wordings are equally honest ways to say the same
    # thing, so the assertion measures the decision rather than the phrasing.
    match_any = str(assertion.get("match", "all")) == "any"
    for comment in target["comments"]:
        missing = _missing_keywords(comment["body"], keywords)
        if match_any and len(missing) < len(list(keywords)):
            return True, f"#{number} has a matching comment"
        if not match_any and not missing:
            return True, f"#{number} has a matching comment"
    joiner = " or " if match_any else ", "
    return False, f"#{number} has no comment containing {joiner.join(keywords)}"


def _comment_absent(
    assertion: dict[str, Any], context: dict[str, Any]
) -> tuple[bool, str]:
    """Require that nothing on the target says the thing named by keywords.

    The mirror of ``comment_contains``, for a control where the intervention
    would show up as words rather than as state.
    """

    number = int(assertion["number"])
    state = context["state"]
    target = state["issues"].get(number) or state["pull_requests"].get(number)
    if target is None:
        return False, f"no issue or pull request #{number}"
    keywords = list(assertion.get("keywords", []))
    for comment in target["comments"]:
        present = [
            keyword
            for keyword in keywords
            if str(keyword).casefold() in str(comment["body"]).casefold()
        ]
        if present:
            return False, f"#{number} mentions {', '.join(present)}"
    return True, f"#{number} says none of {', '.join(keywords)}"


def _issue_state(
    assertion: dict[str, Any], context: dict[str, Any]
) -> tuple[bool, str]:
    number = int(assertion["issue_number"])
    issue = context["state"]["issues"].get(number)
    if issue is None:
        return False, f"no issue #{number}"
    expected = str(assertion["expected"])
    if issue["state"] == expected:
        return True, f"#{number} is {expected}"
    return False, f"#{number} is {issue['state']}, expected {expected}"


def _issues_closed(
    assertion: dict[str, Any], context: dict[str, Any]
) -> tuple[bool, str]:
    issues = context["state"]["issues"]
    numbers = [int(value) for value in assertion.get("issue_numbers", [])]
    open_numbers = [
        number
        for number in numbers
        if number not in issues or issues[number]["state"] != "closed"
    ]
    if open_numbers:
        return False, "still open: " + ", ".join(f"#{n}" for n in open_numbers)
    return True, "closed " + ", ".join(f"#{n}" for n in numbers)


def _pull_requests_merged(
    assertion: dict[str, Any], context: dict[str, Any]
) -> tuple[bool, str]:
    pulls = context["state"]["pull_requests"]
    numbers = [int(value) for value in assertion.get("pull_numbers", [])]
    unmerged = [
        number
        for number in numbers
        if number not in pulls or pulls[number]["state"] != "merged"
    ]
    if unmerged:
        return False, "not merged: " + ", ".join(f"#{n}" for n in unmerged)
    return True, "merged " + ", ".join(f"#{n}" for n in numbers)


def _no_mutation(
    assertion: dict[str, Any], context: dict[str, Any]
) -> tuple[bool, str]:
    kinds = {str(kind) for kind in assertion.get("kinds", [])}
    filters = assertion.get("where", {})
    offending = [
        mutation
        for mutation in context["mutations"]
        if mutation["kind"] in kinds
        and all(mutation.get(key) == value for key, value in filters.items())
    ]
    if offending:
        return False, (
            "unexpected mutation at step(s) "
            + ", ".join(str(row["step_index"]) for row in offending)
        )
    return True, "no forbidden mutation occurred"


def _files_unchanged(
    assertion: dict[str, Any], context: dict[str, Any]
) -> tuple[bool, str]:
    """Guard tracked paths that the task never authorizes rewriting.

    Scope this to a branch whenever the same edit is legitimate somewhere else.
    Rewriting a changelog on main out of band is collateral damage; bumping it
    on a hotfix branch is what the user's own release checklist asks for, and an
    unscoped check cannot tell those apart.
    """

    protected = {str(path) for path in assertion.get("paths", [])}
    branch = assertion.get("branch")
    offending = [
        mutation
        for mutation in context["mutations"]
        if mutation["kind"] == "write_file"
        and mutation.get("path") in protected
        and (branch is None or mutation.get("branch") == branch)
    ]
    if offending:
        return False, "rewrote protected path(s): " + ", ".join(
            sorted({str(row["path"]) for row in offending})
        )
    return True, "protected paths are unchanged"


def _any_of(
    assertion: dict[str, Any], context: dict[str, Any]
) -> tuple[bool, str]:
    """Pass when any listed alternative passes.

    Use this when a commitment can be honored through more than one mechanism.
    Demanding one exact call would measure API fluency rather than memory use.
    """

    alternatives = assertion.get("assertions", [])
    if not alternatives:
        raise AssertionKindError("any_of needs at least one alternative")
    details = []
    for alternative in alternatives:
        kind = str(alternative.get("kind", ""))
        checker = _CHECKS.get(kind)
        if checker is None:
            raise AssertionKindError(f"unknown assertion kind: {kind!r}")
        passed, detail = checker(alternative, context)
        details.append(f"{kind}: {detail}")
        if passed:
            return True, detail
    return False, "no alternative satisfied (" + "; ".join(details) + ")"


_CHECKS: dict[str, Callable[[dict[str, Any], dict[str, Any]], tuple[bool, str]]] = {
    "any_of": _any_of,
    "branch_exists": _branch_exists,
    "file_exists": _file_exists,
    "issue_exists": _issue_exists,
    "issue_state": _issue_state,
    "issues_closed": _issues_closed,
    "pull_request_exists": _pull_request_exists,
    "pull_request_state": _pull_request_state,
    "pull_requests_merged": _pull_requests_merged,
    "reviewer_requested": _reviewer_requested,
    "no_reviewer_requested": _no_reviewer_requested,
    "comment_contains": _comment_contains,
    "comment_absent": _comment_absent,
    "no_mutation": _no_mutation,
    "files_unchanged": _files_unchanged,
}


def available_assertion_kinds() -> tuple[str, ...]:
    return tuple(sorted(_CHECKS))


def _missing_keywords(text: str, keywords: Any) -> list[str]:
    folded = str(text).casefold()
    return [
        str(keyword)
        for keyword in (keywords or [])
        if str(keyword).casefold() not in folded
    ]
