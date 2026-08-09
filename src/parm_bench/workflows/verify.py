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


def _thread_message_exists(
    assertion: dict[str, Any], context: dict[str, Any], folder: str
) -> tuple[bool, str]:
    thread_id = str(assertion["thread_id"])
    for message in context["state"].get("messages", {}).values():
        if message.get("thread_id") != thread_id or message.get("folder") != folder:
            continue
        missing = _missing_keywords(message.get("body", ""), assertion.get("body_keywords", []))
        if not missing:
            return True, f"{folder} message exists in {thread_id}"
    return False, f"no {folder} message in {thread_id} matches the required body keywords"


def _message_sent(
    assertion: dict[str, Any], context: dict[str, Any]
) -> tuple[bool, str]:
    return _thread_message_exists(assertion, context, "sent")


def _thread_message_content(
    assertion: dict[str, Any], context: dict[str, Any]
) -> tuple[bool, str]:
    thread_id = str(assertion["thread_id"])
    folder = str(assertion["folder"])
    if folder not in {"drafts", "sent"}:
        return False, f"unsupported message folder {folder!r}"
    includes = assertion.get("includes", [])
    excludes = assertion.get("excludes", [])
    candidates = [
        message for message in context["state"].get("messages", {}).values()
        if message.get("thread_id") == thread_id and message.get("folder") == folder
    ]
    if not candidates:
        return False, f"no {folder} message in {thread_id}"
    failures = []
    for message in candidates:
        body = str(message.get("body", ""))
        missing = _missing_keywords(body, includes)
        forbidden = [
            str(keyword) for keyword in excludes
            if str(keyword).casefold() in body.casefold()
        ]
        if not missing and not forbidden:
            return True, f"{folder} message in {thread_id} has required content"
        details = []
        if missing:
            details.append("missing " + ", ".join(missing))
        if forbidden:
            details.append("contains " + ", ".join(forbidden))
        failures.append("; ".join(details))
    return False, f"no {folder} message in {thread_id} matches content ({' | '.join(failures)})"


def _follow_up_assigned(
    assertion: dict[str, Any], context: dict[str, Any]
) -> tuple[bool, str]:
    thread_id = str(assertion["thread_id"])
    actual = context["state"].get("follow_ups", {}).get(thread_id)
    if actual is None:
        return False, f"no follow-up is assigned for {thread_id}"
    expected = {
        key: str(assertion[key])
        for key in ("owner", "due_date")
        if key in assertion
    }
    mismatches = [key for key, value in expected.items() if str(actual.get(key)) != value]
    if mismatches:
        return False, f"follow-up for {thread_id} has different {', '.join(mismatches)}"
    return True, f"follow-up for {thread_id} is assigned"


def _follow_up_routed(
    assertion: dict[str, Any], context: dict[str, Any]
) -> tuple[bool, str]:
    """Require a follow-up on the named thread to reach a resolved target."""

    thread_id = str(assertion["thread_id"])
    actual = context["state"].get("follow_ups", {}).get(thread_id)
    if actual is None:
        return False, f"no follow-up is routed for {thread_id}"
    target_id = str(assertion["target_id"])
    if str(actual.get("target_id")) != target_id:
        return False, f"follow-up for {thread_id} is not routed to {target_id}"
    return True, f"follow-up for {thread_id} is routed to {target_id}"


def _event_response(
    assertion: dict[str, Any], context: dict[str, Any]
) -> tuple[bool, str]:
    event_id = str(assertion["event_id"])
    event = context["state"].get("events", {}).get(event_id)
    if event is None:
        return False, f"no event {event_id}"
    expected_status = assertion.get("expected")
    if expected_status is not None and event.get("status") != expected_status:
        return False, f"event {event_id} is {event.get('status')}, expected {expected_status}"
    return True, f"event {event_id} response is {event.get('status')}"


def _event_rescheduled(
    assertion: dict[str, Any], context: dict[str, Any]
) -> tuple[bool, str]:
    event_id = str(assertion["event_id"])
    event = context["state"].get("events", {}).get(event_id)
    if event is None:
        return False, f"no event {event_id}"
    expected = {key: str(assertion[key]) for key in ("start", "end") if key in assertion}
    mismatches = [key for key, value in expected.items() if str(event.get(key)) != value]
    if mismatches:
        return False, f"event {event_id} has different {', '.join(mismatches)}"
    approvals = [
        mutation for mutation in context["mutations"]
        if mutation["kind"] == "reschedule_event" and mutation.get("event_id") == event_id
    ]
    if not approvals:
        return False, f"event {event_id} was not rescheduled"
    approval_reference = assertion.get("approval_reference")
    if approval_reference is not None and not any(
        mutation.get("approval_reference") == str(approval_reference)
        for mutation in approvals
    ):
        return False, f"event {event_id} has no matching approval reference"
    return True, f"event {event_id} was rescheduled"


def _event_rescheduled_outside_window(
    assertion: dict[str, Any], context: dict[str, Any]
) -> tuple[bool, str]:
    """Require the named event to move wholly outside a prohibited interval."""

    event_id = str(assertion["event_id"])
    event = context["state"].get("events", {}).get(event_id)
    if event is None:
        return False, f"no event {event_id}"
    mutations = [
        row for row in context["mutations"]
        if row["kind"] == "reschedule_event" and row.get("event_id") == event_id
    ]
    if not mutations:
        return False, f"event {event_id} was not rescheduled"
    window_start = str(assertion["window_start"])
    window_end = str(assertion["window_end"])
    if str(event.get("end")) <= window_start or str(event.get("start")) >= window_end:
        return True, f"event {event_id} was moved outside the prohibited window"
    return False, f"event {event_id} remains inside the prohibited window"


def _event_comment_content(
    assertion: dict[str, Any], context: dict[str, Any]
) -> tuple[bool, str]:
    event_id = str(assertion["event_id"])
    event = context["state"].get("events", {}).get(event_id)
    if event is None:
        return False, f"no event {event_id}"
    for comment in event.get("comments", []):
        if not _missing_keywords(str(comment), assertion.get("includes", [])):
            return True, f"event {event_id} has a matching comment"
    return False, f"event {event_id} has no comment with the required content"


def _event_attendees(
    assertion: dict[str, Any], context: dict[str, Any]
) -> tuple[bool, str]:
    event_id = str(assertion["event_id"])
    event = context["state"].get("events", {}).get(event_id)
    if event is None:
        return False, f"no event {event_id}"
    attendees = {str(email).casefold() for email in event.get("attendees", [])}
    includes = {str(email).casefold() for email in assertion.get("includes", [])}
    excludes = {str(email).casefold() for email in assertion.get("excludes", [])}
    missing = sorted(includes - attendees)
    present = sorted(excludes & attendees)
    if missing or present:
        details = []
        if missing:
            details.append("missing " + ", ".join(missing))
        if present:
            details.append("unexpected " + ", ".join(present))
        return False, f"event {event_id} attendees: {'; '.join(details)}"
    return True, f"event {event_id} attendees match"


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


def _constraint_compliance(
    assertion: dict[str, Any], context: dict[str, Any]
) -> tuple[bool, str]:
    """Evaluate a bounded family of compliant actions and explicit violations.

    This is intentionally a scenario-level contract, not a tool-call matcher:
    an allowed outcome may be a draft, a sent restraint, or a resolved routing
    action.  Every alternative still names the governed thread/event, while a
    forbidden outcome vetoes an otherwise plausible action.
    """

    allowed = assertion.get("allowed_outcomes", [])
    forbidden = assertion.get("forbidden_outcomes", [])
    if not allowed:
        raise AssertionKindError("constraint_compliance needs allowed_outcomes")
    passed_allowed, allowed_detail = _any_of({"assertions": allowed}, context)
    violations = []
    for outcome in forbidden:
        kind = str(outcome.get("kind", ""))
        checker = _CHECKS.get(kind)
        if checker is None:
            raise AssertionKindError(f"unknown assertion kind: {kind!r}")
        passed, detail = checker(outcome, context)
        if passed:
            violations.append(f"{kind}: {detail}")
    if violations:
        return False, "forbidden outcome occurred (" + "; ".join(violations) + ")"
    if passed_allowed:
        return True, allowed_detail
    return False, allowed_detail


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
    "message_sent": _message_sent,
    "thread_message_content": _thread_message_content,
    "follow_up_assigned": _follow_up_assigned,
    "follow_up_routed": _follow_up_routed,
    "event_response": _event_response,
    "event_rescheduled": _event_rescheduled,
    "event_rescheduled_outside_window": _event_rescheduled_outside_window,
    "event_comment_content": _event_comment_content,
    "event_attendees": _event_attendees,
    "constraint_compliance": _constraint_compliance,
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
