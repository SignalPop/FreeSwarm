"""The messages behind the Team collaboration panel's counts.

Every objective iteration the swarm runner posts a collaboration record to #team
(`meta.collab`, see `record_collaboration` in swarm_runner.py) naming the messages the
agent's inbox held (`inbox_seqs`; older runners kept only the count `inbox`), the ones it
answered in that iteration (`answered`) and the ones it posted (`messages_sent`). Over the
last TEAM_WINDOW #team messages, per model, the panel shows:

* **sent**       -- sum of `len(collab.messages_sent)`: what the model's agents posted during
                    their iterations (team_post messages and, since 10-01 19:04, the replies
                    `Worker.answer_feedback` posts before the work).
* every message that was in one of the model's inboxes, counted ONCE (both agents of a model
  read the same messages, and a message stays in the inbox until answered), in one of:

  - **answered**   -- some agent of the model ("Model" or "Model #2": board author, or
                      `meta.agent`) replied to it anywhere on the board: a post with
                      `reply_to` = the message or the message in `meta.answers` -- in that
                      iteration or any later one -- or a record lists it under `answered`.
  - **expired**    -- not answered, and the runner will never answer it (with the reason):
                        `ack`     an acknowledgement (`meta.feedback_reply` = accept): needs no
                                  answer (`_feedback_closed`);
                        `depth`   it replies to one of the model's answers at
                                  `feedback_depth` >= FEEDBACK_MAX_DEPTH: the thread is closed;
                        `age`     posted more than FEEDBACK_MAX_AGE_S (24 h) ago:
                                  answer_feedback skips it;
                        `reread`  posted more than INBOX_WINDOW_S (6 h) ago and already read:
                                  an agent's inbox only holds messages posted since its last
                                  read, and after a restart only the last 6 h, so no agent of
                                  the model is handed it again.
  - **unanswered** -- the rest: not answered and still answerable -- an agent of the model can
                      be handed it again and answer it (its next iteration, or after a
                      restart).

The per-iteration view survives only as provenance: each message carries the iteration that
counted it (the one that answered it, else the newest that read it) and how many read it.
Before 10-01 the panel summed `inbox - len(answered)` per record, so a message read by both
agents counted twice, a reply in a later iteration never cleared it and a three-day-old
message stayed "unanswered" for as long as its record was in the window.

Records from an older runner carry only the inbox count; their inbox is rebuilt the way that
runner built it: the iteration's start is the agent's "Iteration on ..." note in #general
(posted right after the inbox read, by the same agent id), the previous read is its previous
"Iteration on" / "Mentoring: reading" note. What cannot be rebuilt is counted (by the
record's age) and reported as unlocated.

Pure functions over plain message dicts (the board's row shape, `meta` decoded), so the
board routes and the tests share one implementation.
"""

from __future__ import annotations

import bisect
import re
import time

TEAM_WINDOW = 300          # the Team panel reads board.tail('team', 300)
INBOX_TAIL = 150           # the old runner's SwarmAgent.inbox(): /mb/messages?tail=150
INBOX_CAP = 8              # ... return out[-8:]
INBOX_FALLBACK_S = 3600    # ... since = self._inbox_since or (time.time() - 3600)
ITER_MARK = "Iteration on"
MENTOR_MARK = "Mentoring: reading"
# The runner's answering rules (swarm_runner.py; tests/test_team_threads.py checks they match).
FEEDBACK_MAX_AGE_S = 24 * 3600.0   # answer_feedback skips older messages
FEEDBACK_MAX_DEPTH = 2             # a reply to an answer at this depth is not answered
INBOX_WINDOW_S = 6 * 3600.0        # a (re)started agent's first inbox reaches back this far

EXPIRED_WHY = {
    "ack": "an acknowledgement (accept) -- it needs no answer",
    "depth": f"the thread is closed: it replies to an answer at depth {FEEDBACK_MAX_DEPTH}",
    "age": f"older than the {FEEDBACK_MAX_AGE_S / 3600:.0f} h answering window -- the runner skips it",
    "reread": (f"read and not answered then, and older than {INBOX_WINDOW_S / 3600:.0f} h: no agent's inbox "
               "holds it again"),
}
KINDS = ("sent", "answered", "unanswered", "expired")

_REF = re.compile(r"(?:#|\bcandidates?\s+#?)(\d{1,6})\b", re.I)


def _short(model: str) -> str:
    return (model or "?").split("/")[-1]


def _base(name: object) -> str:
    """'Model #2' -> 'Model': the model an agent name belongs to."""
    return str(name or "").split(" #")[0].strip()


def _int(v: object) -> int | None:
    try:
        return int(str(v).strip().lstrip("#")) if v not in (None, "", False) else None
    except (TypeError, ValueError):
        return None


def _reply_target(m: dict) -> int | None:
    r = m.get("reply_to")
    if r is None:
        r = (m.get("meta") or {}).get("reply_to")
    return _int(r)


reply_target = _reply_target


def _answers(m: dict) -> set[int]:
    """The messages a post answers: its reply_to and its `meta.answers`."""
    out = {_int(n) for n in (m.get("meta") or {}).get("answers") or []}
    out.add(_reply_target(m))
    out.discard(None)
    return out  # type: ignore[return-value]


def by_model(m: dict, model: str) -> bool:
    """Posted by one of `model`'s agents: the board author is the model (or "<model> #2"), or
    `meta.agent` names one of its agents."""
    return _base(m.get("author")) == model or _base((m.get("meta") or {}).get("agent")) == model


def _feedback_depth(meta: dict) -> int:
    """swarm_runner._feedback_depth: 1 answers a message, 2 answers a reply to an answer."""
    d = _int(meta.get("feedback_depth"))
    return d if d is not None and d > 0 else (1 if meta.get("feedback_reply") else 0)


def addressed_to(m: dict, agent: str) -> bool:
    """The old runner's inbox test: `meta.to` names the agent or the text @mentions it."""
    meta = m.get("meta") or {}
    return meta.get("to") == agent or f"@{_short(agent).lower()}" in str(m.get("content", "")).lower()


def is_marker(m: dict, agent: str, author_id: str | None) -> bool:
    """A note the agent posts right after reading its inbox (iteration or mentor pass)."""
    if m.get("author") != agent or m.get("channel") != "general":
        return False
    if author_id and m.get("author_id") != author_id:
        return False
    c = str(m.get("content", ""))
    return c.startswith(ITER_MARK) or c.startswith(MENTOR_MARK)


def collab_records(team: list[dict], agent: str) -> list[dict]:
    return [m for m in team if ((m.get("meta") or {}).get("collab") or {}).get("agent") == agent]


def models(team: list[dict]) -> list[str]:
    """The models with a collaboration record in the window, in order of first appearance."""
    seen: dict[str, None] = {}
    for m in team:
        a = ((m.get("meta") or {}).get("collab") or {}).get("agent")
        if a:
            seen.setdefault(a, None)
    return list(seen)


def empty_counts() -> dict:
    return {"iterations": 0, "sent": 0, "answered": 0, "unanswered": 0, "expired": 0}


def referenced_seqs(recs: list[dict]) -> list[int]:
    """Board messages the records name (inbox and answered): the board must reach them."""
    out: list[int] = []
    for r in recs:
        c = r["meta"]["collab"]
        out += [s for s in (_int(x) for x in c.get("inbox_seqs") or []) if s is not None]
        out += [s for s in (_int(a.get("seq")) for a in c.get("answered") or [] if isinstance(a, dict)) if s is not None]
    return out


class _Board:
    """Project messages sorted by seq, with the lookups the classification needs."""

    def __init__(self, board: list[dict], markers: list[dict] | None):
        self.rows = sorted(board, key=lambda m: m["seq"])
        self.seqs = [m["seq"] for m in self.rows]
        self.by_seq = {m["seq"]: m for m in self.rows}
        self.markers = sorted(markers if markers is not None else self.rows, key=lambda m: m["seq"])
        self._answers: dict[str, dict[int, list[dict]]] = {}
        self._marks: dict[str, list[dict]] = {}
        # candidate number -> (objective, candidate id), from board messages that carry both
        self.cands: dict[tuple[str | None, int], str] = {}
        for m in self.rows:
            meta = m.get("meta") or {}
            cid = meta.get("candidate_id")
            seq = meta.get("seq") if isinstance(meta.get("seq"), int) else (meta.get("collab") or {}).get("candidate")
            if cid and isinstance(seq, int):
                self.cands[(meta.get("objective_id"), seq)] = cid

    def marks(self, agent: str) -> list[dict]:
        """The agent's inbox-read notes (any of its agent ids), oldest first."""
        if agent not in self._marks:
            self._marks[agent] = [m for m in self.markers if is_marker(m, agent, None)]
        return self._marks[agent]

    def before(self, seq: int, n: int) -> list[dict]:
        i = bisect.bisect_left(self.seqs, seq)
        return self.rows[max(0, i - n):i]

    def answers_by(self, model: str) -> dict[int, list[dict]]:
        """message seq -> the posts by `model`'s agents answering it (reply_to / meta.answers)."""
        if model not in self._answers:
            idx: dict[int, list[dict]] = {}
            for m in self.rows:
                if by_model(m, model):
                    for n in _answers(m):
                        if n < m["seq"]:
                            idx.setdefault(n, []).append(m)
            self._answers[model] = idx
        return self._answers[model]

    def first_answer(self, target: int, model: str) -> dict | None:
        got = self.answers_by(model).get(target)
        return got[0] if got else None

    def replies(self, target: int, model: str) -> list[dict]:
        """Teammates' replies to `target` (not by the model's own agents)."""
        i = bisect.bisect_right(self.seqs, target)
        return [m for m in self.rows[i:] if not by_model(m, model) and _reply_target(m) == target]


def _iteration_window(b: _Board, agent: str, rec: dict) -> tuple[dict | None, float | None, dict | None]:
    """(the iteration's start marker, the ts of the inbox read before it, that read's marker)."""
    aid = rec.get("author_id")
    mine = [m for m in b.marks(agent) if m["seq"] < rec["seq"] and (not aid or m.get("author_id") == aid)]
    starts = [m for m in mine if str(m.get("content", "")).startswith(ITER_MARK)]
    if not starts:
        return None, None, None
    start = starts[-1]
    prev = [m for m in mine if m["seq"] < start["seq"]]
    return start, (prev[-1]["ts"] if prev else start["ts"] - INBOX_FALLBACK_S), (prev[-1] if prev else None)


def _old_inbox(b: _Board, agent: str, start: dict | None, since: float | None) -> list[dict] | None:
    """An older runner's inbox, rebuilt (its record kept only the count)."""
    if start is None:
        return None
    got = [m for m in b.before(start["seq"], INBOX_TAIL)
           if m["ts"] > since and m.get("author") != agent and addressed_to(m, agent)]
    return got[-INBOX_CAP:]


def _refs(b: _Board, m: dict) -> list[dict]:
    """Candidates a message points at: its own tag, plus `#123` / `candidate 123` in the text
    when the board has seen that candidate number for the message's objective."""
    meta = m.get("meta") or {}
    oid = meta.get("objective_id")
    out: list[dict] = []
    if meta.get("candidate_id"):
        seq = meta.get("seq") if isinstance(meta.get("seq"), int) else (meta.get("collab") or {}).get("candidate")
        out.append({"objective_id": oid, "candidate_id": meta["candidate_id"], "seq": seq})
    seen = {r["candidate_id"] for r in out}
    nums = [int(n) for n in _REF.findall(str(m.get("content", "")))]
    rt = _reply_target(m)
    if rt is not None and rt not in b.by_seq:  # agents often put a candidate number in reply_to
        nums.append(rt)
    for n in nums:
        cid = b.cands.get((oid, int(n)))
        if cid and cid not in seen:
            seen.add(cid)
            out.append({"objective_id": oid, "candidate_id": cid, "seq": int(n)})
    return out[:8]


def _msg(b: _Board, m: dict, fallback: dict | None = None) -> dict:
    """A board message as the drawer shows it. `agent`: which of the author's agents posted it."""
    if m.get("missing"):
        f = fallback or {}
        return {"seq": m["seq"], "located": False, "from": f.get("from"), "agent": f.get("from"), "to": None,
                "channel": f.get("channel"), "ts": None, "text": f.get("text") or "",
                "objective_id": None, "candidate_id": None, "refs": [], "reply_to": None}
    meta = m.get("meta") or {}
    return {"seq": m["seq"], "located": True, "from": m.get("author"),
            "agent": meta.get("agent") or m.get("author"), "to": meta.get("to"),
            "channel": m.get("channel"), "ts": m.get("ts"), "text": str(m.get("content", "")),
            "objective_id": meta.get("objective_id"), "candidate_id": meta.get("candidate_id"),
            "refs": _refs(b, m), "reply_to": _reply_target(m)}


def _match_sent(b: _Board, agent: str, s: dict, lo: int, hi: int, used: set[int]) -> dict | None:
    """The board message a `messages_sent` entry was posted as (the entry keeps 160 chars)."""
    text = str(s.get("text") or "").strip()
    i, j = bisect.bisect_right(b.seqs, lo), bisect.bisect_left(b.seqs, hi)
    for m in b.rows[i:j]:
        meta = m.get("meta") or {}
        if m["seq"] in used or m.get("author") != agent or not meta.get("team"):
            continue
        if s.get("channel") and m.get("channel") != s["channel"]:
            continue
        if _reply_target(m) != s.get("reply_to"):
            continue
        body = str(m.get("content", ""))
        if text and text[:120] not in body:
            continue
        return m
    return None


def _why_expired(b: _Board, m: dict, model: str, now: float) -> str | None:
    """Why the runner will never answer inbox message `m` (a key of EXPIRED_WHY), or None."""
    meta = m.get("meta") or {}
    if meta.get("feedback_reply") == "accept":
        return "ack"
    rt = _reply_target(m)
    own = b.by_seq.get(rt) if rt is not None else None
    if own is not None and by_model(own, model) and _feedback_depth(own.get("meta") or {}) >= FEEDBACK_MAX_DEPTH:
        return "depth"
    age = now - float(m.get("ts") or 0)
    if age > FEEDBACK_MAX_AGE_S:
        return "age"
    if age > INBOX_WINDOW_S:
        return "reread"
    return None


def _why_by_age(age: float) -> str | None:
    """The same rule for a message only known to be older than `age` (its record's age)."""
    return "age" if age > FEEDBACK_MAX_AGE_S else "reread" if age > INBOX_WINDOW_S else None


def _iter(rec: dict) -> dict:
    c = rec["meta"]["collab"]
    return {"record_seq": rec["seq"], "ts": rec.get("ts"), "candidate": c.get("candidate"),
            "candidate_id": rec["meta"].get("candidate_id"), "objective_id": rec["meta"].get("objective_id"),
            "mode": c.get("mode"), "agent": rec["meta"].get("agent") or c.get("agent")}


def threads(team: list[dict], board: list[dict], agent: str, markers: list[dict] | None = None,
            now: float | None = None, detail: bool = True, b: _Board | None = None) -> dict:
    """The messages behind one model's counts (see the module docstring), newest first.

    `team`: the #team window the panel counted (oldest first). `board`: the project's
    messages from the oldest message a record names on (all channels; also the model's posts
    after them, which answer them). `markers`: optionally the agent's #general notes if
    `board` does not reach back far enough to hold them (older runners' records only).
    `detail=False` returns the counts only (the lists stay empty).
    """
    now = time.time() if now is None else now
    b = b or _Board(board, markers)
    recs = collab_records(team, agent)
    counts = empty_counts()
    lists: dict[str, list[dict]] = {k: [] for k in KINDS}
    unlocated = {k: 0 for k in KINDS}

    # message seq -> what the records say about it
    seen: dict[int, dict] = {}

    def note(m: dict, rec: dict, answered_entry: dict | None = None) -> None:
        e = seen.setdefault(m["seq"], {"m": m, "recs": [], "answered_in": None, "copy": None})
        if e["m"].get("missing") and not m.get("missing"):
            e["m"] = m
        e["recs"].append(rec)
        if answered_entry is not None:
            e["answered_in"] = e["answered_in"] or rec
            e["copy"] = e["copy"] or answered_entry

    prev_rec_seq = 0
    last_by_agent: dict[str, int] = {}
    for rec in recs:
        c = rec["meta"]["collab"]
        counts["iterations"] += 1
        counts["sent"] += len(c.get("messages_sent") or [])
        has_seqs = isinstance(c.get("inbox_seqs"), list)
        start, since, prev_mark = (_iteration_window(b, agent, rec) if (detail or not has_seqs) else (None, None, None))

        ans = [a for a in c.get("answered") or [] if isinstance(a, dict) and _int(a.get("seq")) is not None]
        ans_seqs = {_int(a["seq"]) for a in ans}
        for a in ans:
            s = _int(a["seq"])
            note(b.by_seq.get(s) or {"seq": s, "missing": True}, rec, a)
        if has_seqs:
            for s in (_int(x) for x in c["inbox_seqs"]):
                if s is not None and s not in ans_seqs:
                    note(b.by_seq.get(s) or {"seq": s, "missing": True}, rec)
        else:
            # An older runner: rebuild the inbox; what cannot be found is counted by the record's age.
            want = max(0, int(c.get("inbox") or 0) - len(ans))
            left = [m for m in (_old_inbox(b, agent, start, since) or []) if m["seq"] not in ans_seqs][-want:] if want else []
            for m in left:
                note(m, rec)
            gone = want - len(left)
            if gone:
                why = _why_by_age(now - float(rec.get("ts") or now))
                kind = "expired" if why else "unanswered"
                counts[kind] += gone
                unlocated[kind] += gone

        # --- sent: the record's messages_sent, matched to the board posts for full text. They
        # follow this agent's previous record (the feedback answers precede the iteration's
        # "Iteration on" note, so that note does not bound them).
        aid = rec.get("author_id")
        if detail:
            it = _iter(rec)
            lo = last_by_agent.get(aid) if aid in last_by_agent else (prev_mark or {}).get("seq", prev_rec_seq)
            used: set[int] = set()
            for s in c.get("messages_sent") or []:
                m = _match_sent(b, agent, s, lo, rec["seq"], used)
                if m:
                    used.add(m["seq"])
                    item = _msg(b, m)
                    item["replies"] = [_msg(b, x) for x in b.replies(m["seq"], agent)][:5]
                else:
                    unlocated["sent"] += 1
                    item = {"seq": None, "located": False, "from": agent, "agent": it["agent"], "to": s.get("to"),
                            "channel": s.get("channel"), "ts": None, "text": str(s.get("text") or ""),
                            "objective_id": rec["meta"].get("objective_id"), "candidate_id": None,
                            "refs": [], "reply_to": s.get("reply_to"), "replies": []}
                item["iteration"] = it
                lists["sent"].append(item)
        prev_rec_seq = rec["seq"]
        if aid:
            last_by_agent[aid] = rec["seq"]

    # --- every inbox message once: answered / expired / unanswered
    for seq, e in seen.items():
        m, newest = e["m"], e["recs"][-1]
        reply = b.first_answer(seq, agent)
        if reply is not None or e["answered_in"] is not None:
            kind, why = "answered", None
        elif m.get("missing"):
            why = _why_by_age(now - float(e["recs"][0].get("ts") or now))
            kind = "expired" if why else "unanswered"
        else:
            why = _why_expired(b, m, agent, now)
            kind = "expired" if why else "unanswered"
        counts[kind] += 1
        if m.get("missing"):
            unlocated[kind] += 1
        if not detail:
            continue
        item = _msg(b, m, e["copy"])
        item.update(iteration=_iter(e["answered_in"] or newest), read_in=len(e["recs"]),
                    reply=_msg(b, reply) if reply is not None else None)
        if kind == "expired":
            item["expired"] = {"reason": why, "text": EXPIRED_WHY[why]}
        lists[kind].append(item)

    def newest_sent(items: list[dict]) -> list[dict]:
        return sorted(items, key=lambda x: (x["iteration"]["record_seq"], x.get("seq") or 0), reverse=True)

    def newest_msg(items: list[dict]) -> list[dict]:
        return sorted(items, key=lambda x: (x.get("seq") or 0), reverse=True)

    return {"agent": agent, "now": now, "counts": counts, "sent": newest_sent(lists["sent"]),
            "answered": newest_msg(lists["answered"]), "unanswered": newest_msg(lists["unanswered"]),
            "expired": newest_msg(lists["expired"]), "unlocated": unlocated,
            "rules": {"answer_window_s": FEEDBACK_MAX_AGE_S, "reread_window_s": INBOX_WINDOW_S,
                      "max_depth": FEEDBACK_MAX_DEPTH, "expired_why": EXPIRED_WHY}}


def summary(team: list[dict], board: list[dict], markers: list[dict] | None = None,
            now: float | None = None) -> dict[str, dict]:
    """Every model's counts over the window (the panel's numbers), from one board load."""
    b = _Board(board, markers)
    return {a: threads(team, board, a, markers, now, detail=False, b=b)["counts"] for a in models(team)}


def first_start_seq(team: list[dict], agent: str, markers: list[dict]) -> int | None:
    """Seq of the first counted iteration's start note -- the board must reach INBOX_TAIL
    project messages before it for an older runner's inbox to be rebuilt."""
    recs = collab_records(team, agent)
    if not recs:
        return None
    start, _, _ = _iteration_window(_Board([], markers), agent, recs[0])
    return start["seq"] if start else recs[0]["seq"]
