"""The messages behind the Team collaboration panel's counts.

Every objective iteration the swarm runner posts a collaboration record to #team
(`meta.collab`, see `record_collaboration` in swarm_runner.py). The console's Team panel
totals, per model, over the last 300 #team messages:

* **sent**       -- sum of `len(collab.messages_sent)`: the `team_post` messages the agent
                    wrote during its iterations.
* **answered**   -- sum of `len(collab.answered)`: messages in that iteration's inbox the
                    agent replied to (a `team_post` with `reply_to` = the message number)
                    in the SAME iteration.
* **unanswered** -- sum of `max(0, collab.inbox - len(collab.answered))`: messages in the
                    iteration's inbox it did not reply to in that iteration.

The inbox is what `SwarmAgent.inbox()` handed the agent at the start of the iteration:
board messages from someone else addressed to it (`meta.to` == the model, or an
`@<short name>` in the text) posted since its previous inbox read, taken from the last 150
project messages and capped at the newest 8. So "unanswered" means *messages teammates
addressed to this model that it read and did not reply to in that iteration* -- not the
model's own questions that nobody answered.

Records written by newer runners carry `inbox_seqs`. Older ones carry only the count, so
the inbox is rebuilt the way the runner built it: the iteration's start is the agent's
"Iteration on ..." note in #general (posted right after the inbox read, by the same agent
id), the previous read is its previous "Iteration on" / "Mentoring: reading" note.

Pure functions over plain message dicts (the board's row shape, `meta` decoded), so the
board route and the tests share one implementation.
"""

from __future__ import annotations

import bisect
import re

TEAM_WINDOW = 300          # the Team panel reads board.tail('team', 300)
INBOX_TAIL = 150           # SwarmAgent.inbox(): /mb/messages?tail=150
INBOX_CAP = 8              # ... return out[-8:]
INBOX_FALLBACK_S = 3600    # ... since = self._inbox_since or (time.time() - 3600)
ITER_MARK = "Iteration on"
MENTOR_MARK = "Mentoring: reading"

_REF = re.compile(r"(?:#|\bcandidates?\s+#?)(\d{1,6})\b", re.I)


def _short(model: str) -> str:
    return (model or "?").split("/")[-1]


def _reply_target(m: dict) -> int | None:
    r = m.get("reply_to")
    if r is None:
        r = (m.get("meta") or {}).get("reply_to")
    try:
        return int(r) if r is not None else None
    except (TypeError, ValueError):
        return None


def addressed_to(m: dict, agent: str) -> bool:
    """The runner's inbox test: `meta.to` names the agent or the text @mentions it."""
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


def counts(records: list[dict]) -> dict:
    """Exactly the Team panel's arithmetic (TeamPanel.tsx)."""
    out = {"iterations": 0, "sent": 0, "answered": 0, "unanswered": 0}
    for r in records:
        c = r["meta"]["collab"]
        ans = len(c.get("answered") or [])
        out["iterations"] += 1
        out["sent"] += len(c.get("messages_sent") or [])
        out["answered"] += ans
        out["unanswered"] += max(0, int(c.get("inbox") or 0) - ans)
    return out


class _Board:
    """Project messages sorted by seq, with the lookups the reconstruction needs."""

    def __init__(self, board: list[dict], markers: list[dict] | None):
        self.rows = sorted(board, key=lambda m: m["seq"])
        self.seqs = [m["seq"] for m in self.rows]
        self.by_seq = {m["seq"]: m for m in self.rows}
        self.markers = sorted(markers if markers is not None else self.rows, key=lambda m: m["seq"])
        # candidate number -> (objective, candidate id), from board messages that carry both
        self.cands: dict[tuple[str | None, int], str] = {}
        for m in self.rows:
            meta = m.get("meta") or {}
            cid = meta.get("candidate_id")
            seq = meta.get("seq") if isinstance(meta.get("seq"), int) else (meta.get("collab") or {}).get("candidate")
            if cid and isinstance(seq, int):
                self.cands[(meta.get("objective_id"), seq)] = cid

    def before(self, seq: int, n: int) -> list[dict]:
        i = bisect.bisect_left(self.seqs, seq)
        return self.rows[max(0, i - n):i]

    def first_reply(self, target: int, author: str) -> dict | None:
        i = bisect.bisect_right(self.seqs, target)
        for m in self.rows[i:]:
            if m.get("author") == author and _reply_target(m) == target:
                return m
        return None

    def replies(self, target: int, exclude: str) -> list[dict]:
        i = bisect.bisect_right(self.seqs, target)
        return [m for m in self.rows[i:] if m.get("author") != exclude and _reply_target(m) == target]


def _iteration_window(b: _Board, agent: str, rec: dict) -> tuple[dict | None, float | None]:
    """(the iteration's start marker, the ts of the inbox read before it)."""
    aid = rec.get("author_id")
    mine = [m for m in b.markers if m["seq"] < rec["seq"] and is_marker(m, agent, aid)]
    starts = [m for m in mine if str(m.get("content", "")).startswith(ITER_MARK)]
    if not starts:
        return None, None
    start = starts[-1]
    prev = [m for m in mine if m["seq"] < start["seq"]]
    return start, (prev[-1]["ts"] if prev else start["ts"] - INBOX_FALLBACK_S)


def _inbox(b: _Board, agent: str, rec: dict, start: dict | None, since: float | None) -> list[dict] | None:
    c = rec["meta"]["collab"]
    if isinstance(c.get("inbox_seqs"), list):
        return [b.by_seq.get(s) or {"seq": s, "missing": True} for s in c["inbox_seqs"]]
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
    """A board message as the drawer shows it."""
    if m.get("missing"):
        f = fallback or {}
        return {"seq": m["seq"], "located": False, "from": f.get("from"), "to": None,
                "channel": f.get("channel"), "ts": None, "text": f.get("text") or "",
                "objective_id": None, "candidate_id": None, "refs": [], "reply_to": None}
    meta = m.get("meta") or {}
    return {"seq": m["seq"], "located": True, "from": m.get("author"), "to": meta.get("to"),
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


def threads(team: list[dict], board: list[dict], agent: str, markers: list[dict] | None = None) -> dict:
    """The messages behind one model's sent / answered / unanswered counts, newest first.

    `team`: the #team window the panel counted (oldest first). `board`: the project's
    messages from well before the first record on (all channels). `markers`: optionally the
    agent's #general notes if `board` does not reach back far enough to hold them.
    """
    b = _Board(board, markers)
    recs = collab_records(team, agent)
    total = counts(recs)
    sent: list[dict] = []
    answered: list[dict] = []
    unanswered: list[dict] = []
    unlocated = {"sent": 0, "answered": 0, "unanswered": 0}
    prev_rec_seq = 0
    for rec in recs:
        c = rec["meta"]["collab"]
        it = {"record_seq": rec["seq"], "ts": rec.get("ts"), "candidate": c.get("candidate"),
              "candidate_id": rec["meta"].get("candidate_id"), "objective_id": rec["meta"].get("objective_id"),
              "mode": c.get("mode")}
        start, since = _iteration_window(b, agent, rec)

        # --- answered: the record names them; show the question and the reply under it
        ans = c.get("answered") or []
        ans_seqs = {a.get("seq") for a in ans}
        for a in ans:
            q = b.by_seq.get(a.get("seq")) or {"seq": a.get("seq"), "missing": True}
            item = _msg(b, q, a)
            if not item["located"]:
                unlocated["answered"] += 1
            r = b.first_reply(a["seq"], agent) if a.get("seq") is not None else None
            item.update(iteration=it, reply=_msg(b, r) if r else None)
            answered.append(item)

        # --- unanswered: the iteration's inbox minus what it answered
        want = max(0, int(c.get("inbox") or 0) - len(ans))
        inbox = _inbox(b, agent, rec, start, since)
        left = [m for m in (inbox or []) if m["seq"] not in ans_seqs]
        if inbox is None or len(left) != want:
            # Could not rebuild this iteration's inbox exactly: show what was found, say how many are missing.
            left = left[-want:] if want else []
            unlocated["unanswered"] += want - len(left)
        for m in left:
            item = _msg(b, m)
            if not item["located"]:
                unlocated["unanswered"] += 1
            later = b.first_reply(m["seq"], agent)
            item.update(iteration=it, reply=None, later_reply=_msg(b, later) if later else None)
            unanswered.append(item)

        # --- sent: the record's messages_sent, matched to the board posts for full text
        lo = start["seq"] if start else prev_rec_seq
        used: set[int] = set()
        for s in c.get("messages_sent") or []:
            m = _match_sent(b, agent, s, lo, rec["seq"], used)
            if m:
                used.add(m["seq"])
                item = _msg(b, m)
                item["replies"] = [_msg(b, x) for x in b.replies(m["seq"], agent)][:5]
            else:
                unlocated["sent"] += 1
                item = {"seq": None, "located": False, "from": agent, "to": s.get("to"),
                        "channel": s.get("channel"), "ts": None, "text": str(s.get("text") or ""),
                        "objective_id": rec["meta"].get("objective_id"), "candidate_id": None,
                        "refs": [], "reply_to": s.get("reply_to"), "replies": []}
            item["iteration"] = it
            sent.append(item)
        prev_rec_seq = rec["seq"]

    def newest(items: list[dict]) -> list[dict]:
        return sorted(items, key=lambda x: (x["iteration"]["record_seq"], x.get("seq") or 0), reverse=True)

    return {"agent": agent, "counts": total, "sent": newest(sent), "answered": newest(answered),
            "unanswered": newest(unanswered), "unlocated": unlocated}


def first_start_seq(team: list[dict], agent: str, markers: list[dict]) -> int | None:
    """Seq of the first counted iteration's start note -- the board must reach INBOX_TAIL
    project messages before it for that iteration's inbox to be rebuilt."""
    recs = collab_records(team, agent)
    if not recs:
        return None
    start, _ = _iteration_window(_Board([], markers), agent, recs[0])
    return start["seq"] if start else recs[0]["seq"]
