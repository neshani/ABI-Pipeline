"""Resumable stage work: an append-only partial beside the artifact.

The analysis stages are the expensive ones and they used to be all-or-nothing:
``gather.json``, ``extract.json`` and ``scene_map.json`` are written once, after
the last model call, so stopping a 20-minute segmentation at minute 18 threw all
18 minutes away. This module keeps the finished units on disk as they arrive, so
a stopped or crashed run resumes at the unit after the last one it completed.

Two rules make this safe:

* **A unit carries the digest of everything the stage read.** A partial written
  against one transcript or one scene map can never be resumed into a different
  one - the entries simply do not match, and ``reset_if_stale`` drops them.
* **The artifact is still written once, at the end.** A partial is not an
  artifact: nothing downstream reads it, no stage is recorded in
  ``pipeline.json`` until the stage completes, and a resume therefore cannot
  hand a half-built stage to the next one. "Unknown is never current" (see
  ``pipeline_manifest.stage_current``) already covers the rest.

The file is JSONL rather than a rewritten JSON document for one reason: a
process killed mid-write leaves at most a torn *last line*, which is skipped,
instead of an unparseable file. Appending also costs nothing regardless of how
much has accumulated.
"""
from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, Optional, Tuple, TypeVar


class Cancelled(Exception):
    """The user asked the stage to stop.

    Raising this leaves the checkpoint in place and the artifact unwritten, so
    the stage still reads as not current and can be resumed. Callers that catch
    it should report "stopped", not "failed".
    """


def partial_path(book_dir, stage: str) -> Path:
    """``<book>/<stage>.partial.jsonl`` - beside the artifact it will become."""
    return Path(book_dir) / f"{stage}.partial.jsonl"


def add(book_dir, stage: str, sha: str, key: str, data: Any) -> None:
    """Record one finished unit.

    Never raises: a checkpoint is an optimisation and must not be able to fail a
    run. The ``fsync`` is deliberate - the interesting crashes here are the app
    being killed and the machine losing power, and both lose buffered writes.
    """
    path = partial_path(book_dir, stage)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps({"sha": sha, "key": key, "data": data},
                          ensure_ascii=False)
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(line + "\n")
            fh.flush()
            os.fsync(fh.fileno())
    except Exception:
        pass


def load(book_dir, stage: str, sha: str) -> Dict[str, Any]:
    """``{key: data}`` for the units that are still valid for ``sha``."""
    path = partial_path(book_dir, stage)
    if not path.exists():
        return {}
    out: Dict[str, Any] = {}
    try:
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except Exception:
                    # The torn line from a killed process - and only ever that.
                    continue
                if not isinstance(row, dict) or row.get("sha") != sha:
                    continue
                key = row.get("key")
                if key:
                    out[str(key)] = row.get("data")
    except Exception:
        return {}
    return out


def summary(book_dir, stage: str, sha: str) -> Dict[str, int]:
    """``{"done": n}`` for a UI that wants a progress count, no data attached."""
    return {"done": len(load(book_dir, stage, sha))}


def reset_if_stale(book_dir, stage: str, sha: str) -> bool:
    """Drop a partial built from different inputs. True when it was dropped.

    Without this a changed transcript would leave the old entries on disk
    forever - ignored, but confusing to anyone reading the directory, and a slow
    leak of every abort.
    """
    path = partial_path(book_dir, stage)
    if not path.exists():
        return False
    try:
        with open(path, "r", encoding="utf-8") as fh:
            first = fh.readline().strip()
        if not first:
            path.unlink()
            return True
        recorded = json.loads(first).get("sha")
        if recorded != sha:
            path.unlink()
            return True
    except Exception:
        # Unreadable: it is worth less than a re-run. Drop it.
        try:
            path.unlink()
        except Exception:
            pass
        return True
    return False


def discard(book_dir, stage: str) -> None:
    """Remove a partial - called once the artifact it describes is written."""
    path = partial_path(book_dir, stage)
    try:
        if path.exists():
            path.unlink()
    except Exception:
        pass


def stopped(should_cancel: Optional[callable] = None) -> None:
    """Raise :class:`Cancelled` when the caller's predicate says to stop.

    A no-op with no predicate, so every stage keeps working when nothing is
    watching for a stop (the probes and the docs harness do not pass one).
    """
    if should_cancel is not None and should_cancel():
        raise Cancelled()


T = TypeVar("T")


async def await_cancellable(awaitable: Awaitable[T],
                            should_cancel: Optional[Callable[[], bool]] = None,
                            poll: float = 0.25) -> T:
    """Await ``awaitable``, raising :class:`Cancelled` as soon as the predicate says so.

    Checking between units is enough for a stage that owns a loop, but one LLM
    request can take a minute or more (a chapter's worth of shot planning, a
    long prompt), and that single request is where Stop looks dead. This races
    the request against a cheap poll, so the wait is interruptible no matter
    whether the caller's transport would honour a cancellation by itself.

    The in-flight request is cancelled, not awaited to completion: the point of
    a stop is that it stops paying for tokens now, and a lost reply is worthless
    to a run that is not continuing. No predicate means plain ``await``.
    """
    if should_cancel is None:
        return await awaitable
    if should_cancel():
        _close(awaitable)
        raise Cancelled()

    task = asyncio.ensure_future(awaitable)
    try:
        while True:
            done, _pending = await asyncio.wait({task}, timeout=poll)
            if done:
                return task.result()
            if should_cancel():
                raise Cancelled()
    finally:
        if not task.done():
            task.cancel()
            # ``return_exceptions`` so a cancelled transport cannot replace the
            # Cancelled we are already unwinding with.
            await asyncio.gather(task, return_exceptions=True)


def _close(awaitable) -> None:
    """Close a coroutine that was never started, so Python does not warn."""
    close = getattr(awaitable, "close", None)
    if close is not None:
        try:
            close()
        except Exception:
            pass
