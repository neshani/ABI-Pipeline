"""Prompt Lab - prompt editor and A/B tester.

Prompts live in ``prompt_templates/shot_roles`` and are edited by *slot*:

    Shared          the common block, spliced with a role via its ``<role>`` marker
    Planner         the standalone Shot Planner prompt (WIDE / CHARACTER / INSERT)
    Single/Wide/Character/Insert   role fragments spliced into Shared

One textarea edits the selected slot; the variant dropdown picks which file that
slot reads and writes. The left rail lists slots only, so it never grows into a
list of every variant.

Edits are held per slot in memory, so switching slots never loses work. The
Test button composes from the *drafts*, not the saved files, so unsaved text
can be tested - the sandbox behaviour the old playground had. Testing a role
composes Shared + role; testing the Planner runs it over random chunks.
"""
from __future__ import annotations

import asyncio
import json
from collections import Counter
from typing import Dict, List

from nicegui import ui

from database.connection import get_setting
from services import prompt_library as pl
from services import shot_planner as sp
from services.prompt_engine import (
    fetch_test_chunks, generate_prompt_for_chunk_async, get_llm_response,
    parse_llm_response,
)
from ui import state

# --- module-level working state (persists across tab switches) -------------
_lab_active_slot = "wide"
_lab_draft: Dict[str, str] = {}
_lab_saved: Dict[str, str] = {}
_lab_results: List[dict] = []
_plan_results: List[dict] = []
_lab_loading = False

_ab_role = "wide"
_ab_a = ""
_ab_b = ""
_ab_results: List[dict] = []
_ab_loading = False

# Full chain (plan -> role -> prompt) run over one chapter.
_chain_book = ""
_chain_chapter = ""
_chain_mode = ""          # "" = use the book's effective (production) mode
_chain_results: List[dict] = []
_chain_meta: Dict[str, object] = {}
_chain_loading = False
_chain_force_decline = False

_SHOT_MODE_OPTIONS = {
    "auto": "Plan shots (auto)",
    "wide": "All WIDE",
    "character": "All CHARACTER",
    "insert": "All INSERT",
    "off": "Custom single prompt",
}

_SLOT_LABELS = {
    "shared": "Shared",
    "shot_direction": "Planner",
    "single": "Single",
    "wide": "Wide",
    "character": "Character",
    "insert": "Insert",
}

_SLOT_BLURB = {
    "shared": "Common block. Spliced with a role via its <role> marker; holds the quote/freeze-frame/output rules.",
    "shot_direction": "Shot Planner. Assigns WIDE / CHARACTER / INSERT to every passage; edits apply on the next plan run.",
    "single": "Fallback role. Used when the planner is off, or when a passage has no parsed shot.",
    "wide": "Shot type WIDE. The place is the subject.",
    "character": "Shot type CHARACTER. People are the subject; bracket-tags proper names.",
    "insert": "Shot type INSERT. One object, close; no people.",
}


def _dirty(slot: str) -> bool:
    return _lab_draft.get(slot, "") != _lab_saved.get(slot, "")


def _load_draft(slot: str, variant: str | None = None) -> None:
    variant = variant or pl.get_active_variant(slot)
    text = pl.read_variant(slot, variant)
    _lab_draft[slot] = text
    _lab_saved[slot] = text


def _ensure_drafts() -> None:
    global _lab_active_slot
    for slot in pl.EDIT_SLOTS:
        if slot not in _lab_draft:
            _load_draft(slot)
    if _lab_active_slot not in pl.EDIT_SLOTS:
        _lab_active_slot = "wide"


# --- editor blocks ---------------------------------------------------------

@ui.refreshable
def render_slot_list() -> None:
    for slot in pl.EDIT_SLOTS:
        active = slot == _lab_active_slot
        mark = " •" if _dirty(slot) else ""
        btn = ui.button(
            f"{_SLOT_LABELS.get(slot, slot)}{mark}",
            on_click=lambda s=slot: _select_slot(s),
        ).props('flat no-caps align=left').classes(
            'w-full justify-start text-xs px-3 py-2 rounded-md ' +
            ('bg-blue-50 text-blue-700 font-bold' if active else 'text-slate-600')
        )
        btn.tooltip(_SLOT_BLURB.get(slot, ""))


def _select_slot(slot: str) -> None:
    """Switch the edited slot. Drafts are per slot, so nothing is lost."""
    global _lab_active_slot
    _lab_active_slot = slot
    _load_draft(slot)
    render_slot_list.refresh()
    render_slot_editors.refresh()
    _render_lab_results.refresh()


@ui.refreshable
def render_slot_editors() -> None:
    _ensure_drafts()
    slot = _lab_active_slot
    variant = pl.get_active_variant(slot)
    variants = pl.list_variants(slot)

    with ui.card().classes('w-full border p-4 shadow-sm bg-white gap-3'):
        with ui.row().classes('w-full items-center justify-between gap-2'):
            with ui.column().classes('gap-0'):
                ui.label(_SLOT_LABELS.get(slot, slot)).classes('text-sm font-bold text-slate-800')
                ui.label(_SLOT_BLURB.get(slot, "")).classes('text-[10px] text-slate-400')
            with ui.row().classes('items-center gap-1'):
                ui.select(
                    options=variants, value=variant, label="Variant",
                    on_change=lambda e, s=slot: _switch_variant(s, e.value),
                ).classes('w-56').props('dense outlined')
                if variant != slot:
                    ui.button(
                        icon='delete',
                        on_click=lambda s=slot, v=variant: _delete_variant(s, v),
                    ).props('flat dense color=red').tooltip('Delete this variant')

        ui.textarea(
            value=_lab_draft.get(slot, ""),
            on_change=lambda e, s=slot: _lab_draft.__setitem__(s, e.value),
        ).classes('w-full font-mono text-xs leading-relaxed').props('outlined autogrow')

        with ui.row().classes('w-full items-center gap-2'):
            with ui.row().classes('items-center gap-1'):
                ui.button('Save', icon='save',
                          on_click=lambda s=slot: _save(s)).props('outline dense')
                name_input = ui.input(placeholder="new name").props('dense outlined').classes('w-36')
                ui.button('Save As', icon='save_as',
                          on_click=lambda s=slot, w=name_input: _save_as(s, w)
                          ).props('outline dense')
                ui.button('Revert', icon='undo',
                          on_click=lambda s=slot: _revert(s)).props('flat dense').classes('text-slate-500')
            if _dirty(slot):
                ui.label('unsaved changes').classes('text-[10px] text-amber-600 font-bold')


def _switch_variant(slot: str, variant: str) -> None:
    pl.set_active_variant(slot, variant)
    _load_draft(slot, variant)
    render_slot_editors.refresh()
    render_slot_list.refresh()
    ui.notify(f"{slot}: active variant '{variant}'", type="info")


def _save(slot: str) -> None:
    variant = pl.get_active_variant(slot)
    pl.save_variant(slot, variant, _lab_draft.get(slot, ""))
    _lab_saved[slot] = _lab_draft.get(slot, "")
    render_slot_editors.refresh()
    render_slot_list.refresh()
    ui.notify(f"Saved {variant}.txt", type="positive")


def _save_as(slot: str, name_input) -> None:
    raw = (name_input.value or "").strip()
    if not raw:
        ui.notify("Enter a name for the new variant.", type="warning")
        return
    stem = pl.variant_stem(slot, raw)
    pl.save_variant(slot, stem, _lab_draft.get(slot, ""))
    pl.set_active_variant(slot, stem)
    _lab_saved[slot] = _lab_draft.get(slot, "")
    name_input.set_value("")
    render_slot_editors.refresh()
    render_slot_list.refresh()
    ui.notify(f"Saved as {stem}.txt and made active", type="positive")


def _revert(slot: str) -> None:
    _load_draft(slot)
    render_slot_editors.refresh()
    render_slot_list.refresh()
    ui.notify(f"Reverted {slot} to the saved file", type="info")


def _delete_variant(slot: str, variant: str) -> None:
    if not pl.delete_variant(slot, variant):
        ui.notify("The base variant cannot be deleted.", type="warning")
        return
    pl.set_active_variant(slot, slot)
    _load_draft(slot, slot)
    render_slot_editors.refresh()
    render_slot_list.refresh()
    ui.notify(f"Deleted {variant}.txt", type="warning")


# --- test bench ------------------------------------------------------------

@ui.refreshable
def render_slot_results() -> None:
    if _lab_loading:
        ui.label("Testing…").classes('text-xs text-slate-500')
        return
    if not _lab_results:
        ui.label("No test run yet.").classes('text-xs text-slate-400')
        return
    for i, res in enumerate(_lab_results, start=1):
        with ui.card().classes('w-full border p-3 shadow-sm bg-white gap-1'):
            ui.label(f"Chunk {i} • {res['pct']}% into volume").classes(
                'text-[10px] font-bold text-slate-500 uppercase')
            ui.label(f'"{res["quote"]}"').classes('text-xs italic text-slate-700')
            ui.label(res["prompt"]).classes('text-xs font-semibold text-blue-700')


@ui.refreshable
def render_plan_results() -> None:
    if _lab_loading:
        ui.label("Planning…").classes('text-xs text-slate-500')
        return
    if not _plan_results:
        ui.label("No plan run yet.").classes('text-xs text-slate-400')
        return
    dist = Counter(r["shot"] for r in _plan_results)
    ui.label("plan: " + "   ".join(f"{k} {v}" for k, v in dist.most_common())).classes(
        'text-xs font-bold text-slate-600')
    for i, res in enumerate(_plan_results, start=1):
        with ui.card().classes('w-full border p-3 shadow-sm bg-white gap-1'):
            with ui.row().classes('items-center gap-2'):
                ui.badge(res["shot"], color='indigo').classes('text-[10px]')
                ui.label(f"chunk {i}").classes('text-[10px] text-slate-400')
            ui.label(" ".join(res["chunk"].split())[:200] + "…").classes(
                'text-[11px] text-slate-500 italic')


@ui.refreshable
def _render_lab_results() -> None:
    """Show the result set that matches the slot currently being edited."""
    if _lab_active_slot == pl.PLANNER_SLOT:
        render_plan_results()
    else:
        render_slot_results()


async def run_slot_test(project_name: str) -> None:
    global _lab_loading
    slot = _lab_active_slot
    if slot == "shared":
        ui.notify("Pick a role or the Planner to test.", type="warning")
        return
    book = state.playground_book_selection
    if not book:
        ui.notify("Select a book volume to test against.", type="warning")
        return

    _lab_loading = True
    _lab_results.clear()
    _plan_results.clear()
    _render_lab_results.refresh()

    chunks = fetch_test_chunks(
        project_name=project_name, book_name=book,
        count=int(state.playground_chunk_count or 2),
        mode="Seeded Random", start_index=0, seed=int(state.playground_seed or 42),
    )
    if not chunks:
        _lab_loading = False
        _render_lab_results.refresh()
        ui.notify("No transcript found for that volume.", type="negative")
        return

    llm_url = get_setting("llm_url", "http://127.0.0.1:11434")
    model = get_setting("llm_model", "local-model")

    if slot == pl.PLANNER_SLOT:
        # Test the unsaved planner draft by planning the sampled chunks with it.
        template = _lab_draft.get(slot, "")
        raw = await get_llm_response(
            sp.plan_prompt(chunks, template), llm_url, model, temperature=0.0,
            extra_payload={"chat_template_kwargs": {"enable_thinking": False}},
        )
        shots = sp.parse_shot_lines(raw, len(chunks))
        for idx, chunk in enumerate(chunks):
            _plan_results.append({"shot": shots.get(idx + 1, "NONE"), "chunk": chunk})
    else:
        shared = _lab_draft.get("shared", "")
        fragment = _lab_draft.get(slot, "")
        for idx, chunk in enumerate(chunks):
            prompt = pl.render_from(shared, fragment, chunk)
            raw = await get_llm_response(
                prompt, llm_url, model, temperature=0.7,
                extra_payload={"chat_template_kwargs": {"enable_thinking": False}},
            )
            parsed = parse_llm_response(raw)
            _lab_results.append({
                "quote": parsed["quote"], "prompt": parsed["prompt"],
                "status": parsed["status"], "pct": 100 * idx // max(1, len(chunks)),
            })

    _lab_loading = False
    _render_lab_results.refresh()
    ui.notify("Test complete.", type="positive")


# --- A/B tab ---------------------------------------------------------------

@ui.refreshable
def render_ab_results() -> None:
    if _ab_loading:
        ui.label("Running A/B…").classes('text-xs text-slate-500')
        return
    if not _ab_results:
        ui.label("Choose two variants and run.").classes('text-xs text-slate-400')
        return
    same = sum(1 for r in _ab_results if r["a"] == r["b"])
    ui.label(f"{same}/{len(_ab_results)} chunks produced identical prompts").classes(
        'text-xs font-bold text-slate-600')
    for i, res in enumerate(_ab_results, start=1):
        with ui.card().classes('w-full border p-3 shadow-sm bg-white gap-2'):
            ui.label(f"Chunk {i}").classes('text-[10px] font-bold text-slate-500 uppercase')
            ui.label(" ".join(res['src'].split())[:180] + "…").classes(
                'text-[11px] text-slate-500 italic')
            with ui.grid(columns=2).classes('w-full gap-3'):
                with ui.column().classes('gap-0'):
                    ui.label("A").classes('text-[10px] font-black text-slate-400')
                    ui.label(res["a"]).classes('text-xs text-slate-800')
                with ui.column().classes('gap-0'):
                    ui.label("B").classes('text-[10px] font-black text-slate-400')
                    ui.label(res["b"]).classes('text-xs text-slate-800')
            ui.badge('identical' if res["a"] == res["b"] else 'differs',
                     color='slate' if res["a"] == res["b"] else 'amber').classes('text-[10px]')


async def run_ab_test(project_name: str) -> None:
    global _ab_loading
    book = state.playground_book_selection
    if not book:
        ui.notify("Select a book volume to test against.", type="warning")
        return
    if not _ab_a or not _ab_b:
        ui.notify("Choose both variants.", type="warning")
        return

    _ab_loading = True
    _ab_results.clear()
    render_ab_results.refresh()

    chunks = fetch_test_chunks(
        project_name=project_name, book_name=book,
        count=int(state.playground_chunk_count or 2),
        mode="Seeded Random", start_index=0, seed=int(state.playground_seed or 42),
    )
    llm_url = get_setting("llm_url", "http://127.0.0.1:11434")
    model = get_setting("llm_model", "local-model")
    shared = pl.shared_text()

    for chunk in chunks:
        prompt_a = pl.render_from(shared, pl.read_variant(_ab_role, _ab_a), chunk)
        prompt_b = pl.render_from(shared, pl.read_variant(_ab_role, _ab_b), chunk)
        raw_a = await get_llm_response(prompt_a, llm_url, model, temperature=0.7,
                                       extra_payload={"chat_template_kwargs": {"enable_thinking": False}})
        raw_b = await get_llm_response(prompt_b, llm_url, model, temperature=0.7,
                                       extra_payload={"chat_template_kwargs": {"enable_thinking": False}})
        _ab_results.append({
            "src": chunk,
            "a": parse_llm_response(raw_a)["prompt"],
            "b": parse_llm_response(raw_b)["prompt"],
        })

    _ab_loading = False
    render_ab_results.refresh()
    ui.notify("A/B complete.", type="positive")


# --- top level -------------------------------------------------------------

def _render_editor_tab(project, books) -> None:
    _ensure_drafts()
    with ui.row().classes('w-full gap-4 items-start'):
        with ui.column().classes('w-48 shrink-0 gap-1'):
            ui.label("Prompts").classes('text-[10px] font-black text-slate-400 uppercase tracking-wider px-3')
            render_slot_list()
        with ui.column().classes('flex-1 gap-4 min-w-0'):
            render_slot_editors()

            if _lab_active_slot != "shared":
                with ui.card().classes('w-full border p-4 shadow-sm bg-white gap-3'):
                    with ui.row().classes('w-full items-center gap-4'):
                        ui.select(options=[b.name for b in books], label="Book Volume",
                                  value=state.playground_book_selection
                                  ).bind_value(state, 'playground_book_selection').classes('flex-1')
                        ui.number(label="Chunks", value=state.playground_chunk_count, min=1, max=10
                                  ).bind_value_to(state, 'playground_chunk_count').classes('w-24')
                        ui.number(label="Seed", value=state.playground_seed, precision=0
                                  ).bind_value_to(state, 'playground_seed').classes('w-24')
                        ui.button('Run test', icon='bolt', color='positive',
                                  on_click=lambda: asyncio.create_task(run_slot_test(project.name))
                                  ).classes('text-white font-bold text-xs h-10')
                    if _lab_active_slot == pl.PLANNER_SLOT:
                        desc = ("Plans random chunks with the Planner draft   "
                                "(tests unsaved edits; nothing is written)")
                    else:
                        desc = (f"Composes: Shared + {_lab_active_slot.upper()}   "
                                "(tests unsaved edits; nothing is written)")
                    ui.label(desc).classes('text-[10px] text-slate-400')
            _render_lab_results()


def _render_ab_tab(project, books) -> None:
    role_variants = pl.list_variants(_ab_role)
    with ui.card().classes('w-full border p-4 shadow-sm bg-white gap-3'):
        with ui.row().classes('w-full items-center gap-3'):
            ui.select(options=list(pl.SHOT_ROLES), label="Role", value=_ab_role,
                      on_change=lambda e: _set_ab_role(e.value)).classes('w-40')
            ui.select(options=role_variants, label="Variant A", value=_ab_a or role_variants[0],
                      on_change=lambda e: _set_ab('a', e.value)).classes('flex-1')
            ui.select(options=role_variants, label="Variant B", value=_ab_b or role_variants[-1],
                      on_change=lambda e: _set_ab('b', e.value)).classes('flex-1')
        with ui.row().classes('w-full items-center gap-3'):
            ui.select(options=[b.name for b in books], label="Book Volume",
                      value=state.playground_book_selection
                      ).bind_value(state, 'playground_book_selection').classes('flex-1')
            ui.number(label="Chunks", value=state.playground_chunk_count, min=1, max=10
                      ).bind_value_to(state, 'playground_chunk_count').classes('w-24')
            ui.number(label="Seed", value=state.playground_seed, precision=0
                      ).bind_value_to(state, 'playground_seed').classes('w-24')
            ui.button('Run A/B', icon='compare_arrows', color='primary',
                      on_click=lambda: asyncio.create_task(run_ab_test(project.name))
                      ).classes('text-white font-bold text-xs h-10')
    render_ab_results()


def _set_ab_role(role: str) -> None:
    global _ab_role, _ab_a, _ab_b
    _ab_role = role
    variants = pl.list_variants(role)
    _ab_a, _ab_b = variants[0], variants[-1]


def _set_ab(which: str, value: str) -> None:
    global _ab_a, _ab_b
    if which == 'a':
        _ab_a = value
    else:
        _ab_b = value


# --- Full Chain tab ---------------------------------------------------------

def _book_segments(project_name: str, book_name: str) -> List[str]:
    """Transcript segments. segs[N] is book chapter N; segs[0] is front matter.

    Headings are ASR output and unreliable ("CHAPTER XHES OF DRAGONS" is
    chapter 10), so the index is the only safe key.
    """
    text = sp.read_book_text(project_name, book_name) if book_name else None
    if not text:
        return []
    return [s for s in text.replace("\r\n", "\n").split("==CHAPTER==") if s.strip()]


def _chapter_options(segs: List[str]) -> Dict[str, str]:
    return {str(i): f"{i} · {' '.join(s.split()[:6])[:40]}" for i, s in enumerate(segs)}


@ui.refreshable
def render_chain_results() -> None:
    if _chain_loading:
        ui.label("Running plan + prompt extraction…").classes('text-xs text-slate-500')
        return
    if not _chain_results:
        ui.label("No run yet. Choose a book and chapter, then run.").classes('text-xs text-slate-400')
        return

    dist = Counter(r["shot_type"] for r in _chain_results)
    ui.label("plan: " + "   ".join(f"{k} {v}" for k, v in dist.most_common())
             + f"      scenes: {len(_chain_results)}"
             ).classes('text-xs font-bold text-slate-600')
    fell_back = [r for r in _chain_results if r["planned_role"] != r["role"]]
    if fell_back:
        chain = Counter(f"{r['planned_role']} → {r['role']}" for r in fell_back)
        ui.label("fell back: " + "   ".join(f"{k} {v}" for k, v in chain.most_common())
                 ).classes('text-xs font-bold text-amber-700')
    unusable = [r for r in _chain_results if r["status"] != "success"]
    if unusable:
        ui.label(f"no role could render {len(unusable)} scene(s) - "
                 f"they would be stored as REFUSAL"
                 ).classes('text-xs font-bold text-rose-700')

    for res in _chain_results:
        with ui.card().classes('w-full border p-3 shadow-sm bg-white gap-1'):
            with ui.row().classes('items-center gap-2'):
                ui.badge(res["shot_type"], color='indigo').classes('text-[10px]')
                ui.label(f"scene {res['shot']} · role {res['planned_role']}").classes(
                    'text-[10px] text-slate-400')
                if res["planned_role"] != res["role"]:
                    label = f"fell back to {res['role']}"
                    if res.get("forced"):
                        label += " (forced)"
                    ui.badge(label, color='amber').classes('text-[10px]')
                if res["status"] != "success":
                    ui.badge(f"{res['status']} by every role", color='rose').classes('text-[10px]')
            ui.label(" ".join(res["chunk"].split())[:170] + "…").classes(
                'text-[11px] text-slate-500 italic')
            ui.label(f'"{res["quote"]}"').classes('text-xs text-slate-700')
            ui.label(res["prompt"]).classes(
                'text-xs font-semibold '
                + ('text-rose-700' if res["status"] != "success" else 'text-blue-700'))


async def run_full_chain(project_name: str, project_id: int = 0) -> None:
    global _chain_loading
    segs = _book_segments(project_name, _chain_book)
    if not segs or _chain_chapter == "":
        ui.notify("Choose a book and chapter first.", type="warning")
        return
    idx = int(_chain_chapter)
    if not 0 <= idx < len(segs):
        ui.notify("Chapter out of range for that book.", type="negative")
        return

    chunk_words = int(getattr(state, "playground_chunk_size", 350) or 350)
    chunks = sp.smart_chunk_text(segs[idx], chunk_words)
    if not chunks:
        ui.notify("That chapter produced no chunks.", type="negative")
        return

    _chain_loading = True
    _chain_results.clear()
    render_chain_results.refresh()

    llm_url = get_setting("llm_url", "http://127.0.0.1:11434")
    model = get_setting("llm_model", "local-model")
    mode = _chain_mode or pl.get_shot_mode(project_name, _chain_book)
    selection = pl.active_selection()

    # One planner pass, keeping the focus and place it read for each shot rather
    # than just the label: the shared block asks for them ("the director has
    # chosen a focus for this shot: <focus>", "<place_rules>"), and a tab that
    # drops them tests a prompt the pipeline never sends.
    detail = await sp.plan_chunks_detail(chunks, llm_url, model)
    # The production function, not a copy of it: this run has to exercise the
    # same decline-and-fall-back walk the pipeline does, or the tab stops
    # predicting what a real run will produce. Sequential on purpose - one
    # semaphore slot, so the console order matches the scene order.
    semaphore = asyncio.Semaphore(1)
    for i, chunk in enumerate(chunks, start=1):
        shot, focus, place = (detail[i - 1] if i - 1 < len(detail)
                              else ("NONE", "", ""))
        role = pl.resolve_role(shot, mode) or "single"
        task = {
            "chunk": chunk,
            "chapter_num": idx,
            "shot_num": i,
            "shot_role": role,
            "project_id": project_id,
            "selection": selection,
            "focus": focus,
            "place": place,
            "prompt": pl.render_with_selection(role, chunk, selection,
                                               focus=focus, place=place),
        }
        res = await generate_prompt_for_chunk_async(
            task, llm_url, model, semaphore,
            extra_payload={"chat_template_kwargs": {"enable_thinking": False}},
            simulate_decline=_chain_force_decline)
        _chain_results.append({
            "shot": i,
            "shot_type": shot,
            "planned_role": role,
            "role": res["shot_role"],
            "declined_roles": res.get("declined_roles") or [],
            "forced": _chain_force_decline,
            "status": res["status"],
            "chunk": chunk,
            "quote": res["quote"],
            "prompt": res["prompt"],
        })

    _chain_meta.clear()
    _chain_meta.update({
        "project": project_name,
        "book": _chain_book,
        "chapter_index": idx,
        "chapter_head": " ".join(segs[idx].split()[:12]),
        "shot_mode": mode,
        "model": model,
        "chunk_words": chunk_words,
        "active_variants": selection,
        "fallback_roles": dict(pl.FALLBACK_ROLES),
        "forced_decline": _chain_force_decline,
        "plan_counts": dict(Counter(r["shot_type"] for r in _chain_results)),
        "fallback_counts": dict(Counter(
            f"{r['planned_role']}->{r['role']}" for r in _chain_results
            if r["planned_role"] != r["role"] or r["status"] != "success")),
    })
    _chain_loading = False
    render_chain_results.refresh()
    fallbacks = len([r for r in _chain_results if r["planned_role"] != r["role"]])
    if fallbacks:
        ui.notify(f"Full chain complete - {fallbacks} scene(s) fell back to another role.",
                  type="positive")
    else:
        ui.notify("Full chain complete.", type="positive")


def copy_chain_json() -> None:
    """Copy the run as JSON for pasting into an AI chat."""
    if not _chain_results:
        ui.notify("Run the full chain first.", type="warning")
        return
    payload = {**_chain_meta, "scenes": _chain_results}
    ui.clipboard.write(json.dumps(payload, indent=2, ensure_ascii=False))
    ui.notify("Full chain results copied as JSON.", type="positive", icon="content_copy")


def _set_chain(key: str, value) -> None:
    global _chain_book, _chain_chapter, _chain_mode
    if key == "book":
        _chain_book = value
    elif key == "mode":
        _chain_mode = value
    else:
        _chain_chapter = value
    render_chain_panel.refresh()


def _set_chain_force_decline(value: bool) -> None:
    global _chain_force_decline
    _chain_force_decline = bool(value)


@ui.refreshable
def render_chain_panel(project, books) -> None:
    global _chain_book, _chain_chapter
    book_names = [b.name for b in books]
    if _chain_book not in book_names:
        _chain_book = book_names[0] if book_names else ""

    segs = _book_segments(project.name, _chain_book)
    options = _chapter_options(segs)
    if _chain_chapter not in options:
        _chain_chapter = "10" if "10" in options else (next(iter(options), ""))

    with ui.card().classes('w-full border p-4 shadow-sm bg-white gap-3'):
        with ui.row().classes('w-full items-center gap-3'):
            ui.select(options=book_names, label="Book", value=_chain_book,
                      on_change=lambda e: _set_chain("book", e.value)).classes('flex-1')
            ui.select(options=options, label="Chapter", value=_chain_chapter,
                      on_change=lambda e: _set_chain("chapter", e.value)).classes('flex-1')
            ui.select(options={"": "Use book setting", **_SHOT_MODE_OPTIONS}, label="Mode",
                      value=_chain_mode,
                      on_change=lambda e: _set_chain("mode", e.value)).classes('w-48')
            ui.button('Run full chain', icon='account_tree', color='primary',
                      on_click=lambda: asyncio.create_task(
                          run_full_chain(project.name, project.id))
                      ).classes('text-white font-bold text-xs h-10')
        with ui.row().classes('w-full items-center justify-between'):
            ui.label(f"mode: {_chain_mode or pl.get_shot_mode(project.name, _chain_book)}"
                     f"   model: {get_setting('llm_model', '?')}"
                     f"   chunk: {state.playground_chunk_size} words"
                     f"   test only - nothing is written"
                     ).classes('text-[10px] text-slate-400')
            ui.button('Copy results as JSON', icon='content_copy',
                      on_click=copy_chain_json).props('flat dense').classes('text-xs')
        ui.checkbox(
            'Force the planned role to decline (exercises the fallback chain)',
            value=_chain_force_decline,
            on_change=lambda e: _set_chain_force_decline(e.value)
        ).classes('text-[11px] text-slate-600').tooltip(
            'The first answer is replaced with the NONE sentinel, so '
            + ' -> '.join(f"{role}->{'/'.join(dests) or 'nothing'}"
                          for role, dests in pl.FALLBACK_ROLES.items() if dests)
            + ' runs for real from the second role onward. A decline is not '
              'reliably reproducible on demand, which is why this exists.'
        )

    render_chain_results()


def render_prompt_lab(project, books) -> None:
    global _ab_a, _ab_b
    _ensure_drafts()
    if not _ab_a or _ab_a not in pl.list_variants(_ab_role):
        _ab_a = pl.list_variants(_ab_role)[0]
    if not _ab_b or _ab_b not in pl.list_variants(_ab_role):
        _ab_b = pl.list_variants(_ab_role)[-1]

    with ui.tabs().classes('w-full border-b') as sub:
        t_edit = ui.tab('Editor', icon='edit_note')
        t_ab = ui.tab('A/B Test', icon='compare_arrows')
        t_chain = ui.tab('Full Chain', icon='account_tree')
    with ui.tab_panels(sub, value=t_edit).classes('w-full bg-transparent'):
        with ui.tab_panel(t_edit).classes('p-0'):
            _render_editor_tab(project, books)
        with ui.tab_panel(t_ab).classes('p-0'):
            _render_ab_tab(project, books)
        with ui.tab_panel(t_chain).classes('p-0'):
            render_chain_panel(project, books)
