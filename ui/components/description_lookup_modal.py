"""One character, one source, on demand - the dialog the tab and the editor share.

The bulk run answers "describe everyone else"; this answers "describe *this* one",
which is what a principal in the middle of a book, or a walk-on in the character
editor, actually needs. Four sources, one modal:

* **Web (MCP)** - the MCP server from Settings, with an editable first query.
* **Wiki** - one page through the configured wiki's API.
* **Book text** - the direct book lookup: paragraphs that name them, a yes/no
  relevance gate, extraction. Facts only.
* **Deduce Vibe** - the same book pass, then a casting pass that gives the empty
  traits a look which fits their role. The only source that invents, and it says so.

Every source writes through ``character_research.apply_source_fields``, so the rule
is the same everywhere: **Base event, empty fields only, ``source="manual"``**, and
a disagreement is reported rather than written. The dialog owns the run, and the
host refreshes when it closes - never mid-run, because refreshing the panel that
owns a running handler deletes that handler's element (that was a real, long-lived
bug: the editor's form never updated because the click handler died under it).
"""
from __future__ import annotations

import asyncio
from typing import Any, Callable, Dict, List, Optional

from nicegui import ui
from sqlmodel import Session, select

from database.connection import engine, get_setting
from database.models import Book, Character, CharacterAlias, Project
from services import character_research
from ui import state

SOURCES = (
    ("web", "Web (MCP)", "cloud"),
    ("wiki", "Wiki", "travel_explore"),
    ("book", "Book text", "auto_stories"),
    ("deduce", "Deduce Vibe", "theater_comedy"),
)


def default_web_query(character_name: str, series: str) -> str:
    """The first search a single-character web lookup runs.

    The name is quoted because an unquoted one drifts: a two-word name matches the
    actor, the television episode and every page that lists the cast, and a single
    given name matches half the book. Quoting it is a hint the model can see and
    reword, which is exactly the amount of control worth offering.
    """
    series = (series or "").strip()
    return f'"{character_name}" {series} appearance description'.strip()


def default_wiki_query(character_name: str, names: List[str], term: str) -> str:
    """The wiki search a single-character lookup starts with."""
    from services.wiki_lookup import name_bearing_probe
    probe = name_bearing_probe(character_name, names)
    return f'"{probe}" {term}'.strip()


def project_books(project_id: int) -> List[Book]:
    with Session(engine) as session:
        return list(session.exec(select(Book).where(Book.project_id == project_id)
                                 .order_by(Book.book_order)).all())


def available_lookup_sources(project: Project) -> Dict[str, str]:
    """Which single-character sources can run, and what stops the others.

    A disabled button with a reason is better than a button that silently does
    nothing, and better than hiding it: the reason is the instruction.
    """
    from pathlib import Path
    mcp_url = (get_setting("mcp_url", "") or "").strip()
    wiki_url = (getattr(state, "wiki_url", "") or "").strip()
    base_dir = Path(get_setting("output_dir", "./output")).resolve()
    has_transcript = any(
        (base_dir / project.name / book.name / "transcript.txt").exists()
        for book in project_books(project.id))
    book_reason = "" if has_transcript else "no transcripts on disk"
    return {
        "web": "" if mcp_url else "no MCP server in Settings",
        "wiki": "" if wiki_url else "no wiki URL set in Fill Descriptions",
        "book": book_reason,
        "deduce": book_reason,
    }


def open_lookup_dialog(project: Project, char: Character,
                       refresh_callback: Optional[Callable[[], Any]] = None,
                       initial_source: str = "web") -> None:
    """Search one character in one source, showing the query before it is sent.

    The query is shown and editable because the search is the part a user can
    actually improve: a model will not know that the project means a particular
    book, and a quoted name is the cheapest way to say so.
    """
    from services import description_extract, wiki_lookup, web_lookup

    llm_url = get_setting("llm_url", "http://127.0.0.1:11434")
    model = get_setting("llm_model", "local-model")
    mcp_url = (get_setting("mcp_url", "") or "").strip()
    mcp_auth = get_setting("mcp_auth_header", "") or ""
    mcp_prefs = get_setting("mcp_tool_prefs", {}) or {}
    wiki_url = (getattr(state, "wiki_url", "") or "").strip()
    wiki_term = (getattr(state, "wiki_term", "") or project.name).strip()
    series = (getattr(state, "mcp_series_name", "") or project.name).strip()

    with Session(engine) as session:
        aliases = [a.alias for a in session.exec(
            select(CharacterAlias).where(CharacterAlias.character_id == char.id)).all()
            if (a.alias or "").strip()]
    names = list(dict.fromkeys([char.name, *aliases]))

    availability = available_lookup_sources(project)
    run_state = {"source": initial_source if not availability.get(initial_source)
                 else "", "running": False, "changed": False}
    elements: Dict[str, Any] = {"query_input": None, "wiki_query": None}

    def style_sources() -> None:
        for key, button in source_buttons.items():
            active = key == run_state["source"]
            button.classes(
                add=('bg-blue-600 text-white border-blue-600' if active
                     else 'bg-white text-slate-600 border-slate-200'),
                remove=('bg-white text-slate-600 border-slate-200' if active
                        else 'bg-blue-600 text-white border-blue-600'))

    def set_source(source: str) -> None:
        if run_state["running"] or availability.get(source):
            return
        run_state["source"] = source
        style_sources()
        query_panel.clear()
        render_query_help()

    def render_query_help() -> None:
        # Opens its own container: the first render happens after the footer is
        # built, and without this the query widgets were appended to the card and
        # landed *below* the Search button.
        source = run_state["source"]
        elements["query_input"] = None
        elements["wiki_query"] = None
        with query_panel:
            if source == "web":
                ui.label('First search query').classes(
                    'text-[10px] font-bold text-slate-400 uppercase tracking-wider')
                query_input = ui.input(value=default_web_query(char.name, series)) \
                    .classes('w-full text-xs').props('dense outlined')
                elements["query_input"] = query_input
                ui.label('Sent as the model\'s starting point. The quotes bias it to '
                         'the exact spelling; it may reword the query if nothing '
                         'comes back, and it opens a result page when a snippet is '
                         'not enough.').classes(
                    'text-[10px] text-slate-400 leading-tight')
            elif source == "wiki":
                ui.label('Page search').classes(
                    'text-[10px] font-bold text-slate-400 uppercase tracking-wider')
                wiki_query = ui.input(
                    value=default_wiki_query(char.name, names, wiki_term)) \
                    .classes('w-full text-xs').props('dense outlined')
                elements["wiki_query"] = wiki_query
                ui.label('Searches this wiki through its API, one request per '
                         'second. The page title must contain the name; a page '
                         'that merely mentions them is refused.').classes(
                    'text-[10px] text-slate-400 leading-tight')
            elif source == "book":
                ui.label('Paragraphs searched').classes(
                    'text-[10px] font-bold text-slate-400 uppercase tracking-wider')
                ui.label(', '.join(names)).classes(
                    'text-xs text-slate-700 bg-slate-50 border rounded px-2 py-1.5 '
                    'w-full')
                ui.label('Every paragraph that names them, in every book with a '
                         'transcript. One yes/no check per passage, then extraction - '
                         'no attribution pass, so a walk-on can be described where a '
                         'named character is not needed. Facts only; nothing invented.'
                         ).classes('text-[10px] text-slate-400 leading-tight')
            elif source == "deduce":
                ui.label('Evidence used').classes(
                    'text-[10px] font-bold text-slate-400 uppercase tracking-wider')
                ui.label(', '.join(names)).classes(
                    'text-xs text-slate-700 bg-slate-50 border rounded px-2 py-1.5 '
                    'w-full')
                ui.label('Reads the paragraphs that name them for facts first, then '
                         'casts a look for every trait still empty, using their '
                         'titles, rank and bearing as evidence. Values it invents are '
                         'recorded in notes so they never look sourced.').classes(
                    'text-[10px] text-slate-400 leading-tight')
            else:
                ui.label('Pick a source above.').classes(
                    'text-[11px] text-slate-400')

    def close_dialog() -> None:
        dialog.close()
        if run_state["changed"] and refresh_callback:
            # The host panel is rebuilt only here. Refreshing it while a handler
            # inside it is still running deletes that handler's element, which is
            # why this dialog owns the work and the host owns the refresh.
            refreshed = refresh_callback()
            if asyncio.iscoroutine(refreshed):
                asyncio.create_task(refreshed)

    def show_result(lines: List[str], solution: str = "negative") -> None:
        for line in lines:
            with results:
                ui.label(line).classes('text-[11px] leading-tight ' + (
                    'text-slate-600' if solution == "negative" else
                    'text-emerald-700 font-semibold'))

    def show_outcome(fields: Dict[str, str], outcome: Dict,
                     detail: List[str]) -> None:
        filled = outcome["filled"]
        if filled:
            progress.set_text(f"{len(filled)} field(s) filled")
        elif outcome["dropped"]:
            progress.set_text("Nothing written - see below.")
        elif outcome["conflicts"]:
            progress.set_text("Nothing new: every field already has a value.")
        else:
            progress.set_text("Nothing new.")
        for field, value in fields.items():
            if field in ("marks", "notes"):
                continue  # reported as "saved to notes" below
            show_result([f"{field.replace('_', ' ')}: {value}"],
                        solution="positive" if field in filled else "negative")
        for item in outcome["rewritten"]:
            show_result([f"rewrote {item['field']}: "
                         f"{item['was']} -> {item['now']}"], solution="negative")
        for conflict in outcome["conflicts"]:
            show_result([f"kept {conflict['field']}: {conflict['was']} "
                         f"(source said {conflict.get('source') or conflict['found']})"],
                        solution="negative")
        for item in outcome["noted"]:
            show_result([f"{item['field']} saved to notes: {item['value']}"],
                        solution="positive")
        for item in outcome["dropped"]:
            show_result([f"not saved - {item['field']}: {item['found']} "
                         f"({item['reason']})"], solution="negative")
        for line in detail:
            show_result([line], solution="negative")

    with ui.dialog() as dialog, ui.card().classes(
            'w-[620px] max-w-[95vw] max-h-[88vh] p-0 rounded-xl flex flex-col '
            'overflow-hidden'):
        with ui.row().classes('w-full items-center gap-3 px-5 pt-4 pb-2 flex-nowrap'):
            with ui.column().classes('gap-0 flex-1 min-w-0'):
                ui.label(f'Find a description: {char.name}').classes(
                    'text-base font-bold text-slate-800 leading-tight truncate')
                ui.label('One source at a time. Only empty fields are written, so '
                         'nothing you have edited is replaced.').classes(
                    'text-[11px] text-slate-500 leading-tight whitespace-normal')
            ui.button(icon='close', on_click=close_dialog).props('flat dense round') \
                .classes('shrink-0 text-slate-400')

        with ui.column().classes('w-full px-5 gap-3 pb-4 overflow-y-auto flex-1 '
                                 'min-h-0'):
            source_buttons: Dict[str, Any] = {}
            with ui.row().classes('w-full gap-2 flex-nowrap'):
                for key, label, icon in SOURCES:
                    reason = availability.get(key, "")
                    button = ui.button(label, icon=icon, color=None,
                                       on_click=lambda _, k=key: set_source(k)) \
                        .props('unelevated dense no-caps')\
                        .classes('text-xs font-bold px-3 py-1.5 rounded-lg border '
                                 'bg-white text-slate-600 border-slate-200 '
                                 'flex-1 min-w-0')
                    if reason:
                        button.disable()
                        button.classes(add='opacity-50')
                        button.tooltip(reason)
                    source_buttons[key] = button

            query_panel = ui.column().classes('w-full gap-1')
            results = ui.column().classes(
                'w-full gap-0 min-h-[60px] max-h-56 overflow-y-auto border '
                'rounded-lg bg-white px-2 py-1.5')
            with results:
                ui.label('Results appear here.').classes(
                    'text-[11px] text-slate-400')

            async def do_lookup() -> None:
                if run_state["running"] or not run_state["source"]:
                    return
                run_state["running"] = True
                run_button.disable()
                results.clear()
                progress.set_text('Searching…')
                source = run_state["source"]
                try:
                    if source in ("book", "deduce"):
                        result = await character_research.research(
                            project.id, char.id, speculate=(source == "deduce"),
                            log=lambda message: progress.set_text(
                                message.replace("[Research] ", "")
                                       .replace("[Lookup] ", "")
                                       .replace("[Book] ", "")
                                       .replace("[Vibe] ", "")))
                        if result.get("ok"):
                            run_state["changed"] = True
                        detail: List[str] = []
                        if result.get("used"):
                            detail.append(f'{len(result["used"])} passages used')
                        invented_values = result.get("invented_values") or {}
                        if not result.get("ok"):
                            progress.set_text("Nothing found.")
                            show_result([result.get("error") or "Nothing was written."],
                                        solution="negative")
                        else:
                            show_outcome(result.get("fields") or {}, result, detail)
                            for field, value in invented_values.items():
                                show_result([f"deduced {field.replace('_', ' ')}: "
                                             f"{value}"], solution="positive")
                        state.add_console_log(
                            f"[Lookup] {char.name} via {source}: "
                            + (", ".join(result.get("filled") or []) or "no change"))
                    else:
                        fields: Dict[str, str] = {}
                        error = ""
                        detail = []
                        if source == "web":
                            found = await web_lookup.lookup_character(
                                char.name, series, mcp_url=mcp_url,
                                mcp_auth=mcp_auth, tool_prefs=mcp_prefs,
                                llm_url=llm_url, model=model,
                                query=(elements["query_input"].value or "").strip()
                                if elements["query_input"] else "",
                                log=lambda message: progress.set_text(
                                    message.replace("[Lookup] ", "")))
                            fields = found.get("fields") or {}
                            error = found.get("error", "")
                            if found.get("sources"):
                                detail.append("Sources: " +
                                              ", ".join(found["sources"][:3]))
                            if found.get("rounds"):
                                detail.append(f"{found['rounds']} model rounds, "
                                              f"{found.get('seconds')}s")
                        else:
                            got = wiki_lookup.lookup(
                                char.name, base=wiki_url, term=wiki_term,
                                aliases=aliases,
                                query=(elements["wiki_query"].value or "").strip()
                                if elements["wiki_query"] else "")
                            if not got["ok"]:
                                error = got["error"]
                            else:
                                detail.append(f'Page: {got["title"]}')
                                extracted = await description_extract.extract_from_text(
                                    char.name, got["block"], llm_url=llm_url,
                                    model=model)
                                fields = extracted.get("fields") or {}
                                error = extracted.get("error", "")
                        if not fields:
                            progress.set_text('Nothing found.')
                            show_result([error or "The source stated nothing about "
                                         "their appearance."], solution="negative")
                        else:
                            outcome = character_research.apply_source_fields(
                                project.id, char.id, fields)
                            if outcome["filled"]:
                                run_state["changed"] = True
                            show_outcome(fields, outcome, detail)
                            state.add_console_log(
                                f"[Lookup] {char.name} via {source}: "
                                + (", ".join(outcome["filled"]) or "no change")
                                + (f" ({error})" if error else ""))
                except Exception as ex:
                    # A locked database or a vanished character must not leave the
                    # Search button disabled for the rest of the session.
                    progress.set_text('Could not save.')
                    show_result([f"{type(ex).__name__}: {ex}"], solution="negative")
                finally:
                    run_state["running"] = False
                    try:
                        run_button.enable()
                    except RuntimeError:
                        pass

            with ui.row().classes('w-full items-center gap-2 flex-nowrap px-5 py-3 '
                                  'border-t shrink-0'):
                run_button = ui.button('Search', icon='search', on_click=do_lookup) \
                    .props('unelevated dense no-caps')\
                    .classes('bg-blue-600 hover:bg-blue-700 text-white font-bold '
                             'text-xs px-4 py-1.5 rounded-lg shrink-0')
                with ui.column().classes('gap-0 flex-1 min-w-0'):
                    progress = ui.label('Ready.').classes(
                        'text-[11px] text-slate-600 leading-tight truncate')
                    ui.label('Writes to the Base profile only.').classes(
                        'text-[10px] text-slate-400 leading-tight')

            render_query_help()
            style_sources()

    dialog.open()
