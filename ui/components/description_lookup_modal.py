"""One character, one source at a time, on demand - the dialog the tab and the editor share.

The bulk run answers "describe everyone else"; this answers "describe *this* one",
which is what a principal in the middle of a book, or a walk-on in the character
editor, actually needs. Four sources, one modal:

* **Web (MCP)** - the MCP server from Settings, with an editable first query.
* **Wiki** - one page through the configured wiki's API.
* **Book text** - the direct book lookup: paragraphs that name them, a yes/no
  relevance gate, extraction. Facts only.
* **Best guess** - the same book pass, then a casting pass that gives the traits
  the text leaves out a look which fits their role. The only source that invents,
  and it says so.

Nothing is written while a source runs. Every value a source returns is normalised
(``description_runner.preview_fields``) and dropped into a **cart**: one row per
trait, with the value, the source that proposed it and what the profile holds
today. Empty traits start ticked; a trait that already has a value starts unticked,
so a find can never silently replace a value the user chose. The user runs as many
sources as they like and applies the ticked rows once - "check out".

That is a deliberate inversion of the old flow, which wrote whatever came back and
then reported what it had refused. The report was accurate and still useless: the
only answer it offered was "your value blocked this find", with no way to act on
it. The cart also makes cross-source comparison possible, which the old flow could
not express at all.

Two rules survive unchanged:

* **Base profile only.** Everything here lands on the Base event - the snapshot
  every shot starts from. Per-book change points are the theater's timeline, not
  this dialog's. Ticking a row that replaces an existing value propagates the new
  value through the events that still assert the old one, because later events
  hold complete snapshots and would otherwise keep winning.
* **The dialog owns the run; the host refreshes once, on close.** Refreshing the
  panel that owns a running handler deletes that handler's element (a real,
  long-lived bug: the editor's form never updated because the click handler died
  under it).
"""
from __future__ import annotations

import asyncio
from typing import Any, Callable, Dict, List, Optional

from nicegui import ui
from sqlmodel import Session, select

from database.connection import engine, get_setting
from database.models import Book, Character, CharacterAlias, Project
from services import character_research, description_runner
from ui import state

SOURCES = (
    ("web", "Web (MCP)", "cloud"),
    ("wiki", "Wiki", "travel_explore"),
    ("book", "Book text", "auto_stories"),
    ("deduce", "Best guess", "theater_comedy"),
)

# Rendered order and wording. ``description_runner.APPLIED_FIELDS`` decides what
# can be written; this only decides how the row reads.
FIELD_LABELS = {
    "sex": "Sex",
    "species": "Species",
    "age": "Age",
    "build": "Build",
    "hair_color": "Hair colour",
    "hair_style": "Hair style",
    "eye_color": "Eye colour",
    "face": "Face",
    "notes": "Notes",
}


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
                       initial_source: str = "web",
                       on_applied: Optional[Callable[[List[str], Optional[Dict]],
                                                     None]] = None) -> None:
    """Search one character in one or more sources, then apply what the user ticks.

    The query is shown and editable because the search is the part a user can
    actually improve: a model will not know that the project means a particular
    book, and a quoted name is the cheapest way to say so.

    ``on_applied(fields, position)`` is how the host learns what was written -
    ``position`` is None for a Base-profile write. The host needs it to offer the
    stale renders the new values just stranded; this dialog does not know about
    images.
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
    # ``tab`` is a source key or "results". An unavailable initial source leaves
    # nothing selected, which is what shows the query prompt.
    view: Dict[str, Any] = {"tab": initial_source if not availability.get(
        initial_source) else "", "running": False, "changed": False}
    elements: Dict[str, Any] = {"query_input": None, "wiki_query": None}

    # --- the cart ---------------------------------------------------------
    # field -> {"options": [{"source", "value", "invented"}], "pick": int,
    #           "checked": bool}. One row per trait, whatever proposed it.
    cart: Dict[str, Dict[str, Any]] = {}
    problems: List[Dict[str, str]] = []   # found but unstorable, deduped
    current = character_research.base_values(char.id)

    def cart_order() -> List[str]:
        order = [field for field in description_runner.APPLIED_FIELDS if field in cart]
        if "notes" in cart:
            order.append("notes")
        order.extend(field for field in cart if field not in order)
        return order

    def add_candidates(source: str, fields: Dict[str, str],
                       invented: Optional[Dict[str, str]] = None) -> int:
        """Normalise a source's values into the cart; return how many were added.

        Preview first, write never: the row must show the value that will actually
        be stored ("gray", not "grey-haired"), because a checkbox that promises the
        source's wording would look like a lie the moment it was ticked.
        """
        factual = description_runner.preview_fields(fields or {})
        guessed = description_runner.preview_fields(
            {field: value for field, value in (invented or {}).items()
             if field not in (fields or {})})
        added = 0
        for preview, is_guess in ((factual, False), (guessed, True)):
            for field, value in preview["values"].items():
                entry = cart.get(field)
                if entry is None:
                    entry = cart[field] = {
                        "options": [], "pick": 0,
                        # An empty trait is an obvious yes; a trait with a value is
                        # the user's call, and the whole point is that a find cannot
                        # quietly win that argument.
                        "checked": not (current.get(field) or "").strip(),
                    }
                options = [item for item in entry["options"]
                           if item["source"] != source]
                options.append({"source": source, "value": value,
                                "invented": is_guess})
                entry["options"] = options
                entry["pick"] = len(options) - 1
                added += 1
            for field in preview["notes"]:
                entry = cart.get("notes")
                if entry is None:
                    entry = cart["notes"] = {"options": [], "pick": 0,
                                             "checked": True}
                options = [item for item in entry["options"]
                           if item["source"] != source]
                options.append({"source": source, "value": field["value"],
                                "invented": is_guess})
                entry["options"] = options
                entry["pick"] = len(options) - 1
                added += 1
            for line in preview["dropped"] + preview["unknown"]:
                item = {"field": line["field"], "found": line["found"],
                        "reason": line["reason"]}
                if not any(existing["field"] == item["field"]
                           and existing["found"] == item["found"]
                           for existing in problems):
                    problems.append(item)
        return added

    def checked_fields() -> List[str]:
        return [field for field in cart_order() if cart[field]["checked"]]

    def chosen(field: str) -> Dict[str, str]:
        entry = cart[field]
        options = entry["options"]
        if not options:
            return {"source": "", "value": "", "invented": False}
        return options[min(entry["pick"], len(options) - 1)]

    # --- rendering --------------------------------------------------------
    def style_sources() -> None:
        for key, button in source_buttons.items():
            active = key == view["tab"]
            button.classes(
                add=('bg-blue-600 text-white border-blue-600' if active
                     else 'bg-white text-slate-600 border-slate-200'),
                remove=('bg-white text-slate-600 border-slate-200' if active
                        else 'bg-blue-600 text-white border-blue-600'))

    def set_tab(tab: str) -> None:
        if view["running"]:
            return
        if tab != "results" and availability.get(tab):
            return
        view["tab"] = tab
        style_sources()
        if tab != "results":
            # The log belongs to the source that produced it. Leaving it in place
            # meant the Book text run's output sat under the Best guess tab, which
            # reads as if that source had found it.
            results.clear()
            with results:
                ui.label('Results appear here.').classes(
                    'text-[11px] text-slate-400')
        show_tab()

    def show_tab() -> None:
        is_results = view["tab"] == "results"
        source_panel.visible = not is_results
        cart_panel.visible = is_results
        search_footer.visible = not is_results
        apply_footer.visible = is_results
        if not is_results:
            query_panel.clear()
            render_query_help()

    def render_query_help() -> None:
        # Opens its own container: the first render happens after the footer is
        # built, and without this the query widgets were appended to the card and
        # landed *below* the Search button.
        source = view["tab"]
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
                         'proposes a look for every trait still worth a value, using '
                         'their titles, rank and bearing as evidence. Values it '
                         'invents are marked "guess" and recorded in notes when '
                         'applied, so they never look sourced.').classes(
                    'text-[10px] text-slate-400 leading-tight')
            else:
                ui.label('Pick a source above.').classes(
                    'text-[11px] text-slate-400')

    def close_dialog() -> None:
        dialog.close()
        if view["changed"] and refresh_callback:
            # The host panel is rebuilt only here. Refreshing it while a handler
            # inside it is still running deletes that handler's element, which is
            # why this dialog owns the work and the host owns the refresh.
            refreshed = refresh_callback()
            if asyncio.iscoroutine(refreshed):
                asyncio.create_task(refreshed)

    def note(line: str, positive: bool = False) -> None:
        with results:
            ui.label(line).classes('text-[11px] leading-tight ' + (
                'text-slate-600' if not positive else
                'text-emerald-700 font-semibold'))

    # --- the cart view ----------------------------------------------------
    chip_refs: List[Dict[str, Any]] = []

    def style_chip(field: str) -> None:
        entry = cart.get(field)
        if not entry:
            return
        for ref in chip_refs:
            if ref["field"] != field:
                continue
            selected = ref["index"] == entry["pick"]
            ref["element"].classes(
                add=('bg-blue-600 text-white border-blue-600' if selected
                     else 'bg-white text-slate-500 border-slate-200'),
                remove=('bg-white text-slate-500 border-slate-200' if selected
                        else 'bg-blue-600 text-white border-blue-600'))

    def pick_option(field: str, index: int) -> None:
        cart[field]["pick"] = index
        style_chip(field)
        update_apply_summary()

    def toggle_field(field: str, value: bool) -> None:
        cart[field]["checked"] = bool(value)
        update_apply_summary()

    def update_apply_summary() -> None:
        fields = checked_fields()
        if not fields:
            apply_button.disable()
            apply_button.set_text("Apply")
            cart_summary.set_text("Nothing ticked yet.")
            return
        replacing = [field for field in fields
                     if (current.get(field) or "").strip() and field != "notes"]
        apply_button.enable()
        apply_button.set_text(f"Apply {len(fields)} field(s)")
        text = f"{len(fields)} will be written to the Base profile"
        if replacing:
            text += f" · {len(replacing)} replaces an existing value"
        cart_summary.set_text(text)

    def render_cart() -> None:
        cart_box.clear()
        chip_refs.clear()
        fields = cart_order()
        if not fields:
            with cart_box:
                ui.label('No results yet. Run a source above; whatever it finds '
                         'lands here to be reviewed.').classes(
                    'text-[11px] text-slate-400 py-3')
        for field in fields:
            entry = cart[field]
            options = entry["options"]
            active = chosen(field)
            have = (current.get(field) or "").strip()
            with cart_box:
                with ui.column().classes('w-full gap-1 border-b border-slate-100 '
                                         'pb-1.5'):
                    with ui.row().classes('w-full items-center gap-2 flex-nowrap'):
                        ui.checkbox(value=entry["checked"],
                                    on_change=lambda e, f=field: toggle_field(
                                        f, e.value)).props('dense').classes('shrink-0')
                        ui.label(FIELD_LABELS.get(field, field)).classes(
                            'text-[11px] font-bold text-slate-600 w-[74px] shrink-0')
                        if len(options) <= 1:
                            # Wraps: a truncated face or attire is the one thing on
                            # the row the user has to read before ticking it.
                            ui.label(active["value"]).classes(
                                'text-xs text-slate-800 font-semibold min-w-0 '
                                'leading-snug')
                        else:
                            with ui.row().classes('gap-1 flex-nowrap min-w-0'):
                                for index, option in enumerate(options):
                                    chip = ui.label(
                                        f'{option["source"]}: {option["value"]}') \
                                        .classes('text-[10px] px-1.5 py-0.5 rounded '
                                                 'border cursor-pointer truncate') \
                                        .on('click', lambda _, f=field, i=index:
                                            pick_option(f, i))
                                    chip_refs.append({"field": field, "index": index,
                                                      "element": chip})
                        if active.get("invented"):
                            ui.label('guess').classes(
                                'text-[9px] font-bold uppercase tracking-wider '
                                'text-amber-700 bg-amber-50 border border-amber-200 '
                                'rounded px-1 py-0.5 shrink-0')
                        ui.element('div').classes('flex-1 min-w-0')
                        if have:
                            ui.label(f'you have: {have}').classes(
                                'text-[10px] text-slate-400 truncate max-w-[45%] '
                                'shrink-0')
                    if len(options) > 1:
                        style_chip(field)
        if problems:
            with cart_box:
                with ui.column().classes('w-full gap-0.5 pt-2'):
                    ui.label('Found but not storable').classes(
                        'text-[10px] font-bold text-slate-400 uppercase '
                        'tracking-wider')
                    for item in problems:
                        ui.label(f'{FIELD_LABELS.get(item["field"], item["field"])}: '
                                 f'{item["found"]} - {item["reason"]}').classes(
                            'text-[10px] text-slate-400 leading-tight')
        update_apply_summary()

    # --- running a source -------------------------------------------------
    async def do_apply() -> None:
        fields = checked_fields()
        if not fields or view["running"]:
            return
        view["running"] = True
        apply_button.disable()
        values: Dict[str, str] = {}
        replacing: List[str] = []
        invented: List[str] = []
        notes: List[str] = []
        for field in fields:
            option = chosen(field)
            if not option["value"]:
                continue
            if field == "notes":
                notes.append(option["value"])
                continue
            values[field] = option["value"]
            if (current.get(field) or "").strip():
                replacing.append(field)
            if option["invented"]:
                invented.append(field)
        if notes:
            values["notes"] = "; ".join(notes)
        try:
            outcome = character_research.apply_source_fields(
                project.id, char.id, values, overwrite=replacing)
        except Exception as ex:
            apply_button.enable()
            view["running"] = False
            ui.notify(f"Could not save: {type(ex).__name__}: {ex}", type="negative")
            return
        applied = [field for field in outcome["filled"] if field in values]
        applied_invented = [field for field in invented if field in applied]
        if applied_invented:
            # The one field that is stored and never rendered: an invented value
            # must not look sourced six months from now.
            notes_outcome = character_research.apply_source_fields(
                project.id, char.id, {
                    "notes": "look deduced, not stated in the text: "
                             + ", ".join(sorted(applied_invented))})
            if "notes" not in applied and "notes" in notes_outcome["filled"]:
                applied.append("notes")
        if applied and on_applied is not None:
            # The host owns the stale-render offer; this dialog only knows what it
            # wrote. Without this the purge button never appears for a value found
            # here, and the images rendered with the old one stay behind.
            on_applied([field for field in applied if field != "notes"], None)
        view["running"] = False
        if applied:
            view["changed"] = True
            for field in applied:
                if field in values:
                    current[field] = values[field]
            state.add_console_log(
                f"[Lookup] {char.name}: applied {len(applied)} field(s) - "
                + ", ".join(applied))
        unchanged = [field for field in fields
                     if field in values and field not in applied]
        cart.clear()
        problems.clear()
        render_cart()
        results_button.visible = False
        results_button.set_text("Results")
        cart_status.clear()
        with cart_status:
            if applied:
                ui.label(f"Applied {len(applied)} field(s): " + ", ".join(applied)) \
                    .classes('text-[11px] text-emerald-700 font-semibold '
                             'leading-tight')
                for change in outcome.get("replaced") or []:
                    ui.label(f"replaced {change['field']}: {change['was']} -> "
                             f"{change['now']}").classes(
                        'text-[10px] text-slate-500 leading-tight')
            else:
                ui.label("Nothing was written.").classes(
                    'text-[11px] text-slate-600')
            for field in unchanged:
                ui.label(f"{FIELD_LABELS.get(field, field)}: already "
                         f"{current.get(field)} - nothing to change").classes(
                    'text-[10px] text-slate-500 leading-tight')
            for item in outcome["dropped"]:
                ui.label(f"not saved - {item['field']}: {item['found']} "
                         f"({item['reason']})").classes(
                    'text-[10px] text-slate-400 leading-tight')

    async def do_lookup() -> None:
        if view["running"] or not view["tab"] or view["tab"] == "results":
            return
        view["running"] = True
        run_button.disable()
        results.clear()
        progress.set_text('Searching…')
        source = view["tab"]
        try:
            if source in ("book", "deduce"):
                result = await character_research.research(
                    project.id, char.id, speculate=(source == "deduce"),
                    write=False,
                    log=lambda message: progress.set_text(
                        message.replace("[Research] ", "")
                               .replace("[Lookup] ", "")
                               .replace("[Book] ", "")
                               .replace("[Guess] ", "")))
                fields = result.get("fields") or {}
                invented = result.get("invented_values") or {}
                added = add_candidates(source, fields, invented)
                if result.get("used"):
                    note(f'{len(result["used"])} passages used')
                if result.get("invented_values"):
                    note(f'{len(result["invented_values"])} trait(s) cast, marked '
                         f'"guess"', positive=False)
                if not added:
                    progress.set_text("Nothing found.")
                    note(result.get("error") or "Nothing was written.")
                else:
                    progress.set_text(f"{added} value(s) added to the results tab.")
                    note(f"Added {added} value(s) - review them in the Results tab.",
                         positive=True)
                state.add_console_log(
                    f"[Lookup] {char.name} via {source}: {added} candidate(s)")
            else:
                fields: Dict[str, str] = {}
                error = ""
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
                        note("Sources: " + ", ".join(found["sources"][:3]))
                    if found.get("rounds"):
                        note(f"{found['rounds']} model rounds, "
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
                        note(f'Page: {got["title"]}')
                        extracted = await description_extract.extract_from_text(
                            char.name, got["block"], llm_url=llm_url,
                            model=model)
                        fields = extracted.get("fields") or {}
                        error = extracted.get("error", "")
                added = add_candidates(source, fields)
                if not fields:
                    progress.set_text('Nothing found.')
                    note(error or "The source stated nothing about their appearance.")
                else:
                    progress.set_text(f"{added} value(s) added to the results tab.")
                    note(f"Added {added} value(s) - review them in the Results tab.",
                         positive=True)
                state.add_console_log(
                    f"[Lookup] {char.name} via {source}: {added} candidate(s)"
                    + (f" ({error})" if error else ""))
        except Exception as ex:
            # A locked database or a vanished character must not leave the Search
            # button disabled for the rest of the session.
            progress.set_text('Could not search.')
            note(f"{type(ex).__name__}: {ex}")
        finally:
            view["running"] = False
            try:
                run_button.enable()
            except RuntimeError:
                pass
            render_cart()
            results_button.visible = bool(cart)
            if cart:
                results_button.set_text(f"Results ({len(cart_order())})")
                set_tab("results")

    with ui.dialog() as dialog, ui.card().classes(
            'w-[640px] max-w-[95vw] max-h-[88vh] p-0 rounded-xl flex flex-col '
            'flex-nowrap overflow-hidden'):
        with ui.row().classes('w-full items-center gap-3 px-5 pt-4 pb-2 flex-nowrap'):
            with ui.column().classes('gap-0 flex-1 min-w-0'):
                ui.label(f'Find a description: {char.name}').classes(
                    'text-base font-bold text-slate-800 leading-tight truncate')
                ui.label('Run one or more sources. Nothing is written until you tick '
                         'the values you want and apply them.').classes(
                    'text-[11px] text-slate-500 leading-tight whitespace-normal')
            ui.button(icon='close', on_click=close_dialog).props('flat dense round') \
                .classes('shrink-0 text-slate-400')

        source_buttons: Dict[str, Any] = {}
        with ui.row().classes('w-full gap-2 flex-nowrap px-5 pb-2'):
            for key, label, icon in SOURCES:
                reason = availability.get(key, "")
                button = ui.button(label, icon=icon, color=None,
                                   on_click=lambda _, k=key: set_tab(k)) \
                    .props('unelevated dense no-caps') \
                    .classes('text-xs font-bold px-2 py-1.5 rounded-lg border '
                             'bg-white text-slate-600 border-slate-200 '
                             'flex-1 min-w-0')
                if reason:
                    button.disable()
                    button.classes(add='opacity-50')
                    button.tooltip(reason)
                source_buttons[key] = button
            results_button = ui.button('Results', icon='shopping_cart', color=None,
                                       on_click=lambda: set_tab("results")) \
                .props('unelevated dense no-caps') \
                .classes('text-xs font-bold px-2 py-1.5 rounded-lg border '
                         'bg-white text-slate-600 border-slate-200 flex-1 min-w-0')
            results_button.visible = False

        with ui.column().classes('w-full px-5 gap-3 pb-4 overflow-y-auto flex-1 '
                                 'min-h-0'):
            # Source view: the query for the selected source, and what its last run
            # reported. The result log lives here; the decision lives in the cart.
            source_panel = ui.column().classes('w-full gap-3')
            with source_panel:
                query_panel = ui.column().classes('w-full gap-1')
                results = ui.column().classes(
                    'w-full gap-0 min-h-[60px] max-h-56 overflow-y-auto border '
                    'rounded-lg bg-white px-2 py-1.5')
                with results:
                    ui.label('Results appear here.').classes(
                        'text-[11px] text-slate-400')

            cart_panel = ui.column().classes('w-full gap-2')
            cart_panel.visible = False
            with cart_panel:
                ui.label('Base profile only - the look every image starts from. '
                         'Scene-level changes belong on the theater timeline.')\
                    .classes('text-[10px] text-slate-400 leading-tight')
                cart_status = ui.column().classes('w-full gap-0')
                cart_box = ui.column().classes('w-full gap-0')
                with cart_box:
                    ui.label('No results yet. Run a source above; whatever it finds '
                             'lands here to be reviewed.').classes(
                        'text-[11px] text-slate-400 py-3')

        with ui.row().classes('w-full items-center gap-2 flex-nowrap px-5 py-3 '
                              'border-t shrink-0') as search_footer:
            run_button = ui.button('Search', icon='search', on_click=do_lookup) \
                .props('unelevated dense no-caps') \
                .classes('bg-blue-600 hover:bg-blue-700 text-white font-bold '
                         'text-xs px-4 py-1.5 rounded-lg shrink-0')
            with ui.column().classes('gap-0 flex-1 min-w-0'):
                progress = ui.label('Ready.').classes(
                    'text-[11px] text-slate-600 leading-tight truncate')
                ui.label('Searches only; nothing is written until you apply.').classes(
                    'text-[10px] text-slate-400 leading-tight')

        with ui.row().classes('w-full items-center gap-2 flex-nowrap px-5 py-3 '
                              'border-t shrink-0') as apply_footer:
            apply_button = ui.button('Apply', icon='save', on_click=do_apply) \
                .props('unelevated dense no-caps') \
                .classes('bg-emerald-600 hover:bg-emerald-700 text-white font-bold '
                         'text-xs px-4 py-1.5 rounded-lg shrink-0')
            with ui.column().classes('gap-0 flex-1 min-w-0'):
                cart_summary = ui.label('Nothing ticked yet.').classes(
                    'text-[11px] text-slate-600 leading-tight truncate')
                ui.label('Writes to the Base profile only; a replaced value carries '
                         'forward until the book changes it.').classes(
                    'text-[10px] text-slate-400 leading-tight')
        apply_footer.visible = False

        render_query_help()
        style_sources()
        update_apply_summary()

    dialog.open()
