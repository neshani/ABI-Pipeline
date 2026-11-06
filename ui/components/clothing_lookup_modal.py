"""What is this character wearing here? - the dialog behind "Find Clothing".

One search, one position, one value. The rule that makes it different from the
description dialog is that **attire is not a profile field**: it belongs to a
(chapter, shot) change point and holds forward from there, so this dialog writes a
timeline row, never the Base profile. Everything on screen is anchored to the
position being searched, including the warning you get when that position is not
where you are looking.

Results arrive as mentions, not as one answer, because the passages around a scene
usually dress several people: the character you opened this for, and whoever else
the text mentions. The mentions shown for your character are **checkboxes**, and
every ticked one is stored together as one outfit ("a coat of matching silk and a
cream-colored blouse") - the book often describes a look as a jacket here and
breeches three lines down. A mention that only says "he" is left unticked for the
user to decide: the model is asked to name who it means, and the quote is checked
against the character's own name before anything is ticked on the user's behalf.
"""
from __future__ import annotations

import asyncio
from typing import Any, Callable, Dict, List, Optional

from nicegui import ui
from sqlmodel import Session, select

from database.connection import engine, get_setting
from database.models import Book, Character, Project
from services import clothing_lookup
from ui import state


def _book_names(project_id: int) -> Dict[int, str]:
    return {book.id: book.name for book in clothing_lookup.project_books(project_id)}


# Three shots is this dialog's idea of "basically here": within it the banner stops
# warning that the searched position has drifted, and the change a stored outfit
# would land on is the look already on screen rather than one from another scene.
NEAR_SHOT_SPAN = 3


def _change_is_here(target: Dict) -> bool:
    """Does the change this position resolves to sit at, or just before, it?

    ``attire_target``'s distance is a signed delta of the searched position minus
    the change, so "just before" is zero chapters and a small non-negative shot
    gap. Anything else - an earlier chapter, an earlier volume, the Base profile,
    no change at all - is far enough away that a change of its own is worth
    offering.
    """
    if target.get("at_position"):
        return True
    distance = target.get("distance") or {}
    if distance.get("kind") != "same_book":
        return False
    chapters = int(distance.get("chapters") or 0)
    shots = int(distance.get("shots") or 0)
    return chapters == 0 and 0 <= shots <= NEAR_SHOT_SPAN


def _distance_text(book_id: int, chapter: int, shot: int, *, active_book_id: int,
                   active_chapter: int, active_shot: int,
                   names: Dict[int, str]) -> str:
    """How far the searched position is from the scene on screen, in words."""
    if book_id != active_book_id:
        return (f"Searching {names.get(book_id, 'another volume')}, chapter "
                f"{chapter}, shot {shot} - a different volume from the one on "
                f"screen.")
    chapters = chapter - active_chapter
    shots = shot - active_shot
    if not chapters and abs(shots) <= NEAR_SHOT_SPAN:
        return ""
    direction = "after" if (chapters > 0 or (not chapters and shots > 0)) else "before"
    parts: List[str] = []
    if chapters:
        parts.append(f"{abs(chapters)} chapter{'s' if abs(chapters) != 1 else ''}")
    if shots:
        parts.append(f"{abs(shots)} shot{'s' if abs(shots) != 1 else ''}")
    where = " and ".join(parts) or "a few shots"
    return (f"Searching chapter {chapter}, shot {shot} - {where} {direction} the "
            f"scene on screen. Anything you keep is written here, not there.")


def open_clothing_dialog(project: Project, char: Character, *,
                         book_id: int, chapter: int, shot: int,
                         active_book_id: Optional[int] = None,
                         active_chapter: int = 1, active_shot: int = 1,
                         refresh_callback: Optional[Callable[[], Any]] = None,
                         on_applied: Optional[Callable[[List[str], Optional[Dict]],
                                                       None]] = None,
                         initial_search: bool = False,
                         auto_search: bool = True) -> None:
    """Search the passages around one position for what they are wearing.

    ``on_applied(fields, position)`` reports the write to the host, which owns the
    stale-render offer: an attire change strands every image rendered with the old
    one, and this dialog knows nothing about images.

    ``auto_search`` starts the search without a click when the change in force is
    at this shot or within a few of it - the case where the outfit being looked for
    is the one already on screen. ``initial_search`` forces it either way.
    """
    llm_url = get_setting("llm_url", "http://127.0.0.1:11434")
    model = get_setting("llm_model", "local-model")
    active_book_id = active_book_id if active_book_id is not None else book_id
    names = _book_names(project.id)
    if book_id not in names:
        book_id = next(iter(names), 0) or book_id

    run_state: Dict[str, Any] = {"running": False, "changed": False,
                                 "mentions": [], "target": None}

    def position() -> Dict[str, int]:
        return {"book_id": int(book_select.value or book_id),
                "chapter": max(1, int(chapter_input.value or 1)),
                "shot": max(0, int(shot_input.value or 0))}

    def close_dialog() -> None:
        dialog.close()
        if run_state["changed"] and refresh_callback:
            refreshed = refresh_callback()
            if asyncio.iscoroutine(refreshed):
                asyncio.create_task(refreshed)

    def _distance_words(distance: Dict[str, int]) -> str:
        """'3 chapters and 12 shots back' from a target's distance."""
        kind = distance.get("kind")
        if kind == "earlier_book":
            books = distance.get("books") or 1
            return f"in an earlier volume ({books} book{'s' if books != 1 else ''} back)"
        if kind in ("same_book",):
            parts = []
            chapters = distance.get("chapters") or 0
            shots = distance.get("shots") or 0
            if chapters:
                parts.append(f"{chapters} chapter{'s' if abs(chapters) != 1 else ''}")
            if shots:
                parts.append(f"{shots} shot{'s' if abs(shots) != 1 else ''}")
            return " and ".join(parts) + " back" if parts else "at this shot"
        return ""

    def refresh_position_text() -> None:
        here = position()
        at_scene = (here["book_id"] == active_book_id
                    and here["chapter"] == active_chapter
                    and here["shot"] == active_shot)
        text = _distance_text(here["book_id"], here["chapter"], here["shot"],
                              active_book_id=active_book_id,
                              active_chapter=active_chapter,
                              active_shot=active_shot, names=names)
        if not text and at_scene:
            # Say it out loud rather than leaving the line blank: the position
            # otherwise only exists as two numbers in inputs, and "which scene am I
            # actually searching" was the question this dialog kept leaving open.
            text = (f"Searching the scene on screen: chapter {here['chapter']}, "
                    f"shot {here['shot']}.")
        banner_label.set_text(text)
        banner_label.visible = bool(text)
        # Moving the search back is only an offer while it has moved.
        use_scene_button.visible = not at_scene

        target = clothing_lookup.attire_target(
            char.id, here["book_id"], here["chapter"], here["shot"])
        run_state["target"] = target
        if not target.get("ok"):
            target_label.set_text(target.get("error") or "Could not read the "
                                                         "timeline for this character.")
        elif target.get("is_base"):
            target_label.set_text(
                "No change exists at or before this point, so the outfit is stored "
                "on the base profile and applies to every book until a change "
                "overrides it.")
        elif target.get("at_position"):
            target_label.set_text(
                f"A change already sits at this shot: storing updates its outfit, "
                f"and nothing before it.")
        else:
            where = (f"{target.get('book_name')} · chapter {target.get('chapter')}, "
                     f"shot {target.get('shot')}")
            note = (f"Last change is {_distance_words(target.get('distance') or {})} "
                    f"({where}). Storing updates that change, so every image from "
                    f"there on changes with it.")
            if _change_is_here(target):
                # Say why the button is missing: an absent button with no reason
                # reads as a broken dialog, not as "this is close enough".
                note += " No new change is offered this close to it."
            target_label.set_text(note)
        create_button.visible = bool(target.get("ok")) and not _change_is_here(target)
        banner.visible = bool(text) or bool(target_label.text) or create_button.visible

        worn = (target or {}).get("attire") or ""
        if worn:
            where = ("the base profile" if target.get("is_base")
                     else f"chapter {target.get('chapter')}, shot {target.get('shot')}")
            wearing_label.set_text(f"Currently wearing: {worn} (from {where})")
        else:
            wearing_label.set_text("Nothing recorded for attire at this position.")

    def create_change_here() -> None:
        """Insert a change point at the chosen shot - the explicit act.

        No dialog: the row is a copy of the state it inherits with the outfit you
        store next, so creating one changes no image on its own.
        """
        here = position()
        made = clothing_lookup.create_attire_change(
            project.id, char.id, here["book_id"], here["chapter"], here["shot"])
        if made.get("error"):
            ui.notify(made["error"], type="negative")
            return
        run_state["changed"] = True
        refresh_position_text()
        if made.get("created"):
            ui.notify(f"Change created at chapter {here['chapter']}, "
                      f"shot {here['shot']}.", type="positive", timeout=4000)
            state.add_console_log(
                f"[Clothing] {char.name}: change created at ch {here['chapter']} / "
                f"sh {here['shot']}")
        else:
            ui.notify("A change already exists at that shot.", type="info")

    def invalidate() -> None:
        """A result belongs to the position it was searched at.

        Moving the position after a run must not leave rows that would be written
        somewhere else, so the list goes with the position it came from.
        """
        if run_state.get("searched") is None:
            refresh_position_text()
            return
        run_state["searched"] = None
        run_state["mentions"] = []
        row_checks.clear()
        results.clear()
        position_label.set_text("")
        apply_button.disable()
        apply_button.set_text("Store attire")
        refresh_position_text()

    def use_active_scene() -> None:
        book_select.value = active_book_id
        chapter_input.value = active_chapter
        shot_input.value = active_shot
        invalidate()

    # The mention rows are checkboxes, and their ticks are never written from the
    # server after construction: the pre-tick is passed as the checkbox's initial
    # ``value`` while it is built, and from then on only the user (and the Store
    # read at apply time) touches it. Assigning a checkbox's ``value`` from the
    # server runs its own ``on_change`` in this NiceGUI, which is what made an
    # earlier "tick one, untick the rest" group here re-enter itself and leave a
    # ticked row with a dead Store button. A change handler that only refreshes
    # the button's text cannot cascade, so it is safe to attach.
    row_checks: Dict[int, Any] = {}

    def ticked_mentions() -> List[Dict]:
        """The mentions the user ticked, in row (passage) order.

        Read at store time, never cached: the checkbox ``value`` is the truth,
        however it came to be set.
        """
        selectable = [item for item in run_state["mentions"] if item["selectable"]]
        return [selectable[index] for index in sorted(row_checks)
                if row_checks[index].value and index < len(selectable)]

    def refresh_store_button() -> None:
        ticked = len(ticked_mentions())
        apply_button.set_text(f"Store attire ({ticked})" if ticked
                              else "Store attire")
        # Enabled whenever rows exist - "Store" with none ticked asks before it
        # acts, so a greyed-out button would only read as "this cannot be done",
        # which is how a lost selection used to look.
        if row_checks:
            apply_button.enable()
        else:
            apply_button.disable()

    def describe_storage(items: List[Dict]) -> str:
        """One outfit out of one or more mentions: joined with ' and '.

        The phrases are kept whole - grammar games with leading articles turn
        'a wrinkled coat and breeches of brocaded green silk' into something
        worse than the duplicate article they avoid.
        """
        if len(items) == 1:
            return items[0]["attire"]
        head = ", ".join(item["attire"] for item in items[:-1])
        return f"{head} and {items[-1]['attire']}"

    def render_mentions() -> None:
        """Draw the list, and say so out loud if drawing fails.

        A blank results area is indistinguishable from "the search found nothing",
        which is how a broken render presented itself once. Fail visibly instead of
        silently.
        """
        try:
            _render_mentions()
        except Exception as ex:                              # noqa: BLE001
            state.add_console_log(
                f"[Clothing] render failed: {type(ex).__name__}: {ex}")
            # The rows died with the failed render; a half-built checkbox list
            # left behind here could store a mention the user never saw.
            row_checks.clear()
            try:
                results.clear()
                with results:
                    ui.label(f"Could not show the results: {type(ex).__name__}: {ex}") \
                        .classes('text-[11px] text-red-600')
            except RuntimeError:
                # The page is gone, so there is nothing left to draw into. Trying
                # anyway raised a second exception that buried the first one.
                pass

    def _render_mentions() -> None:
        row_checks.clear()
        results.clear()
        mentions: List[Dict] = run_state["mentions"]
        target = [item for item in mentions if item["selectable"]]
        others = [item for item in mentions if not item["selectable"]]
        if not mentions:
            apply_button.disable()
            apply_button.set_text("Store attire")
            with results:
                ui.label(run_state.get("empty_note")
                         or "No clothing was mentioned in these passages.").classes(
                    'text-[11px] text-slate-500 py-1')
            return
        with results:
            if not target:
                ui.label("Nothing in these passages is attributed to them - no "
                         "mention names them and none is left unattributed. Other "
                         "people's clothing is listed below.").classes(
                    'text-[11px] text-slate-500 py-1')
            for index, item in enumerate(target):
                with ui.column().classes('w-full gap-0.5 border-b border-slate-100 '
                                         'pb-1 pt-1 rounded px-1'):
                    with ui.row().classes('w-full items-start gap-1.5 flex-nowrap'):
                        # The attire text is the checkbox's own label, so the whole
                        # phrase is the click target and the row needs no click
                        # handler of its own. `min-w-0` is what keeps this on
                        # screen: without it the label's min-content width pushes
                        # the checkbox past the card edge - the "checkboxes render
                        # offscreen" failure this dialog had before.
                        row_checks[index] = \
                            ui.checkbox(item["attire"], value=item["pre_tick"]) \
                            .props('dense') \
                            .classes('flex-1 min-w-0 text-xs font-semibold '
                                     'text-slate-800 leading-tight '
                                     'whitespace-normal') \
                            .on_value_change(lambda _: refresh_store_button())
                        if item["unnamed"]:
                            ui.label('unnamed').classes(
                                'text-[9px] font-bold uppercase tracking-wider '
                                'text-slate-500 bg-slate-100 border rounded px-1 '
                                'shrink-0 mt-0.5')
                        else:
                            ui.label(item["character"]).classes(
                                'text-[11px] font-bold text-slate-600 shrink-0 mt-0.5')
                        ui.label(f'ch {item["chapter"]} · sh {item["shot"]}').classes(
                            'text-[9px] text-slate-400 shrink-0 mt-1')
                        if item["when"] == "earlier":
                            ui.label('earlier').classes(
                                'text-[9px] font-bold uppercase tracking-wider '
                                'text-slate-500 bg-slate-100 border rounded px-1 '
                                'shrink-0 mt-0.5')
                        if item["target"] == "unsure":
                            ui.label('who?').classes(
                                'text-[9px] font-bold uppercase tracking-wider '
                                'text-amber-700 bg-amber-50 border border-amber-200 '
                                'rounded px-1 shrink-0 mt-0.5')
                    ui.label(f'"{item["quote"]}"').classes(
                        'text-[10px] text-slate-500 leading-snug pl-6')
        refresh_store_button()
        if others:
            # One heading per person. The flat list read as "Nynaeve appears three
            # times", which looks like a bug rather than three different items.
            grouped: Dict[str, List[Dict]] = {}
            for item in others:
                grouped.setdefault(item["character"], []).append(item)
            with results:
                ui.label('Also mentioned (someone else - open their profile to '
                         'store it)').classes(
                    'text-[10px] font-bold text-slate-400 uppercase tracking-wider '
                    'pt-2')
                for person, items in grouped.items():
                    ui.label(person).classes(
                        'text-[10px] font-bold text-slate-600 pt-1')
                    for item in items:
                        # Full quote, no height clip: the list scrolls, and a
                        # half-visible sentence reads as broken rather than dense.
                        ui.label(f'- {item["attire"]}').classes(
                            'text-[10px] text-slate-500 leading-snug pl-2')
                        ui.label(f'"{item["quote"]}"').classes(
                            'text-[9px] text-slate-400 leading-snug pl-4')

    async def do_search() -> None:
        if run_state["running"]:
            return
        here = position()
        run_state["running"] = True
        run_state["searched"] = dict(here)
        run_button.disable()
        apply_button.disable()
        row_checks.clear()
        results.clear()
        with results:
            ui.label('Searching the passages around this position…').classes(
                'text-[11px] text-slate-400')
        progress.set_text('Searching…')
        try:
            found = await clothing_lookup.find(
                project.id, here["book_id"], char.id, here["chapter"],
                here["shot"], llm_url=llm_url, model=model,
                log=lambda message: progress.set_text(
                    message.replace("[Clothing] ", "")))
        except Exception as ex:                              # noqa: BLE001
            found = {"ok": False, "error": f"{type(ex).__name__}: {ex}",
                     "mentions": [], "position": {}}
        run_state["running"] = False
        if results.is_deleted:
            # The page was reloaded or closed while the model was thinking, so
            # every element this dialog owns is gone. Say it in the console and
            # stop: touching the UI here raises "The client this element belongs
            # to has been deleted", which reads as the search having crashed.
            state.add_console_log(
                f"[Clothing] {char.name}: the page went away before the search "
                f"finished; the result was dropped.")
            return
        run_button.enable()
        run_state["mentions"] = found.get("mentions") or []
        run_state["empty_note"] = ("" if run_state["mentions"]
                                   else (found.get("error") or ""))
        run_state["position"] = found.get("position") or {}
        if not found.get("ok"):
            progress.set_text('Nothing found.')
            results.clear()
            with results:
                ui.label(found.get("error") or "Nothing was found.").classes(
                    'text-[11px] text-slate-500')
            apply_button.disable()
            return
        position_text = found.get("position") or {}
        progress.set_text(
            f"{len(run_state['mentions'])} mention(s) in {len(found.get('passages') or [])} "
            f"passage(s), {found.get('seconds')}s")
        searched = (position_text.get("chapter"), position_text.get("shot"))
        offered = sum(1 for item in run_state["mentions"] if item["selectable"])
        position_label.set_text(
            f"Searched chapter {searched[0]}, shot {searched[1]} - "
            f"{len(found.get('passages') or [])} chunk(s) around it, "
            f"{len(run_state['mentions'])} mention(s) found, "
            f"{offered} offered for this character.")
        render_mentions()
        position_label.set_text(
            position_label.text + f" {len(row_checks)} row(s) drawn.")
        state.add_console_log(
            f"[Clothing] {char.name} at ch {searched[0]} / sh {searched[1]}: "
            f"{len(run_state['mentions'])} mention(s)")

    async def do_apply() -> None:
        if run_state["running"]:
            return
        chosen = ticked_mentions()
        if not chosen:
            if run_state["mentions"]:
                ui.notify("Tick at least one of the outfits above.", type="info")
            return
        # Every ticked mention is stored as one outfit at the same position: the
        # passages often split a single look ("a coat" here, "breeches" there).
        attire = describe_storage(chosen)
        target_row = run_state.get("target") or {}
        run_state["running"] = True
        apply_button.disable()
        try:
            outcome = clothing_lookup.apply_attire(
                project.id, char.id, target_row.get("event_id"), attire)
        except Exception as ex:                              # noqa: BLE001
            run_state["running"] = False
            apply_button.enable()
            ui.notify(f"Could not save: {type(ex).__name__}: {ex}", type="negative")
            return
        run_state["running"] = False
        if not outcome["written"]:
            apply_button.enable()
            ui.notify(outcome.get("error") or "Nothing was written.",
                      type="warning")
            return
        run_state["changed"] = True
        where = target_row or {}
        if where.get("is_base"):
            verb = "stored on the base profile"
        elif where.get("at_position"):
            verb = (f"updated the change at chapter {where.get('chapter')}, "
                    f"shot {where.get('shot')}")
        else:
            verb = (f"updated the change at chapter {where.get('chapter')}, "
                    f"shot {where.get('shot')}, {_distance_words(where.get('distance') or {})}")
        text = f"{verb}: {attire}"
        if outcome.get("replaced"):
            text += f" (was: {outcome['replaced']})"
        if outcome.get("events"):
            text += f" - carried forward through {outcome['events']} later event(s)"
        if on_applied is not None:
            # The host owns the stale-render offer for this new value. The position
            # is where the change starts, which is not necessarily where we searched.
            on_applied(["attire"], {
                "book_id": where.get("book_id"),
                "chapter": where.get("chapter") or 0,
                "shot": where.get("shot") or 0,
            })
        state.add_console_log(f"[Clothing] {char.name}: {text}")
        # Notify before closing: closing refreshes the host, which tears down the
        # panel this dialog was built in.
        ui.notify(text, type="positive", timeout=6000)
        # One outfit per position, so there is nothing to keep the dialog open for.
        close_dialog()

    with ui.dialog() as dialog, ui.card().classes(
            'w-[660px] max-w-[95vw] max-h-[88vh] p-0 rounded-xl flex flex-col '
            'flex-nowrap overflow-hidden'):
        with ui.row().classes('w-full items-center gap-3 px-5 pt-4 pb-2 flex-nowrap'):
            with ui.column().classes('gap-0 flex-1 min-w-0'):
                ui.label(f'What are they wearing: {char.name}').classes(
                    'text-base font-bold text-slate-800 leading-tight truncate')
                ui.label('Searches the chunks around one position. Attire belongs to '
                         'that position, not to the profile, so a kept value is '
                         'written as a change there.').classes(
                    'text-[11px] text-slate-500 leading-tight whitespace-normal')
            ui.button(icon='close', on_click=close_dialog).props('flat dense round') \
                .classes('shrink-0 text-slate-400')

        with ui.column().classes('w-full px-5 gap-2 pb-1'):
            banner = ui.column().classes(
                'w-full gap-1 bg-amber-50 border border-amber-200 rounded-lg px-3 '
                'py-2')
            banner.visible = False
            with banner:
                banner_label = ui.label('').classes(
                    'text-[10px] text-amber-800 leading-tight')
                use_scene_button = ui.button(
                    'Search from the scene on screen instead', color=None,
                    on_click=use_active_scene) \
                    .props('flat dense no-caps') \
                    .classes('text-[10px] font-bold text-amber-800 px-0 py-0')
                # What storing will actually touch. A change point is data the user
                # asks for, so it is a button, never a side effect of storing.
                target_label = ui.label('').classes(
                    'text-[10px] text-amber-900 leading-tight')
                create_button = ui.button('Create a change here', icon='add',
                                          color=None, on_click=create_change_here) \
                    .props('unelevated dense no-caps') \
                    .classes('bg-amber-600 hover:bg-amber-700 text-white '
                             'text-[10px] font-bold px-2 py-0.5 rounded mt-1')

            with ui.row().classes('w-full items-end gap-2 flex-nowrap'):
                changed = lambda _=None: invalidate()          # noqa: E731
                book_select = ui.select(options=names, value=book_id,
                                        label='Book', on_change=changed) \
                    .classes('flex-1 bg-white min-w-0').props('dense outlined')
                chapter_input = ui.number(label='Chapter', value=int(chapter),
                                          min=1, step=1, on_change=changed) \
                    .classes('w-24').props('dense outlined')
                shot_input = ui.number(label='Shot', value=int(shot), min=0, step=1,
                                       on_change=changed) \
                    .classes('w-20').props('dense outlined')
            wearing_label = ui.label('').classes(
                'text-[10px] text-slate-500 leading-tight')
            position_label = ui.label('').classes(
                'text-[10px] text-slate-400 leading-tight')

        with ui.column().classes('w-full px-5 gap-2 pb-3 overflow-y-auto flex-1 '
                                 'min-h-0'):
            results = ui.column().classes('w-full gap-2 min-h-[60px]')

        with ui.row().classes('w-full items-center gap-2 flex-nowrap px-5 py-3 '
                              'border-t shrink-0'):
            run_button = ui.button('Search', icon='search', on_click=do_search) \
                .props('unelevated dense no-caps') \
                .classes('bg-slate-700 hover:bg-slate-800 text-white font-bold '
                         'text-xs px-4 py-1.5 rounded-lg shrink-0')
            apply_button = ui.button('Store attire', icon='save', on_click=do_apply) \
                .props('unelevated dense no-caps') \
                .classes('bg-emerald-600 hover:bg-emerald-700 text-white font-bold '
                         'text-xs px-4 py-1.5 rounded-lg shrink-0')
            apply_button.disable()
            with ui.column().classes('gap-0 flex-1 min-w-0'):
                progress = ui.label('Ready.').classes(
                    'text-[11px] text-slate-600 leading-tight truncate')
                ui.label('One outfit per position: every ticked mention is stored '
                         'together as one look. It holds until the next change.').classes(
                    'text-[10px] text-slate-400 leading-tight')

        refresh_position_text()
    dialog.open()
    # Search on its own when the change in force is right here. That is the case
    # where the user has already decided - they just added the change, or the outfit
    # they want is the one on screen - and only needs the passages read. Further back
    # the dialog waits for a click, because a search is a model call and its answer
    # may belong at a change that does not exist yet.
    if initial_search or (auto_search
                          and _change_is_here(run_state.get("target") or {})):
        asyncio.create_task(do_search())
