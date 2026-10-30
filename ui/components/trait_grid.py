# ui/components/trait_grid.py
"""The one editor for the character trait fields.

Both character forms - the Characters tab's details panel and the theater's
quick manager - ask the same twelve questions about the same event row. They
used to each carry their own copy of the field list, the label row, the widget
and the blur handler, and the copies had already drifted (the theater's did not
re-render field status after a save). Everything that is *layout* lives here, in
one table, so a spacing or ordering change happens once.

What stays with the host, because it is genuinely different:

* ``value_for`` - which event the value being shown comes from. The Characters
  tab substitutes the Base value for pinned traits and shows the scene value
  otherwise; the theater resolves the effective event at its position.
* ``save`` - what to do with an edited value. The tab carries the change
  forward, recompiles the description and refreshes its panel asynchronously;
  the theater writes the row and refreshes the workspace. Either may be a
  coroutine.

Layout is data: ``span`` is out of six columns, so a row is just "these fields,
these widths" and the table reads top-to-bottom in the order the form renders.

Three rules the widgets enforce, all of them because the value ends up in an
image prompt verbatim:

* Every field is one line of prose. A textarea accepts Enter and multi-line
  paste, so ``clean_trait_value`` collapses all whitespace to single spaces and
  the widget is reset to the cleaned text - otherwise the database and the box
  would disagree until the next panel refresh.
* An enumerated field is a dropdown, not a combo box. QSelect discards text that
  was typed but never committed with Enter or Tab (``state.onControlFocusout``
  resets the input), so a "type it yourself" select loses the value the moment
  the user clicks another field - and the field looks like it forgot them. These
  two fields have a short vocabulary, so they offer it as a list. Whatever value
  the row already holds is added to that list, so an imported ``unknown`` or
  ``male (assumed)`` still displays and can still be re-picked.
* Every other field stays free text. Eye colour, build, attire: no vocabulary.
"""
from __future__ import annotations

import inspect
import re
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, Optional, Sequence, Tuple

from nicegui import ui

from services.character_manager import RENDERED_TRAIT_FIELDS
from services.trait_vocab import AGE_STAGES

# The list sex offers. Sex has no canonical vocabulary in
# services/trait_vocab.py - the extractor writes what the text says - so this is
# only the common pair; a row that already holds something else keeps it as a
# selectable option (see ``_options_for``) rather than being blanked.
SEX_SUGGESTIONS: Tuple[str, ...] = ("male", "female")

# Six columns: three short fields per row, two medium, or one long. Fractional
# widths are deliberately not used - a field that is "half of the row" is what
# made the long values unreadable in the first place.
COLUMNS = 6

# How tall an autogrow field may get before it scrolls instead of pushing the
# rest of the form down. Quasar's measuring path reads a computed max-height and
# switches the overflow on by itself; the CSS below is what gives it one.
MAX_FIELD_HEIGHT_PX = 160

_STYLE_ADDED = False


@dataclass(frozen=True)
class TraitField:
    """One question in the form, and how much room it gets."""

    key: str
    label: str
    span: int = 2
    kind: str = "text"  # "text" | "select"
    options: Sequence[str] = ()


TRAIT_FIELDS: Tuple[TraitField, ...] = (
    TraitField("sex", "Sex", span=2, kind="select", options=SEX_SUGGESTIONS),
    TraitField("age", "Age", span=2, kind="select", options=AGE_STAGES),
    TraitField("species", "Species", span=2),
    TraitField("hair_color", "Hair Colour", span=3),
    TraitField("hair_style", "Hair Style", span=3),
    TraitField("eye_color", "Eye Colour", span=3),
    TraitField("build", "Build (Height / Frame)", span=3),
    TraitField("face", "Face (shape, complexion)", span=6),
    TraitField("distinguishing_marks",
               "Distinguishing Marks & Key Accessories", span=6),
    TraitField("attire", "Attire (Wearing Now)", span=6),
    TraitField("carried", "Carried (Held or Worn)", span=3),
    TraitField("notes", "Notes (stored, never rendered)", span=3),
)

# The older profiler's fields. Kept editable so an existing project loses
# nothing; nothing writes them for a new event.
LEGACY_FIELDS: Tuple[TraitField, ...] = (
    TraitField("demographics", "Demographics (Age, Race, Gender)", span=3),
    TraitField("hair_and_face", "Hair & Face Details", span=3),
    TraitField("physical_build", "Physical Build (Height/Weight/Posture)", span=6),
)


def clean_trait_value(value: object) -> str:
    """One trait, as one line.

    A trait compiles into a comma-joined sentence that gets injected into the
    render prompt, so a newline in it is never meaningful - it is either an
    accidental Enter in the textarea or a pasted paragraph. Collapsing whitespace
    here (rather than in the prompt compiler) keeps the stored value and the
    rendered prompt the same string.
    """
    return re.sub(r"\s+", " ", str(value or "")).strip()


def _ensure_style() -> None:
    """Cap the autogrow height once per process."""
    global _STYLE_ADDED
    if _STYLE_ADDED:
        return
    _STYLE_ADDED = True
    ui.add_css(f"""
        .trait-field--autogrow .q-field__native {{
            max-height: {MAX_FIELD_HEIGHT_PX}px;
            overflow-y: auto;
        }}
    """)


def _options_for(field: TraitField, value: str) -> list:
    """The dropdown's options, with the stored value kept in the list.

    A ``select`` whose options do not contain the model value renders blank, so
    an off-vocabulary value would look like a missing one - and the next person
    to touch the form would replace it with whatever the list does offer.
    """
    options = list(field.options)
    if value and value not in options:
        options.insert(0, value)
    return options


def _render_field(
    field: TraitField,
    value: str,
    *,
    save: Callable[[str, str], object],
    status_for: Optional[Callable[[str], Optional[Tuple[str, str]]]] = None,
    tooltip_for: Optional[Callable[[str], str]] = None,
    pin_for: Optional[Callable[[str], None]] = None,
    pinned_for: Optional[Callable[[str], bool]] = None,
    readonly_for: Optional[Callable[[str], Optional[str]]] = None,
    show_status: bool = True,
) -> ui.element:
    """One labelled field. Returns the input element."""
    value = value or ""
    rendered = field.key in RENDERED_TRAIT_FIELDS

    async def commit(new_value: str) -> None:
        result = save(field.key, new_value)
        if inspect.isawaitable(result):
            await result

    async def on_blur(e) -> None:
        raw = e.sender.value
        cleaned = clean_trait_value(raw)
        if isinstance(raw, str) and cleaned != raw:
            # Show what was stored, not what was typed: a pasted line break is
            # invisible in a textarea but it is not in the row.
            e.sender.set_value(cleaned)
        await commit(cleaned)

    async def on_clear(e) -> None:
        await commit("")

    tooltip = tooltip_for(field.key) if tooltip_for else None

    with ui.column().classes(f'w-full gap-0.5 md:col-span-{field.span}'):
        with ui.row().classes('w-full items-center gap-1'):
            label_el = ui.label(field.label).classes(
                'text-[10px] font-bold text-slate-500 uppercase tracking-wide')
            if tooltip:
                label_el.tooltip(tooltip)

            if show_status and rendered:
                status = status_for(field.key) if status_for else None
                if status and status[0]:
                    status_el = ui.label(status[0]).classes(
                        f'text-[9px] font-semibold {status[1]}')
                    if tooltip:
                        status_el.tooltip(tooltip)
            elif show_status:
                ui.label('note').classes('text-[9px] text-slate-400')

            if show_status and rendered and pin_for:
                is_pinned = bool(pinned_for and pinned_for(field.key))
                ui.button(icon='push_pin',
                          on_click=lambda _, k=field.key: pin_for(k))\
                    .props('flat dense round size=sm').classes(
                        'ml-auto ' + ('text-blue-600' if is_pinned
                                      else 'text-slate-300 hover:text-blue-500'))\
                    .tooltip('Pinned to the Base value; click to unpin'
                             if is_pinned else
                             'Pin this trait to the Base value')
            elif show_status:
                # A pin is taller than the label, so a row that has one is taller
                # than a row that does not - and the fields below them start at
                # different heights, which reads as a broken column. Occupying
                # the same space with a hidden pin keeps every row on one
                # baseline without a hard-coded height to drift from.
                ui.button(icon='push_pin')\
                    .props('flat dense round size=sm').classes('ml-auto invisible')

        if field.kind == "select":
            element = ui.select(
                options=_options_for(field, value),
                value=value or None,
                clearable=True,
            ).classes('w-full bg-white').props('outlined dense')
            element.on('blur', on_blur)
            # A cleared select fires `clear`, not a blur: without this the field
            # would look empty and keep its old value in the row.
            element.on('clear', on_clear)
        else:
            element = ui.textarea(value=value).classes(
                'w-full bg-white trait-field--autogrow')\
                .props('outlined dense autogrow').on('blur', on_blur)

        if readonly_for:
            readonly_tip = readonly_for(field.key)
            if readonly_tip:
                element.props('readonly').tooltip(readonly_tip)

    return element


def render_trait_grid(
    *,
    value_for: Callable[[str], str],
    save: Callable[[str, str], object],
    status_for: Optional[Callable[[str], Optional[Tuple[str, str]]]] = None,
    tooltip_for: Optional[Callable[[str], str]] = None,
    pin_for: Optional[Callable[[str], None]] = None,
    pinned_for: Optional[Callable[[str], bool]] = None,
    readonly_for: Optional[Callable[[str], Optional[str]]] = None,
    fields: Iterable[TraitField] = TRAIT_FIELDS,
) -> Dict[str, ui.element]:
    """Draw the trait form. Returns the widgets, keyed by trait name.

    ``value_for`` and ``save`` are the only required arguments: everything else
    is optional so a host can render the form without pinning or provenance.
    ``readonly_for`` returns the reason a field cannot be edited, or ``None`` -
    the reason doubles as the tooltip, because a read-only box with no
    explanation reads as a bug.
    """
    _ensure_style()
    widgets: Dict[str, ui.element] = {}
    with ui.grid().classes(
            f'w-full grid-cols-1 md:grid-cols-{COLUMNS} gap-2 mt-1'):
        for field in fields:
            widgets[field.key] = _render_field(
                field, value_for(field.key) or "",
                save=save,
                status_for=status_for,
                tooltip_for=tooltip_for,
                pin_for=pin_for,
                pinned_for=pinned_for,
                readonly_for=readonly_for,
            )
    return widgets


def render_legacy_fields(
    *,
    value_for: Callable[[str], str],
    save: Callable[[str, str], object],
) -> Dict[str, ui.element]:
    """The collapsed block of pre-trait fields both forms keep for old projects."""
    _ensure_style()
    widgets: Dict[str, ui.element] = {}
    with ui.expansion('Legacy description fields', icon='history', value=False)\
            .classes('w-full border rounded-xl bg-slate-50/40 text-xs'):
        ui.label(
            'The remaining fields the older profiler wrote. Kept editable so '
            'an existing project loses nothing; new descriptions use the '
            'fields above.'
        ).classes('text-[10px] text-slate-500 leading-tight')
        with ui.grid().classes(
                f'w-full grid-cols-1 md:grid-cols-{COLUMNS} gap-2'):
            for field in LEGACY_FIELDS:
                widgets[field.key] = _render_field(
                    field, value_for(field.key) or "",
                    save=save, show_status=False)
    return widgets
