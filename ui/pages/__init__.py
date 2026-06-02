from .portal import render_portal_view
from .project import render_project_tabs
from .book_workspace import render_book_tabs
from .lora_contact_sheet import render_lora_contact_sheet

# Central global reference to the main layout for triggering dynamic workspace redraws
main_layout_ref = None

def register_main_layout(layout):
    """Registers the main page layout function so sub-panels can trigger full UI updates."""
    global main_layout_ref
    main_layout_ref = layout