from nicegui import ui
from sqlmodel import Session
from database.connection import engine
from database.models import Book
from ui import state

def render_book_tabs(book_id: int):
    with Session(engine) as session:
        book = session.get(Book, book_id)
        
    if not book:
        ui.label("Book details not available.").classes('text-slate-400 text-sm')
        return

    ui.label(f'Book: {book.name}').classes('text-lg font-bold text-slate-800')
    
    with ui.tabs().classes('w-full border-b') as book_tabs:
        tab_book_dash = ui.tab('Dashboard', icon='grid_view')
        tab_book_editor = ui.tab('Proofreader & Editor Grid', icon='edit_note')
        
    # Bind tab selections to persist on UI refresh
    book_tabs.bind_value(state, 'active_book_tab')
        
    with ui.tab_panels(book_tabs, value=state.active_book_tab).classes('w-full bg-transparent p-0'):
        with ui.tab_panel(tab_book_dash):
            with ui.card().classes('w-full border p-5 shadow-sm bg-white'):
                ui.label('Book Progression Dashboard').classes('text-sm font-bold text-slate-800')
                ui.label('Placeholder scaffolding for metrics and chapter listing grids. (Phase 3)').classes('text-xs text-slate-500')
                
        with ui.tab_panel(tab_book_editor):
            with ui.card().classes('w-full border p-5 shadow-sm bg-white'):
                ui.label('Proofreader & Interactive Editor Grid').classes('text-sm font-bold text-slate-800')
                ui.label('Placeholder scaffolding for prompts.csv tabular visual editor. (Phase 5)').classes('text-xs text-slate-500')