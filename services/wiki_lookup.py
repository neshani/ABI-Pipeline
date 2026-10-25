"""One wiki, fetched politely, turned into the same ``## Name`` block as a book.

Why an API instead of the page: Fandom serves its HTML behind bot detection (a
plain GET returns 403) but its MediaWiki API answers happily, and the API is the
sanctioned path for automated use. ``api.php`` exists on Fandom, on A Wiki of Ice
and Fire, on Miraheze and on any stock MediaWiki install, so one implementation
covers every site a user is likely to name.

Politeness is a feature, not a default to be relaxed later: one request per
second per host, page text cached on disk, and a hard stop on 429/403 rather than
a retry loop. A user pointing this at somebody else's wiki should not be able to
hurt them by clicking Run twice.
"""
from __future__ import annotations

import json
import re
import time
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from urllib.parse import urlsplit

import httpx

UA = {"User-Agent": "ABI-Pipeline/1.0 (personal illustration tool)"}
_CACHE = Path("workspace_temp/wiki_cache")
_MIN_INTERVAL = 1.0
_last_call: Dict[str, float] = {}

# Prefixes that mean "a page about the same character", not "a different one".
_SUBPAGE = re.compile(r"/(Chronology|Parallels|Gallery|Quotes|Theories|TV series)$", re.I)
_APPEARANCE_HEADING = re.compile(r"appear|physical|description|looks|features", re.I)
_INFOBOX_FIELDS = {
    "hair": "Hair", "haircolor": "Hair", "eyes": "Eyes", "eyecolor": "Eyes",
    "height": "Height", "skin": "Skin", "species": "Species", "race": "Race",
    "weight": "Weight", "build": "Build",
}


class WikiError(RuntimeError):
    """A site that refused us. The caller should end the step, not retry."""


def normalise_base(url: str) -> str:
    """``https://host`` from anything the user pasted, including a page link."""
    url = (url or "").strip()
    if not url:
        return ""
    if "//" not in url:
        url = "https://" + url
    parts = urlsplit(url)
    if not parts.scheme or not parts.netloc:
        return ""
    return f"{parts.scheme}://{parts.netloc}"


def _api(base: str) -> str:
    return f"{base.rstrip('/')}/api.php"


def _throttle(host: str) -> None:
    last = _last_call.get(host, 0.0)
    wait = _MIN_INTERVAL - (time.time() - last)
    if wait > 0:
        time.sleep(wait)
    _last_call[host] = time.time()


def _get(base: str, params: Dict) -> Dict:
    host = urlsplit(base).netloc
    _throttle(host)
    try:
        with httpx.Client(timeout=30, headers=UA, follow_redirects=True) as client:
            response = client.get(_api(base), params=dict(params, format="json"))
    except httpx.HTTPError as ex:
        raise WikiError(f"could not reach {base} ({type(ex).__name__})") from ex
    if response.status_code in (403, 429):
        retry = response.headers.get("retry-after", "")
        raise WikiError(f"{base} refused the request (HTTP {response.status_code}"
                        + (f", retry after {retry}s" if retry else "") + ")")
    if response.status_code >= 400:
        raise WikiError(f"{base} returned HTTP {response.status_code}")
    try:
        return response.json()
    except json.JSONDecodeError as ex:
        raise WikiError(f"{base} returned non-JSON") from ex


def _cache_path(base: str, title: str) -> Path:
    digest = hashlib.sha1(f"{base}|{title}".encode("utf-8")).hexdigest()[:16]
    return _CACHE / f"{digest}.json"


def _cache_read(base: str, title: str, max_age_days: int = 30) -> Optional[Dict]:
    path = _cache_path(base, title)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if time.time() - float(data.get("fetched", 0)) > max_age_days * 86400:
        return None
    return data


def _cache_write(base: str, title: str, payload: Dict) -> None:
    try:
        _CACHE.mkdir(parents=True, exist_ok=True)
        data = dict(payload, fetched=time.time())
        _cache_path(base, title).write_text(json.dumps(data), encoding="utf-8")
    except Exception:
        pass  # a cache miss next time is not worth failing a lookup for


# --- wikitext ---------------------------------------------------------------

def clean_wikitext(text: str) -> str:
    """Prose from wikitext: links unwrapped, templates and refs dropped."""
    text = re.sub(r"(?is)<ref[^>]*/>|<ref[^>]*>.*?</ref>", "", text)
    text = re.sub(r"(?is)<(gallery|timeline|tabber|div|span|br|center)[^>]*>", " ", text)
    text = re.sub(r"(?is)<[^>]+>", " ", text)
    text = re.sub(r"\{\{[^{}]*\}\}", " ", text)
    text = re.sub(r"\[\[(?:[^|\]]*\|)?([^\]]+)\]\]", r"\1", text)
    text = re.sub(r"'{2,}", "", text)
    text = re.sub(r"^=+\s*(.*?)\s*=+\s*$", r"\n\1\n", text, flags=re.M)
    text = re.sub(r"^\s*[*#:;].*$", "", text, flags=re.M)
    text = text.replace("&nbsp;", " ")
    return re.sub(r"\s+", " ", text).strip()


def infobox_line(wikitext: str) -> str:
    """Appearance values from the infobox, as a sentence the extractor can read.

    Wiki infoboxes are inconsistent - one franchise keeps height in metres with a
    per-era hair colour, another writes ``height=Short`` - so the values are
    quoted as-is with their labels rather than parsed here. The extractor upstream
    is good at turning "Height: Short. Hair: Brown." into fields.
    """
    parts = []
    for key, label in _INFOBOX_FIELDS.items():
        match = re.search(rf"^\s*\|\s*{key}\s*=\s*(.+)$", wikitext, re.I | re.M)
        if not match:
            continue
        value = clean_wikitext(match.group(1))
        value = re.sub(r"^[*,]\s*", "", value).strip(" ,;")
        if value and len(value) < 160:
            parts.append(f"{label}: {value}.")
    return " ".join(parts)


def appearance_section(wikitext: str) -> str:
    """The Appearance section's prose, or the lead if there is no such section."""
    headings = list(re.finditer(r"^=+\s*(.+?)\s*=+\s*$", wikitext, re.M))
    for index, match in enumerate(headings):
        if not _APPEARANCE_HEADING.search(match.group(1)):
            continue
        end = headings[index + 1].start() if index + 1 < len(headings) else len(wikitext)
        return clean_wikitext(wikitext[match.end():end])
    return ""


# --- pages ------------------------------------------------------------------

def search_pages(base: str, query: str, limit: int = 5) -> List[Dict]:
    data = _get(base, {"action": "query", "list": "search",
                       "srsearch": query, "srlimit": max(1, min(limit, 20))})
    return [{"title": hit.get("title", ""),
             "snippet": clean_wikitext(hit.get("snippet", ""))}
            for hit in (data.get("query", {}).get("search") or [])]


def _tokens(text: str) -> set:
    return set(re.findall(r"[a-z']+", (text or "").lower()))


def _title_is_about(title: str, candidates: List[str]) -> bool:
    """Does the page title name one of these characters?

    The surname decides. A search for "Bornhald" must not return "Jaret Byar" -
    both pages mention Bornhald, only one is about him - and a search for
    "Master Gill" must accept "Basel Gill" while rejecting "Gilda".
    """
    title_tokens = _tokens(title)
    for candidate in candidates:
        words = [w for w in re.findall(r"[a-z']+", (candidate or "").lower()) if len(w) > 2]
        if words and words[-1] in title_tokens:
            return True
    return False


def _title_score(title: str, candidates: List[str], *,
                 require_name_word: bool = True) -> int:
    """How well a page title names a candidate: 0 exact, 9 not this character.

    The score has to prefer the *fuller* match, not the shorter title. A project
    character called "Bornhald" with the alias "Geofram Bornhald" must resolve to
    Geofram, not to his son Dain, and both pages contain the surname.

    ``require_name_word`` is the guard against a page about an office, a place or
    a rank: the title must use a word from the character's own name, not merely
    from an alias. The alias table is full of titles ("the Amyrlin Seat", "Lord of
    Fal Dara"), and a title alias scores 0 against the page about that title - so
    "Siuan Sanche" was being answered with the article on the Amyrlin Seat, and
    the extractor reported "sex: female" from it. A user-supplied query relaxes
    this, because typing a title is an explicit request for that page.
    """
    title_tokens = _tokens(title)
    title_low = title.strip().lower()
    if require_name_word and candidates:
        own = [w for w in re.findall(r"[a-z']+", (candidates[0] or "").lower())
               if len(w) > 2]
        if own and not any(word in title_tokens for word in own):
            return 9
    best = 9
    for candidate in candidates:
        words = [w for w in re.findall(r"[a-z']+", (candidate or "").lower())
                 if len(w) > 2]
        if not words:
            continue
        if title_low == candidate.strip().lower():
            best = min(best, 0)
        elif all(word in title_tokens for word in words):
            best = min(best, 1)
        elif words[-1] in title_tokens:
            best = min(best, 2)
    return best


def name_bearing_probe(character_name: str, names: List[str]) -> str:
    """The fullest name that still contains the character's own name.

    "Longest candidate wins" picks a title when the alias table holds one: for a
    character called Siuan with the alias "the Amyrlin Seat", three words beat two,
    so the wiki gets searched for the office rather than the person, returns the
    article about the office, and the extractor dutifully reports "sex: female"
    from it. An alias is only a better *search* term when it extends the name.
    """
    tokens = set(re.findall(r"[a-z']+", (character_name or "").lower()))
    bearing = [n for n in names
               if tokens & set(re.findall(r"[a-z']+", (n or "").lower()))]
    return max(bearing or [character_name],
               key=lambda n: (len(n.split()), len(n)))


def resolve_title(base: str, names: List[str], term: str = "",
                  query: str = "") -> str:
    """The page about this character, or "" when no page is about them.

    ``list=search`` rather than an exact title, because characters are filed under
    later names ("Perrin t'Bashere Aybara") and page names carry disambiguators
    ("James Holden (Books)"). Every hit has to pass a title check, because a
    search hit only proves the page *mentions* the name - returning the first hit
    is how one character's description lands on another.
    """
    names = [n for n in names if n and n.strip()]
    if not names:
        return ""
    # Search with the most specific name available; the project's own name is
    # often a nickname ("Master Gill", "Bornhald"). A user-supplied query wins,
    # and typing one also relaxes the name-word check: it is an explicit request.
    explicit = bool((query or "").strip())
    probe = name_bearing_probe(names[0], names)
    search = (query.strip() if explicit
              else (f'"{probe}" {term}'.strip() if term else f'"{probe}"'))
    hits = search_pages(base, search, limit=10)
    scored = []
    for hit in hits:
        score = _title_score(hit["title"], names, require_name_word=not explicit)
        if score > 2:
            continue
        title = hit["title"]
        low = title.lower()
        scored.append((score,
                       1 if _SUBPAGE.search(title) else 0,
                       0 if ("(books)" in low or "(novel" in low) else 1,
                       1 if "(tv" in low else 0,
                       len(title), title))
    if not scored:
        return ""
    return sorted(scored)[0][-1]


def page_block(base: str, title: str) -> Dict:
    """``{title, block, has_appearance}`` for one page, cached."""
    cached = _cache_read(base, title)
    if cached:
        return cached
    data = _get(base, {"action": "parse", "page": title, "prop": "wikitext"})
    wikitext = (data.get("parse", {}).get("wikitext") or {}).get("*", "")
    if not wikitext:
        raise WikiError(f"{title} has no readable content on {base}")
    info = infobox_line(wikitext)
    body = appearance_section(wikitext)
    if not body:
        # No dedicated section: keep the lead, which often describes the character
        # before the plot starts.
        lead = clean_wikitext(re.split(r"^=+\s*.+?\s*=+\s*$", wikitext, maxsplit=1,
                                       flags=re.M)[0])
        body = lead[:2000]
    block = "\n".join(part for part in (info, body) if part).strip()
    payload = {"title": title, "block": block,
               "has_appearance": bool(appearance_section(wikitext))}
    _cache_write(base, title, payload)
    return payload


def lookup(name: str, *, base: str, term: str = "",
           aliases: Optional[List[str]] = None, query: str = "") -> Dict:
    """``{ok, title, block, error, tried}`` for one character.

    Tries the project name, then the aliases: the page is often under a fuller
    name ("Basel Gill" for "Master Gill") and the alias table is the only thing
    that knows that.

    ``query`` overrides the search string. The default is built from the most
    specific name, but a user who can see the term is the one who knows whether
    the wiki files someone under a title, a nickname or a full name - and the
    title check below still decides whether the hit is really about this person.
    """
    base = normalise_base(base)
    if not base:
        return {"ok": False, "title": "", "block": "", "error": "no wiki URL set",
                "tried": ""}
    names = list(dict.fromkeys([name, *(aliases or [])]))
    tried = query.strip() or ", ".join(names[:4])
    try:
        title = resolve_title(base, names, term, query=query)
        if not title:
            return {"ok": False, "title": "", "block": "",
                    "error": f"no page is about this character ({tried})",
                    "tried": tried}
        page = page_block(base, title)
    except WikiError as ex:
        return {"ok": False, "title": "", "block": "", "error": str(ex),
                "tried": tried}
    if not page.get("block"):
        return {"ok": False, "title": title, "block": "",
                "error": "the page has no appearance text", "tried": tried}
    return {"ok": True, "title": title, "block": page["block"], "error": "",
            "tried": tried}


def test_site(base: str, term: str, samples: Optional[List[str]] = None) -> Dict:
    """Is this wiki the right one for the series, and does it describe people?

    Runs from the settings panel before anything is imported: the search term
    shows whether the wiki covers the series at all, and two sample pages show
    whether it has appearance text worth extracting.
    """
    base = normalise_base(base)
    if not base:
        return {"ok": False, "error": "enter a wiki URL", "hits": [], "pages": []}
    try:
        hits = search_pages(base, term or "", limit=5)
    except WikiError as ex:
        return {"ok": False, "error": str(ex), "hits": [], "pages": []}
    pages = []
    for name in (samples or [])[:2]:
        try:
            title = resolve_title(base, [name], term)
            if not title:
                pages.append({"name": name, "title": "", "chars": 0,
                              "appearance": False})
                continue
            page = page_block(base, title)
            pages.append({"name": name, "title": title,
                          "chars": len(page.get("block", "")),
                          "appearance": bool(page.get("has_appearance"))})
        except WikiError as ex:
            pages.append({"name": name, "title": "", "chars": 0,
                          "appearance": False, "error": str(ex)})
            break
    return {"ok": bool(hits), "error": "", "hits": hits, "pages": pages}
