import wikipedia

def fetch_wikipedia_summary(name: str, sentences: int = 5) -> str:
    """Return a short Wikipedia summary for the given player name.
       If page not found, return an informative placeholder string."""
    try:
        page = wikipedia.page(name, auto_suggest=False)
        return page.summary[:sentences * 500]  # crude clipping
    except wikipedia.DisambiguationError as e:
        option = e.options[0] if e.options else name
        try:
            page = wikipedia.page(option, auto_suggest=False)
            return page.summary[:sentences * 500]
        except Exception:
            return f"No unambiguous wiki page found for '{name}'."
    except Exception as e:
        return f"Could not retrieve Wikipedia summary for '{name}': {e}"
