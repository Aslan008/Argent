"""Long-term memory as plain Markdown notes: an index in the system prompt, nodes read on demand.

    <project>/.argent/memory/MEMORY.md       the index (one line per note, generated)
    <project>/.argent/memory/<name>.md       a note: a small header, then the text
    ~/.argent/memory/                        the same, for everything that holds in every project

`store` is one folder of notes, `hub` is the two levels seen as one memory, `dream` is the consolidation that
runs while the PC is idle. The working memory of one task (`.argent/memory.json`, memory_manager.py) is another
thing and stays as it is.
"""
