These scripts are meant to be run with the local virtual environment:

Main CSV paths used by the scripts:

What it does:

- loads the raw CSV with pandas
- removes null and empty `title` / `content`
- removes short content
- removes OCR-like artifacts
- normalizes whitespace
- strips agency boilerplate from titles
- applies Turkish-aware title casing
- drops duplicate rows by `uuid` and content hash
- drops very long titles
- saves charts to an EDA folder

Current script setup is tailored to a specific csv format for my personal use. Editable encode/format/output paths.
