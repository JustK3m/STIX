# io/loader.py - FITS reading and caching

`loader.py` is the single place in the codebase that reads FITS files with Astropy. All four windows go through it rather than calling `astropy.io.fits.open` directly (the one exception is `inputWindow.py::ShowHeader`, which keeps its own `HDUList` open to dump the raw primary header, and `downloader.py`, which reads/writes FITS files directly as part of merging downloaded products).

## Public functions

- **`get_data(fpath)`** - returns a dict with (subject to availability in the file) `time`, `timedel`, `counts`, `counts_err`, `triggers`, `obt_start`, `obt_end`, `e_low`, `e_high`.
- **`get_header(fpath)`** - returns a flat dict merging every card of `hdulist[0].header` (primary) and `hdulist[3].header` (fourth HDU, expected to be `ENERGIES`) into one dictionary. If a later card has the same keyword as an earlier one, the later value wins (`for key, value, comment in ...: result[key] = value` just keeps overwriting).
- **`get_srm_data(rpath)`** - returns a dict with `MATRIX`, `ENERG_LO`, `ENERG_HI` read from whichever HDU in the SRM file has those column names.
- **`activeFile()`** / **`activeSRMfile()`** - return the path most recently passed to `get_data` / `get_srm_data`, or `None` if neither has been called yet. This is how a window opened after another defaults its file picker to the same file (see the "Shared state" section of [../README.md](../README.md)).
- **`concat_data(hdu1, hdu2)`** - a one-line helper (`hdu1.data + hdu2.data`) that is currently unused anywhere in the codebase; `downloader.py`'s own `merge_stix_fits` implements file concatenation independently and does not call this.

## Caching

Two module-level dicts:

```python
_spec_cache = {"fpath": None, "data": None, "headers": None}
_srm_cache  = {"rpath": None, "data": None}
```

`get_data` / `get_header` both check `fpath != _spec_cache["fpath"]` before doing any I/O; if the requested path matches what is already cached, the cached dict is returned immediately without touching the filesystem. `_reload_spec(fpath)` is the only place that actually opens the file (`with fits.open(fpath) as hdulist:`), and it populates `data` and `headers` together in one pass so switching between `get_data` and `get_header` for the same file never triggers two separate reads. `get_srm_data` works the same way with `_srm_cache`.

Because the cache holds exactly one entry (not an LRU of several files), opening file A, then file B, then file A again re-reads file A from disk - there is no history beyond "the last file opened of each kind."

## Practical implications for callers

- Calling `get_data`/`get_header`/`get_srm_data` with the *same path* from multiple windows is cheap; the FITS file is parsed once.
- Calling with a *different* path anywhere evicts the previous cache entry - so `activeFile()` always reflects "the last spectrum file loaded by any window", which is exactly what the other windows rely on for their auto-fill behavior, but also means there is no way to have two spectrum files "active" at once across the app.
- `get_header`'s fixed-index read of `hdulist[3]` is the most fragile part of this module; a FITS file with extra or reordered HDUs before `ENERGIES` will produce wrong or missing header values without any error being raised.
