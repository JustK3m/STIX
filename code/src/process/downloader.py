# process/stix_downloader.py

import os
import threading
from tkinter import *
from tkinter import messagebox, filedialog
from tkinter.ttk import Progressbar, Combobox


class STIXDownloader:
    """Fenêtre tkinter pour télécharger des données STIX science via stixdcpy."""

    # Types de produits disponibles via FitsQuery.query()
    PRODUCT_TYPES = {
        "Spectrogram L1A (xray-spec)":    ("xray-spec",   "L1A"),
    }

    def __init__(self, root):
        self.root = root
        self.win = Toplevel()
        self.win.title("STIX Data Downloader")
        self.win.geometry("580x360")
        self.win.resizable(False, False)
        self._build_ui()

    def _build_ui(self):
        pad = dict(padx=10, pady=6)

        Label(self.win, text="Download STIX Data",
              font="Helvetica 12 bold italic").pack(pady=(12, 0))
        Label(self.win, text="Source: datacenter.stix.i4ds.net",
              font="Helvetica 9", fg="gray").pack(pady=(0, 8))

        frame = Frame(self.win)
        frame.pack(fill=X, padx=25)

        # ── Start time ────────────────────────────────────────
        Label(frame, text="Start time (UTC):", anchor=W, width=20).grid(
            row=0, column=0, sticky=W, **pad)
        self.entry_start = Entry(frame, width=28)
        self.entry_start.insert(0, "2023-03-19T17:55:00")
        self.entry_start.grid(row=0, column=1, sticky=W, **pad)

        # ── End time ──────────────────────────────────────────
        Label(frame, text="End time (UTC):", anchor=W, width=20).grid(
            row=1, column=0, sticky=W, **pad)
        self.entry_end = Entry(frame, width=28)
        self.entry_end.insert(0, "2023-03-20T00:00:00")
        self.entry_end.grid(row=1, column=1, sticky=W, **pad)

        # ── Product type ──────────────────────────────────────
        Label(frame, text="Product type:", anchor=W, width=20).grid(
            row=2, column=0, sticky=W, **pad)
        self.dtype_var = StringVar(value=list(self.PRODUCT_TYPES.keys())[0])
        self.combo_dtype = Combobox(
            frame, textvariable=self.dtype_var,
            values=list(self.PRODUCT_TYPES.keys()),
            state="readonly", width=34)
        self.combo_dtype.grid(row=2, column=1, sticky=W, **pad)

        # ── Output directory ──────────────────────────────────
        Label(frame, text="Save to:", anchor=W, width=20).grid(
            row=3, column=0, sticky=W, **pad)
        dir_frame = Frame(frame)
        dir_frame.grid(row=3, column=1, sticky=W, **pad)
        self.entry_outdir = Entry(dir_frame, width=24)
        self.entry_outdir.insert(0, os.path.expanduser("~/Downloads"))
        self.entry_outdir.pack(side=LEFT)
        Button(dir_frame, text="Browse", command=self._browse).pack(
            side=LEFT, padx=(6, 0))

        # ── Progress + status ─────────────────────────────────
        self.progress = Progressbar(self.win, mode='indeterminate', length=480)
        self.progress.pack(pady=(12, 2))
        self.status_var = StringVar(value="Ready.")
        Label(self.win, textvariable=self.status_var,
              fg="gray", font="Helvetica 9").pack()

        # ── Buttons ───────────────────────────────────────────
        btn_frame = Frame(self.win)
        btn_frame.pack(pady=10)
        Button(btn_frame, text="Search & Download", bg="#1e40af", fg="white",
               padx=16, command=self._start_download).pack(side=LEFT, padx=8)
        Button(btn_frame, text="Close", padx=16,
               command=self.win.destroy).pack(side=LEFT, padx=8)

    def _browse(self):
        d = filedialog.askdirectory(title="Select output directory")
        if d:
            self.entry_outdir.delete(0, END)
            self.entry_outdir.insert(0, d)

    def _start_download(self):
        threading.Thread(target=self._download, daemon=True).start()

    def _download(self):
        # ── Vérification import ───────────────────────────────
        try:
            from stixdcpy.net import FitsQuery
        except ImportError:
            messagebox.showerror(
                "Missing package",
                "stixdcpy is not installed.\n\n"
                "Install it with:\n"
                "pip install stixdcpy"
            )
            return

        start  = self.entry_start.get().strip()
        end    = self.entry_end.get().strip()
        outdir = self.entry_outdir.get().strip()
        product_label = self.dtype_var.get()
        product_type, level = self.PRODUCT_TYPES[product_label]

        if not start or not end:
            messagebox.showwarning("Missing input",
                                   "Please enter start and end times.")
            return

        os.makedirs(outdir, exist_ok=True)
        FitsQuery.chdir(outdir)

        self.progress.start(10)
        self.status_var.set(f"Querying STIX Data Center ({product_type}, {level})...")
        self.win.update_idletasks()
        try:
            # ── Requête ───────────────────────────────────────
            results = FitsQuery.query(
                begin_utc=start,
                end_utc=end,
                product_type=product_type,
                level=level
            )
            results.result = [entry for entry in results.result if entry["level"] == level  ]
            n = len(results)
            if n == 0:
                self.status_var.set("No files found.")
                messagebox.showinfo(
                    "No data found",
                    f"No {product_label} data found between\n{start}\nand {end}.\n\n"
                    "Try a broader range or check:\n"
                    "https://datacenter.stix.i4ds.net"
                )
                return

            self.status_var.set(f"Found {n} file(s). Downloading...")
            self.win.update_idletasks()

            # ── Téléchargement ────────────────────────────────
            downloaded = FitsQuery.fetch(results.result)
            downloaded = [f for f in downloaded if f is not None]

            self.status_var.set(
                f"Done — {len(downloaded)} file(s) saved to {outdir}")
            messagebox.showinfo(
                "Download complete",
                f"{len(downloaded)} file(s) saved to:\n{outdir}\n\n"
                + "\n".join(os.path.basename(f) for f in downloaded))

        except Exception as e:
            self.status_var.set(f"Error: {e}")
            messagebox.showerror("Download error", str(e))

        finally:
            self.progress.stop()