# process/stix_downloader.py

import os
import threading
from fileinput import close
from tkinter import *
from tkinter import messagebox, filedialog
from tkinter.ttk import Progressbar, Combobox
from stixdcpy.net import FitsQuery
from stixdcpy.science import spec_fits_concatenate
from astropy.io import fits
import numpy as np

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
        self.entry_start.insert(0, "2023-03-19T17:00:00.000")
        self.entry_start.grid(row=0, column=1, sticky=W, **pad)

        # ── End time ──────────────────────────────────────────
        Label(frame, text="End time (UTC):", anchor=W, width=20).grid(
            row=1, column=0, sticky=W, **pad)
        self.entry_end = Entry(frame, width=28)
        self.entry_end.insert(0, "2023-03-20T00:00:00.000")
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

    # def _download(self):
    #     # ── Vérification import ───────────────────────────────
    #     try:
    #         from stixdcpy.net import FitsQuery
    #     except ImportError:
    #         messagebox.showerror(
    #             "Missing package",
    #             "stixdcpy is not installed.\n\n"
    #             "Install it with:\n"
    #             "pip install stixdcpy"
    #         )
    #         return
    #
    #     start  = self.entry_start.get().strip()
    #     end    = self.entry_end.get().strip()
    #     outdir = self.entry_outdir.get().strip()
    #     product_label = self.dtype_var.get()
    #     product_type, level = self.PRODUCT_TYPES[product_label]
    #
    #     if not start or not end:
    #         messagebox.showwarning("Missing input",
    #                                "Please enter start and end times.")
    #         return
    #
    #     os.makedirs(outdir, exist_ok=True)
    #     FitsQuery.chdir(outdir)
    #
    #     self.progress.start(10)
    #     self.status_var.set(f"Querying STIX Data Center ({product_type}, {level})...")
    #     self.win.update_idletasks()
    #     try:
    #         # ── Requête ───────────────────────────────────────
    #         results = FitsQuery.query(
    #             begin_utc=start,
    #             end_utc=end,
    #             product_type=product_type,
    #             level=level
    #         )
    #         results.result = [entry for entry in results.result if entry["level"] == level  ]
    #         n = len(results)
    #         if n == 0:
    #             self.status_var.set("No files found.")
    #             messagebox.showinfo(
    #                 "No data found",
    #                 f"No {product_label} data found between\n{start}\nand {end}.\n\n"
    #                 "Try a broader range or check:\n"
    #                 "https://datacenter.stix.i4ds.net"
    #             )
    #             return
    #
    #         self.status_var.set(f"Found {n} file(s). Downloading...")
    #         self.win.update_idletasks()
    #
    #         # ── Téléchargement ────────────────────────────────
    #         downloaded = FitsQuery.fetch(results.result)
    #         downloaded = [f for f in downloaded if f is not None]
    #
    #         self.status_var.set(
    #             f"Done — {len(downloaded)} file(s) saved to {outdir}")
    #         messagebox.showinfo(
    #             "Download complete",
    #             f"{len(downloaded)} file(s) saved to:\n{outdir}\n\n"
    #             + "\n".join(os.path.basename(f) for f in downloaded))
    #
    #     except Exception as e:
    #         self.status_var.set(f"Error: {e}")
    #         messagebox.showerror("Download error", str(e))
    #
    #     finally:
    #         self.progress.stop()

    def _download(self):

        start = self.entry_start.get().strip()
        end = self.entry_end.get().strip()
        outdir = self.entry_outdir.get().strip()
        product_label = self.dtype_var.get()
        product_type, level = self.PRODUCT_TYPES[product_label]

        if not start or not end:
            messagebox.showwarning("Missing input", "Please enter start and end times.")
            return

        os.makedirs(outdir, exist_ok=True)
        FitsQuery.chdir(outdir)

        self.progress.start(10)
        self.status_var.set(f"Querying STIX Data Center ({product_type}, {level})...")
        self.win.update_idletasks()
        output = ""
        try:
            results = FitsQuery.query(
                begin_utc=start,
                end_utc=end,
                product_type=product_type,
                level=level
            )
            results.result = [entry for entry in results.result if entry["level"] == level]
            n = len(results)
            if n == 0:
                self.status_var.set("No files found.")
                messagebox.showinfo(
                    "No data found",
                    f"No {product_label} data found between\n{start}\nand {end}.\n\n"
                    "Try a broader range or check:\nhttps://datacenter.stix.i4ds.net"
                )
                return

            self.status_var.set(f"Found {n} file(s). Downloading...")
            self.win.update_idletasks()

            downloaded = FitsQuery.fetch(results.result)

            # ── Concaténation si plusieurs fichiers ───────────────────────────
            if len(downloaded) == 1:
                self.status_var.set(f"Done — 1 file saved to {outdir}")

            else:
                self.status_var.set(
                    f"Concatenating {len(downloaded)} overlapping files...")
                self.win.update_idletasks()

            output = os.path.join(
                outdir,
                f"stix_concat_{start.replace(':', '').replace('-', '')}_{end.replace(':', '').replace('-', '')}.fits"
            )
            self.merge_stix_fits(start, end, downloaded,
                                 output)

        except Exception as e:
            print(e)
            self.status_var.set(f"Error: {e}")
            messagebox.showerror("Download error", str(e))

        finally:
            self.progress.stop()

        self.status_var.set(
            f"Done — concatenated into {os.path.basename(output)}")

        messagebox.showinfo(
            "Download complete",
            f"File saved to:\n{output}"
        )

    def merge_stix_fits(self, start, end, file_list, outfile):
        """
        Fusionne une liste de fichiers FITS STIX L1A (xray-spec) en un seul fichier.

        Stratégie :
            - Tri chronologique par temps absolu (MJDREF + time).
            - Conversion de tous les temps en temps absolu (secondes) pour
              comparaison inter-fichiers (chaque fichier a son propre MJDREF).
            - Overlap : moyenne pondérée par timedel des counts et counts_comp_err
              pour les pas de temps communs entre deux fichiers.
            - Gap : les pas de temps manquants sont comblés par NaN.
            - Le fichier de sortie utilise le MJDREF du premier fichier comme
              référence commune, et tous les temps sont recalculés en conséquence.

        Parameters
        ----------
        file_list : list of str
            Liste des chemins FITS à fusionner.
        outfile : str
            Chemin du fichier FITS de sortie.
        """
        import numpy as np
        from astropy.io import fits

        # ── 1. Lecture de tous les fichiers ───────────────────────────────────
        all_data = []  # liste de dicts par fichier
        ref_e_low = None
        ref_e_high = None
        primary_header = None
        control_data = None
        control_header = None
        energies_header = None
        ref_channel = None

        for fpath in file_list:
            with fits.open(fpath, memmap=False) as hdul:
                mjdref = float(hdul[0].header['MJDREF'])
                e_low = hdul['ENERGIES'].data['e_low'].copy()
                e_high = hdul['ENERGIES'].data['e_high'].copy()

                if ref_e_low is None:
                    ref_e_low = e_low
                    ref_e_high = e_high
                    ref_mjdref = mjdref
                    primary_header = hdul[0].header.copy()
                    primary_header['DATE_BEG'] = start
                    primary_header['DATE_END'] = end
                    control_data = hdul['CONTROL'].data.copy()
                    control_header = hdul['CONTROL'].header.copy()
                    energies_header = hdul['ENERGIES'].header.copy()
                    ref_channel = hdul['ENERGIES'].data['channel'].copy()
                    data_header = hdul['DATA'].header.copy()
                else:
                    if not (np.allclose(e_low, ref_e_low, equal_nan=True) and
                            np.allclose(e_high, ref_e_high, equal_nan=True)):
                        raise ValueError(f"Grille d'énergie incompatible : {fpath}")

                d = hdul['DATA'].data
                # Temps absolu en secondes pour comparaison inter-fichiers
                t_abs = mjdref * 86400.0 + d['time'].copy()

                all_data.append({
                    't_abs': t_abs,
                    'timedel': d['timedel'].copy(),
                    'counts': d['counts'].copy(),
                    'counts_comp_err': d['counts_comp_err'].copy(),
                    'mjdref': mjdref,
                })

        # ── 3. Construction de la timeline fusionnée ───────────────────────────
        # On empile tous les (t_abs, timedel, counts, counts_comp_err) de tous
        # les fichiers, puis on gère overlaps et gaps sur cette base commune.

        all_t = np.concatenate([d['t_abs'] for d in all_data])
        all_td = np.concatenate([d['timedel'] for d in all_data])
        all_counts = np.concatenate([d['counts'] for d in all_data], axis=0)
        all_err = np.concatenate([d['counts_comp_err'] for d in all_data], axis=0)

        # Tri global chronologique
        sort_idx = np.argsort(all_t, kind='stable')
        all_t = all_t[sort_idx]
        all_td = all_td[sort_idx]
        all_counts = all_counts[sort_idx]
        all_err = all_err[sort_idx]

        n_energy = all_counts.shape[1]

        # ── 4. Fusion des overlaps par moyenne pondérée ────────────────────────
        # Deux pas de temps i et j sont en overlap si leurs intervalles
        # [t-td/2, t+td/2] se chevauchent. On groupe les pas de temps proches
        # (tolérance = moitié du plus petit timedel) et on les moyenne.

        tol = 0.5 * np.min(all_td)  # tolérance de groupement (~0.25 s)

        merged_t = []
        merged_td = []
        merged_counts = []
        merged_err = []

        i = 0
        N = len(all_t)
        while i < N:
            # Groupe : tous les pas de temps dans [t_i - tol, t_i + tol]
            group = [i]
            j = i + 1
            while j < N and abs(all_t[j] - all_t[i]) <= tol:
                group.append(j)
                j += 1

            if len(group) == 1:
                # Pas d'overlap : on prend directement
                merged_t.append(all_t[i])
                merged_td.append(all_td[i])
                merged_counts.append(all_counts[i])
                merged_err.append(all_err[i])
            else:
                # Overlap : moyenne pondérée par timedel
                g_idx = np.array(group)
                weights = all_td[g_idx]  # shape (n_group,)
                w_sum = weights.sum()

                t_avg = float(np.sum(weights * all_t[g_idx]) / w_sum)
                td_avg = float(np.sum(weights * all_td[g_idx]) / w_sum)

                # counts : moyenne pondérée — shape (n_group, n_energy)
                c_avg = (np.sum(all_counts[g_idx] * weights[:, np.newaxis], axis=0)
                         / w_sum)

                # counts_comp_err : propagation quadratique pondérée
                e_avg = (np.sqrt(np.sum((weights[:, np.newaxis] * all_err[g_idx]) ** 2,
                                        axis=0))
                         / w_sum)

                merged_t.append(t_avg)
                merged_td.append(td_avg)
                merged_counts.append(c_avg)
                merged_err.append(e_avg)

            i = j

        merged_t = np.array(merged_t)
        merged_td = np.array(merged_td)
        merged_counts = np.array(merged_counts)
        merged_err = np.array(merged_err)

        # ── 5. Détection et remplissage des gaps par NaN ───────────────────────
        # Un gap est détecté quand l'écart entre deux pas de temps consécutifs
        # est supérieur à 2 * max(timedel[i], timedel[i+1]).

        final_t = [merged_t[0]]
        final_td = [merged_td[0]]
        final_counts = [merged_counts[0]]
        final_err = [merged_err[0]]

        for i in range(1, len(merged_t)):
            dt = merged_t[i] - merged_t[i - 1]
            gap_thr = 2.0 * max(merged_td[i - 1], merged_td[i])

            if dt > gap_thr:
                # Remplissage du gap par NaN avec un pas synthétique
                gap_step = merged_td[i - 1]
                t_fill = merged_t[i - 1] + gap_step
                while t_fill < merged_t[i] - gap_step / 2:
                    final_t.append(t_fill)
                    final_td.append(gap_step)
                    final_counts.append(np.full(n_energy, np.nan))
                    final_err.append(np.full(n_energy, np.nan))
                    t_fill += gap_step

            final_t.append(merged_t[i])
            final_td.append(merged_td[i])
            final_counts.append(merged_counts[i])
            final_err.append(merged_err[i])

        final_t = np.array(final_t)
        final_td = np.array(final_td)
        final_counts = np.array(final_counts)
        final_err = np.array(final_err)

        # ── 6. Reconversion en temps relatif au MJDREF de référence ───────────
        time_rel = final_t - ref_mjdref * 86400.0

        # ── 7. Construction des HDUs et écriture unique ────────────────────────
        n_steps = len(final_t)

        col_data = fits.ColDefs([
            fits.Column(name='time', array=time_rel, format='D'),
            fits.Column(name='timedel', array=final_td, format='D'),
            fits.Column(name='rcr', array=np.zeros(n_steps, dtype=np.uint8), format='B'),
            fits.Column(name='triggers', array=np.zeros(n_steps, dtype=np.int64), format='K'),
            fits.Column(name='triggers_comp_err', array=np.zeros(n_steps, dtype=np.float64), format='D'),
            fits.Column(name='counts', array=final_counts, format=f'{n_energy}D'),
            fits.Column(name='counts_comp_err', array=final_err, format=f'{n_energy}D'),
            fits.Column(name='control_index', array=np.zeros(n_steps, dtype=np.int64), format='K'),
        ])
        hdu_data = fits.BinTableHDU.from_columns(col_data, header=data_header)
        hdu_data.name = 'DATA'

        col_energies = fits.ColDefs([
            fits.Column(name='channel', array=ref_channel, format='K'),
            fits.Column(name='e_low', array=ref_e_low, format='D'),
            fits.Column(name='e_high', array=ref_e_high, format='D'),
        ])
        hdu_energies = fits.BinTableHDU.from_columns(col_energies, header=energies_header)
        hdu_energies.name = 'ENERGIES'

        hdul_out = fits.HDUList([
            fits.PrimaryHDU(header=primary_header),
            fits.BinTableHDU(control_data, header=control_header, name='CONTROL'),
            hdu_data,
            hdu_energies,
        ])

        hdul_out.writeto(outfile, overwrite=True)
        hdul_out.close()

        n_gap = len(final_t) - len(merged_t)
        print(f"[merge_stix_fits] {len(file_list)} fichiers | "
              f"{len(final_t)} pas de temps "
              f"({len(merged_t)} après overlap, {n_gap} NaN de gap) → {outfile}")