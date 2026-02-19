# Quick Start Inference (First‑Pass → VideoTile)



## 1) Output del first‑pass (centro e presenza)
L’inferenza del first‑pass genera due grandezze chiave:
1) una **probabilità di presenza** (presence)
2) una **heatmap** da cui si ricava il centro stimato (peak o soft‑argmax).

Il centro è espresso nello spazio **ridimensionato** (stretch) e va riportato nello spazio dell’immagine originale. Il passaggio usa i metadati di resize (scale_x/scale_y e pad).

**Dove approfondire:**
`moduli/firstpass/src/cyclone_locator/infer.py` (decodifica heatmap, soglie, conversioni)
`moduli/firstpass/src/cyclone_locator/transforms/letterbox.py` (mappe avanti/indietro)



## 3) Costruzione della ROI (video‑tile)
La ROI è definita come **quadrato** centrato su `(x_orig, y_orig)` con raggio `roi_radius_px`.
Questo raggio può essere fisso (config inferenza) o dinamico (derivato dalla larghezza del picco).
Il ritaglio avviene direttamente sull’immagine originale, senza distorsioni, preservando il rapporto d’aspetto locale.

Un'altra modalità è stata scelta in cui il ritaglio non viene centrato sul `(x_orig, y_orig)` ma sulla tile avente offsets standard che contiene il centro.

**Esito atteso:** una sequenza di frame originali ritagliati e contenenti il ciclone, pronta per il secondo stadio o per analisi qualitative.

## 4) Video‑tile temporale
Per ogni evento si costruisce una sequenza temporale di 16 frame.
Si usa il centro stimato nel frame centrale per definire la ROI e applicarla coerentemente a tutti i frame della finestra temporale. Questo garantisce coerenza spaziale all’interno della clip.

**Dove approfondire (visualizzazione/ROI):**
`demetra/notebooks/firstpass_videomae_roi_viz.ipynb`
`demetra/notebooks/firstpass_videomae_roi_viz_utils.py`

