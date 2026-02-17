# Quick Start Inference (First‑Pass → VideoTile)

## Obiettivo scientifico
Produrre, per ogni evento, una sequenza temporale di ritagli (tile) centrati sul ciclone, a risoluzione originale, così da alimentare il secondo stadio ad alta risoluzione (VideoMAE) e da poter controllare visivamente la correttezza della localizzazione.

## 1) Output del first‑pass (centro e presenza)
L’inferenza del first‑pass genera due grandezze chiave:
1) una **probabilità di presenza** (presence)
2) una **heatmap** da cui si ricava il centro stimato (peak o soft‑argmax).

Il centro è espresso nello spazio **ridimensionato** (stretch) e va riportato nello spazio dell’immagine originale. Il passaggio usa i metadati di resize (scale_x/scale_y e pad).

**Dove approfondire:**
`moduli/firstpass/src/cyclone_locator/infer.py` (decodifica heatmap, soglie, conversioni)
`moduli/firstpass/src/cyclone_locator/transforms/letterbox.py` (mappe avanti/indietro)

## 2) Conversione al dominio originale
Una volta ottenute le coordinate nel dominio resized, si applica la trasformazione inversa verso l’immagine originale (1290×420). In modalità stretch, l’inversa è lineare e anisotropa:

x_orig = (x_resized - pad_x) / scale_x
y_orig = (y_resized - pad_y) / scale_y

Con stretch, `pad_x = pad_y = 0`.

**Dove approfondire:**
`moduli/firstpass/src/cyclone_locator/transforms/letterbox.py`

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

## 5) Stato della funzione “video‑tile”
Al momento, la costruzione della video‑tile è implementata nel percorso di visualizzazione ROI (notebook + helper).
Se serve un entry point “headless” per batch inference, va estratta in una funzione di utilità riusabile in pipeline.

### Implementazione consigliata (separata dal notebook)
Creare una funzione che:
- riceve frame originali + centro stimato + raggio ROI,
- clippa i bordi,
- estrae il crop quadrato,
- restituisce la sequenza di tile (numpy o tensor).

Suggerita collocazione:
`moduli/firstpass/src/cyclone_locator/utils/` oppure `moduli/firstpass/scripts/`.

## 6) Entry point rapidi
- **First‑pass inference:** `moduli/firstpass/infer.sh` → `src/cyclone_locator/infer.py`
- **ROI visual inspection:** `moduli/firstpass/notebooks/firstpass_videomae_roi_viz.ipynb`
- **Batch video per eventi:** `moduli/firstpass/scripts/render_all_events.py`

## 7) Cosa verificare per validazione scientifica
1) Coerenza tra centro stimato e ROI nel frame originale.
2) Stabilità temporale della ROI lungo i 16 frame.
3) Coerenza con ground truth (marker) nello spazio originale.
4) Assenza di distorsioni introdotte dal resize stretch.
