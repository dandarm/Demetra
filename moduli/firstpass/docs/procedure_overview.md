# Stato Attuale delle Procedure (First‑Pass + ROI)

## Obiettivo scientifico
Il progetto è diviso in due moduli principali. **First‑pass** (`moduli/firstpass/`) esegue la rilevazione full‑basin: stima presenza del ciclone e centro tramite heatmap su immagini ridimensionate. **Secondo stadio (VideoMAE)** (`moduli/videomae/`) usa le ROI del first‑pass per analisi ad alta risoluzione nel tempo. 

Questo modulo firstpass deve produrre, per ogni evento, una sequenza temporale di ritagli (tile) che contengono il ciclone, a risoluzione originale, così da alimentare il secondo stadio ad alta risoluzione (VideoMAE)

Il legame tra i due moduli è il passaggio di **centro e ROI** dall’immagine full‑basin al sotto‑ritaglio ad alta risoluzione.

## 1) Immagini operative e metadati di resize
Le immagini operative sono **stretchate** a lato fisso (es. 224×224) senza padding. Questo rende la mappa pixel‑immagine **lineare e anisotropa** (scale_x ≠ scale_y). I metadati salvati nel `letterbox_meta.csv` contengono **orig_w, orig_h, out_size, scale_x, scale_y, pad_x, pad_y**. In modalità stretch, `pad_x=pad_y=0`. Questi metadati sono generati da `scripts/make_letterboxed_copies.py` e sono usati per proiettare avanti/indietro le coordinate tra spazio originale e spazio ridimensionato.

Una volta ottenute le coordinate nel dominio resized, si applica la trasformazione inversa verso l’immagine originale (1290×420). In modalità stretch, l’inversa è lineare e anisotropa:

x_orig = (x_resized - pad_x) / scale_x
y_orig = (y_resized - pad_y) / scale_y

**Dove approfondire:** `moduli/firstpass/scripts/make_letterboxed_copies.py`, `moduli/firstpass/src/cyclone_locator/transforms/letterbox.py`.




## 2) Manifest e coordinate coerenti con lo stretch
I manifest train/val/test devono contenere `image_path`, `presence`, `cx`, `cy`, e **coordinate resized** (`x_pix_resized`, `y_pix_resized`) già coerenti con lo stretch. La conversione da `cx,cy` (spazio originale) a `x_pix_resized,y_pix_resized` avviene in `scripts/make_manifest_from_windows.py` ed è coerente con i metadati di stretch. Questo evita mismatch fra coordinate del manifest e immagini effettivamente usate durante training/visualizzazione.

**Dove approfondire:** `moduli/firstpass/scripts/make_manifest_from_windows.py`, `moduli/firstpass/mini_data_input/medicanes_new_windows.csv` (origine `x_pix/y_pix`).


## 3) Dataset e generazione heatmap
Il dataset principale `MedFullBasinDataset` legge i manifest e, per ogni frame, produce:
- immagine normalizzata (`image` o `video`),
- heatmap target centrata sulle coordinate resized,
- presenza (0/1),
- metadati di mappatura.
In presenza di `x_pix_resized/y_pix_resized`, il dataset usa direttamente tali coordinate; se assenti usa `cx/cy` + metadati per proiettarle nello spazio resized. La heatmap è costruita a risoluzione ridotta (`image_size / heatmap_stride`) con sigma controllato da config.

**Dove approfondire:** `moduli/firstpass/src/cyclone_locator/datasets/med_fullbasin.py`.


## 4) Output del first‑pass (centro e presenza)
L’inferenza del first‑pass genera due grandezze chiave:
1) una **probabilità di presenza** (presence)
2) una **heatmap** da cui si ricava il centro stimato (peak o soft‑argmax).

Il centro è espresso nello spazio **ridimensionato** (stretch) e va riportato nello spazio dell’immagine originale. Il passaggio usa i metadati di resize (scale_x/scale_y e pad).



## 5) Training first‑pass
Il training esegue:
- **head heatmap** per la localizzazione del centro,
- **head presenza** per classificare il frame/evento.
Le loss combinano errore heatmap e presenza. La densità dei clip è controllata da `manifest_stride` e dal `temporal_T/stride`. Il logging separa il tempo di training da quello di validazione/test, per valutare correttamente il costo di ciascuna fase.

**Entry point:** `moduli/firstpass/train_temporal.sh` (invoca `src/cyclone_locator/train.py`).


## 5) Inferenza e ROI
In inferenza, la heatmap predetta viene usata per ricavare il centro, usando una funzione che la converte in un valore di intensità (logsumexp dei topk valori della heatmap). Se la presenza supera la soglia, il centro della heatmap viene proiettato indietro nello spazio originale e usato per definire una **ROI quadrata** centrata sul ciclone. La ROI è la base per l’hand‑off verso il secondo stadio e per la diagnostica qualitativa.
La ROI è definita come **quadrato** centrato su `(x_orig, y_orig)` con raggio uguale al lato della tile input per videomae (224px).
Il ritaglio avviene direttamente sull’immagine originale non distorta.
Un'altra modalità è stata scelta in cui il ritaglio non viene centrato sul `(x_orig, y_orig)` ma sulla tile avente offsets standard (valori di default degli offsets) che contiene il centro. Si è mostrato che il tracking è migliore se la ROI viene ritagliata in questo modo.

L'ouput di questa fase è una sequenza di 16 frame a risoluzione originale ritagliati e contenenti il ciclone, cioè la videotile pronta per il secondo stadio videomae.

**Dove approfondire:** `moduli/firstpass/src/cyclone_locator/infer.py` (conversioni e soglie), `moduli/firstpass/src/cyclone_locator/transforms/letterbox.py` (mappe avanti/indietro).

## 6) Visualizzazione ROI (controllo scientifico)
La visualizzazione costruisce, per ciascun evento, una sequenza di frame, con il marker del centro predetto dal firstpass, quello predetto da Videomae, e quello del ground truth, che vengono proiettati nello stesso frame per verificare la coerenza. Questo consente di valutare correttezza delle trasformazioni e stabilità temporale della predizione.

**Entry point:** `moduli/firstpass/notebooks/firstpass_videomae_roi_viz.ipynb` e helper `moduli/firstpass/notebooks/firstpass_videomae_roi_viz_utils.py`.

## 8) Secondo stadio (VideoMAE)
Il modulo VideoMAE di tracking gestisce dataset e sequenze ad alta risoluzione. L’input principale è la sequenza ritagliata via ROI dal first‑pass, e l'output costituisce una rifinitura della posizione stimata approssimativamente da firstpass. Videomae ragguinge una precisione maggiore perché opera su immagini a risoluzione maggiore e per via del modello avanzato di processing video.

