# Prompt per Codex — Video letterbox (pred vs GT, unico MP4)

> **Contesto**: È già disponibile `preds.csv` ottenuto dall’inferenza **first‑pass** sulle immagini **letterbox SxS** (senza ROI). Vogliamo produrre **un unico video MP4** che copra tutte le finestre temporali **contigue** presenti nel **manifest di test**, disegnando il **puntino rosso** per la **predizione** e il **puntino verde** per la **ground truth** (se disponibile).
>
> **Vincolo importante**: **non creare** le funzioni di rendering già presenti o caricate nel notebook “View  tracking preds videomae” (es. `make_animation_parallel_ffmpeg`, salvataggio `frames.txt`, ecc.). **Le copierà l’utente** dal repo VideoMAE. Qui devi **solo** creare una pipeline che produce un output simile, prendendo spunto e integrando se serve, le funzioni utili già presenti lì.

---

## Obiettivo

Aggiungere un **tool CLI** (modulo Python eseguibile) che:

1. legge `preds.csv` (predizioni in spazio **letterbox SxS**),
2. legge il **manifest di test** (per GT presence e, se presenti, GT center in SxS),
3. **allinea** predizioni e GT per frame,
4. costruisce i **frame PNG** (punto rosso pred, punto verde GT) usando le **immagini letterbox** SxS,
5. raggruppa i frame in **segmenti contigui** in base a `gap_minutes` e **produce un unico MP4** concatenato in ordine temporale,


---

## Nuovo file

* `tools/render_letterbox_video.py` (nuovo modulo CLI)

---

## Interfaccia CLI

```
python -m tools.render_letterbox_video \
  --preds_csv outputs/eval/test/preds_test.csv \
  --manifest_csv data/manifests/test.csv \
  --out_mp4 outputs/videos/med_letterbox_test.mp4 \
  --frames_dir outputs/videos/frames_letterbox \
  --fps 12 \
  --gap-minutes 20 \
  --segment-slate-seconds 0 \
  --image-size 512 \
  --path-col resized_path \
  --pred-x-col x_g --pred-y-col y_g \
  --gt-x-col x_pix_resized --gt-y-col y_pix_resized
```

### Argomenti (dettaglio)

* `--preds_csv`: CSV con almeno colonne: `resized_path` (o `image_path`), `presence_prob`, `x_g`, `y_g`.
* `--manifest_csv`: CSV test con colonne per join (`resized_path` o `image_path`), `presence` e **se disponibili** `x_pix_resized`, `y_pix_resized`.
* `--out_mp4`: percorso MP4 finale unico.
* `--frames_dir`: cartella temporanea per i PNG inter-frame (verrà creata; pulizia opzionale **non** richiesta).
* `--fps`: frame rate del video.
* `--gap-minutes`: se il **delta** tra timestamp consecutivi supera questa soglia, considera “fine segmento” e inizia un nuovo segmento.
* `--segment-slate-seconds`: secondi di **interstiziale** nero (o slate semplice) tra segmenti; se 0, si concatena diretto.
* `--image-size`: S del letterbox (default 512; deve combaciare con i file immagine SxS).
* `--path-col`: nome colonna che punta al **file immagine SxS** (default `resized_path`).
* `--pred-x-col/--pred-y-col`: colonne con coordinate **pred** in SxS (default `x_g/y_g`).
* `--gt-x-col/--gt-y-col`: colonne con coordinate **GT** in SxS (default `x_pix_resized/y_pix_resized`).

> **Nota**: non introdurre conversioni/letterbox inverse; il video è **sulle SxS**.

---

## Specifiche operative

### 1) Parsing e join

* Carica `preds_csv` e `manifest_csv` in DataFrame.
* Normalizza la colonna path: se `--path-col` non esiste nel CSV, auto‑rileva tra `resized_path`, `image_path`, `path` (documenta in log la scelta).
* Effettua il **join** su questa colonna per avere, per ogni frame, predizione + GT.
* **Timestamp**: se non è presente una colonna `datetime_iso`, ricava il tempo **dal nome file** con le stesse regex/formatter usati in repo (o re‑implementa parsing minimale:

  * prova pattern `yyyy-MM-ddTHH-mm`, `yyyyMMdd_HHmm`, ecc.).
* Ordina per timestamp crescente.

### 2) Filtri opzionali


### 3) Segmentazione per contiguità temporale

* Scorri i frame ordinati; calcola il **delta** (in minuti) tra `t[i]` e `t[i-1]`.
* Inizia un **nuovo segmento** quando `delta > gap_minutes`.
* Registra l’indice di start/stop di ogni segmento.

### 4) Disegno dei frame PNG (soluzione A: punto singolo)

* Per **ogni riga** genera un PNG a partire dall’immagine SxS indicata da `path-col`.
* Disegna **puntino rosso** (pred): coordinate `(pred-x-col, pred-y-col)` (clamp a [0,S‑1]).
* Se GT center è disponibile: disegna **puntino verde** (GT): `(gt-x-col, gt-y-col)` (clamp).
* Dimensione puntini: raggio 3–4 px; bordo nero 1 px per visibilità sullo sfondo.
* (Facoltativo) sopra/sotto: piccola etichetta testo «time: YYYY‑MM‑DD HH:MM • prob: 0.73».
* **Non** implementare trail/traccia cumulata (soluzione B **non** richiesta).

> **Colori fissi**: rosso = predizione (`#FF0000`), verde = ground truth (`#00FF00`).

### 5) Lista frame & encoding video

* Raccogli i PNG di **tutti i segmenti** in **una sola** lista `frames.txt` (ordine temporale).
* Tra un segmento e il successivo:

  * se `--segment-slate-seconds > 0`, inserisci un numero di repliche dell’ultimo PNG di segmento o un PNG nero “slate” di durata equivalente (compatibile con `make_animation_parallel_ffmpeg`).
  * altrimenti, non inserire nulla: concatenazione secca.
* Invoca **le funzioni già presenti** (che l’utente copia dal repo VideoMAE) per:

  1. generare `frames.txt` (percorso PNG + durata/fps);
  2. lanciare ffmpeg con `fps` configurato e produrre `--out_mp4`.

> **Vincolo**: **non** re‑implementare `make_animation_parallel_ffmpeg` o utility equivalenti. Limìtati a **chiamarle**. Se non presenti a runtime, emetti errore chiaro: «Funzione di rendering non disponibile — copiare i tool di View MED nel path indicato».

### 6) Logging e robustezza

* Log iniziale: percorsi input, #righe join riuscite, #frame filtrati, #segmenti creati, fps.
* Warning se mancano GT center: disegna **solo** il puntino rosso; niente verde.
* Clamp coordinate e verifica che i file immagine esistano. Frame mancanti → log warning e **skippa**.
* Alla fine: logga path dell’MP4 e #frame inclusi.

---

## Accettazione

* Esecuzione **senza eccezioni** sulle cartelle di test.
* MP4 **unico** con **tutti i segmenti contigui** (separati solo dal gap detection); ordine temporale rispettato.
* Visualizzazione: **rosso = predizione**, **verde = GT**; assenza di GT → solo rosso.
* Parametri `--fps` e `--gap-minutes` influenzano correttamente durata e segmentazione.
* Il tool **non** crea funzioni già esistenti nei notebook “View MED tracking preds”; fallisce con errore leggibile se non sono state copiate.

---

## Note

* Il join path deve essere **case‑sensitive** su Linux; normalizza gli slash per evitare `.../resized\file.png`.
* Se la colonna timestamp non esiste in nessuno dei due CSV, fai parsing da filename (regex interna minima), documentando nel log il formato rilevato.
* Non produrre `ROI/` né `_orig` (non richiesti qui). Il video è **in SxS**.
