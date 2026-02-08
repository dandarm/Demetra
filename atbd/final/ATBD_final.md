# ATBD_final

## 1. Scopo e contesto

### Summary


### Introduzione, scopo e contesto del prodotto  (_SKIP_)
Lo scopo del prodotto e' fornire una base algoritmica solida per rilevare e localizzare i cicloni mediterranei a partire da sequenze satellitari, mantenendo coerenza con le pratiche di analisi di eventi rari in meteorologia. Il contesto operativo e' definito da immagini AirmassRGB compatibili con la griglia geografica adottata (1290x420) e con la logica di tiling usata nella costruzione dei dataset; entro questi vincoli, il sistema permette di derivare un output interpretabile sia come decisione binaria di presenza del ciclone sia come stima spaziale del centro. Il pubblico di riferimento include ricercatori e operatori con competenze meteorologiche che necessitano di strumenti riproducibili per detection e tracking, affiancati da visualizzazioni e metriche piu' adatte a dataset sbilanciati. Il sistema opera su sequenze AirmassRGB con logiche di clip temporalmente coerenti e fornisce una catena di produzione che integra dataset, training, inferenza e post-processing in un flusso controllabile.

## 2. Panoramica prodotto, definizioni e specifiche

### Target variable
### Input data specs, coverage, resolution, cadence
### Output variables, units, valid ranges, griglia/risoluzione, convenzioni



## 3. Input data & pre-processing

### Dati di input

### Pre-processing e labeling




## 4. Algoritmo

### Overview

(figura VideoMAEv2_flowchart.png)
Fig. 5.1 – Technical details of the VideoMAEv2 model: encoder-decoder structure, with separate masking for each one. The latent representation vector (embedding) is shown between the encoder and the decoder.

### Flowchart (testuale)
```
AirmassRGB frames + TRACKS_CL7 tracks
  -> costruzione 'master' CSV dataframe (tile offset, label, lon/lat RC, x/y pixel RC)
  -> clip video (16 frame) + manifest CSV train/val/test
     -> (A) post-pretraining Masked AutoEncoder (MAE) su clip non etichettati
     -> (B) fine-tuning per cyclone detection su clip etichettati
     -> (C) fine-tuning per cyclone Rotation Center (RC) tracking (solo tile positive)
  -> inferenza + post-processing
  -> validazione quantitativa + visualizzazioni
```

La pipeline algoritmica per DeMeTrA integra la catena di pre-processing e i tre stadi di training.

```text
Dati SEVIRI RSS -> De-normalizzazione BT -> Composito Airmass RGB -> Crop Mediterraneo
-> Tiling 224x224 -> Stack 16 frame -> Etichettatura con TRACKS_CL7
-> Specialization training VideoMAE -> Classificazione presenza ciclone
-> Regressione del centro di rotazione
```

### Pseudocodice (sintetico)
[[4-flowchart_pseudocode.md]]


**Pretraining / specialization**
```
INPUT: clip non etichettati, parametri di mascheramento, checkpoint iniziale
SETUP: inizializza risorse distribuite e logging
MODEL: backbone MAE con decoder e maschere
FOR ogni epoca:
  aggiorna scheduler LR/WD
  ottimizza ricostruzione patch mascherate
  salva checkpoint periodici
  se previsto, esegui test periodico
OUTPUT: checkpoint + log
```

**Classificazione**
```
INPUT: clip etichettati (train/val/test), checkpoint iniziale opzionale
SETUP: risorse, logging, scalatura iperparametri
MODEL: backbone video + head binaria
FOR ogni epoca:
  training su train
  validazione periodica
  aggiorna best checkpoint
OUTPUT: best checkpoint + log
```

**Tracking**
```
INPUT: clip positive con target (x, y) dell'ultimo frame
SETUP: risorse, logging
MODEL: backbone video + regressore 2D
FOR ogni epoca:
  training con perdita MSE
  valutazione su test/val
  aggiorna best checkpoint
OUTPUT: best checkpoint tracking + log
```

**Inferenza classificazione**
```
INPUT: dataset di test/val, checkpoint
MODEL: carica modello
IF richiesta logits/embedding:
  raccogli shard multi-rank, merge, cleanup
ELSE:
  raccogli predizioni standard
OUTPUT: CSV predizioni + metriche
```



## 5. Fase di training


## 6. Verifica: validazione, test, incertezze, limiti

### Metriche

##### Classification

##### Tracking





## 7. Provenance (obbligatoria, compatta)

### Repository originale e licenza
Il modulo deriva da VideoMAE v2 <git url>, che è un codice pensato esclusivamente per la ricerca per video understanding e classificazione. La licenza è MIT, che permette di utilizzare, copiare, modificare, unire, pubblicare, distribuire, concedere in sublicenza e vendere il software.

### Modifiche locali al repository originale 
L'adattamento di VideoMAE per Demetra realizza un applicativo specifico per riconoscimento e comprensione di feature di sequenze video su immagini AirmassRGB, un dominio molto più ristretto rispetto quello per cui è stato pensato e sviluppato il codice alla fonte. Sono state poi implementate tutte le pipeline descritte sopra: costruzione del working dataset con tiling e videoclip, l'integrazione delle tracce dei cicloni come etichettatura per il task di classificazione, applicazioni di analisi e visualizzazione del dataset per labeling manuale. Viene aggiunto un percorso di tracking del centro con regressione a due coordinate. La pipeline include in aggiunta flussi di inferenza e post-processing per mosaici mediterranei. Sono presenti notebook per validazione quantitativa, e visualizzazioni di case study, insieme a un migliorato supporto esplicito al training distribuito. La fase di pretraining è stata accompagnata da ulteriori codici di verifica delle maschere e delle ricostruzioni di patch. 

### Entrypoint, layout del repo, config
Gli entrypoint principali coprono pretraining, classificazione, tracking e inferenza, mentre la pipeline dati include strumenti per la costruzione dei dataset e per il tracking. Il repository e' organizzato con una cartella `docs/` per notebook e guide, `misc/` per note operative e `scripts/` per job di training, e produce output standardizzati come `log.txt` e checkpoint best. La configurazione si basa su parametri via CLI e su CSV di input per train, validation e test.



## 8. References bibliografiche

