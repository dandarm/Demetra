**1. Executive Summary**
- Il modulo VideoMAE estende VideoMAEv2 per lo studio dei medicanes, con pretraining auto-supervisionato e fine‑tuning supervisionato su immagini satellitari AirmassRGB (SEVIRI MSG), includendo detection e tracking del centro ciclone. Da `moduli/videomae/README.md` riga 9-15.
- Sono previsti due task principali: classificazione cyclone/no‑cyclone e regressione delle coordinate del centro ciclone. Da `moduli/videomae/README.md` riga 13-15 e `moduli/videomae/docs/TRACKING.md` riga 6-10.
- I dataset sono costruiti a partire da sequenze video e tracce Manos, con pipeline dedicate per pretraining e supervised fine‑tuning. Da `moduli/videomae/README.md` riga 34-39 e `moduli/videomae/docs/Build_dataset_videoMAE.md` riga 3-5.
- Sezione da completare: mancano obiettivi quantitativi finali e indicatori di performance attesi (non citati nei docs).

**2. Introduzione e scopo del documento**
- Il progetto mira ad adattare VideoMAE v2 ai medicanes, con script e notebook per dataset, addestramento e inferenza. Da `moduli/videomae/README.md` riga 9-15 e `moduli/videomae/README_MORE.md` riga 1-4.
- I principali entry point (specialization, classification, tracking, inference) e la mappa dei documenti sono esplicitati per la navigazione del modulo. Da `moduli/videomae/AGENTS.md` riga 12-35.
- Sezione da completare: manca una dichiarazione formale dello scopo ATBD e del pubblico target (non presente nei docs).

**3. Panoramica del prodotto / variabile misurata**
- Il prodotto copre due output principali: detection binaria del ciclone (cyclone/no‑cyclone) e tracking del centro ciclone come regressione (x,y) in pixel. Da `moduli/videomae/docs/classification.md` riga 1-2 e `moduli/videomae/docs/TRACKING.md` riga 6-10.
- Il tracking usa coordinate del centro ciclone nell’ultimo frame di ciascun clip. Da `moduli/videomae/docs/TRACKING.md` riga 8-10.
- Il pretraining auto-supervisionato è orientato alla ricostruzione di patch su sequenze AirmassRGB. Da `moduli/videomae/README.md` riga 13-14 e `moduli/videomae/README.md` riga 36-36.
- Sezione da completare: manca una descrizione esplicita della variabile geofisica “misurata” (es. intensità del ciclone), oltre alle etichette binarie/coordinate.

**4. Descrizione matematica completa degli algoritmi, equazioni**
- Per il tracking si ottimizza una `MSELoss()` tra coordinate predette e ground‑truth. Da `moduli/videomae/docs/TRACKING.md` riga 18-21.
- Le metriche di classificazione sono definite con formule esplicite: POD/Recall, FAR, CSI, HSS e Balanced Accuracy con formule dalla confusion matrix. Da `moduli/videomae/docs/metrics.md` riga 9-87.
- La conversione errori pixel→km usa la formula dell’haversine con raggio terrestre 6371.0088 km e passaggi dettagliati (dlat, dlon, a, c). Da `moduli/videomae/docs/pixel_km_conversion.md` riga 46-57.
- Sezione da completare: manca la definizione formale della loss di classificazione e dell’obiettivo MAE (ricostruzione/mascheramento) in termini matematici; nei docs è citato il masking ma non l’equazione di loss.

**5. Assunzioni teoriche e loro giustificazione, semplificazioni esplicite**
- Per il tracking si usano solo tile etichettate come ciclone (`label==1`), assumendo che il modello impari solo su esempi positivi. Da `moduli/videomae/docs/TRACKING.md` riga 6-8.
- Il training di tracking assume che le coordinate target siano quelle dell’ultimo frame del clip (semplificazione temporale). Da `moduli/videomae/docs/TRACKING.md` riga 8-10.
- Le pipeline di dataset assumono che immagini e tracce Manos siano già scaricate e temporalmente coerenti. Da `moduli/videomae/docs/Build_dataset_videoMAE.md` riga 35-36.
- Alcuni comportamenti sono hard‑coded: mixup disattivato e loss pesata commentata. Da `moduli/videomae/docs/classification.md` riga 39-41.
- Per le conversioni geografiche si assume una griglia lat/lon fissa 1290×420 e inversione dell’asse Y. Da `moduli/videomae/docs/medicane_utils_geo_const.md` riga 12-15 e `moduli/videomae/docs/pixel_km_conversion.md` riga 78-79.
- Sezione da completare: manca la giustificazione scientifica delle finestre temporali e delle dimensioni delle tile, oltre alla validità fisica delle assunzioni sul centro ciclone.

**6. Dati di Input**
- Dati satellitari AirmassRGB (EUMETSAT) usati come input video. Da `moduli/videomae/README.md` riga 9-9 e 69-76.
- Tracce Manos (`TRACKS_CL*.dat`) integrate e convertite in coordinate pixel per dataset supervisionati. Da `moduli/videomae/docs/Analyze_Manos_tracks.md` riga 4-6 e 22-25.
- I dataset video sono costruiti in sequenze da 16 frame e salvati in folder con CSV di riferimento. Da `moduli/videomae/docs/Build_dataset_videoMAE.md` riga 3-4 e 19-22.
- Formato dati pretraining: CSV con linee `video_path, 0, -1` o `frame_folder_path, start_index, total_frames`. Da `moduli/videomae/docs/PRETRAIN.md` riga 16-21.
- Sezione da completare: manca una specifica formale del formato dei CSV supervisionati (colonne e tipi) in tutti i pipeline.

**7. Logica dell’Algoritmo (flowchart/pseudocodice)**
- Classificazione: parsing argomenti → DataLoader train/test/val → modello TIMM → optimizer/scheduler → train/val → checkpoint best. Da `moduli/videomae/docs/classification.md` riga 7-18.
- Tracking: parsing → DataManager → modello tracking → MSELoss → train_one_epoch/evaluate → checkpoint best. Da `moduli/videomae/docs/tracking.md` riga 4-14.
- Specialization: parsing → DDP → modello MAE → dataset/dataloader → scheduler coseno → train/test periodico. Da `moduli/videomae/docs/specialization.md` riga 7-18.
- Inference classificazione: parsing → DataLoader val → modello → raccolta logits/embedding/pred → merge multi‑rank → CSV. Da `moduli/videomae/docs/inference_classification.md` riga 5-15.
- Costruzione dataset: master df → grouping temporale → tile video → bilanciamento/split → CSV finali. Da `moduli/videomae/docs/Build_dataset_videoMAE.md` riga 9-22.
- Flowchart VideoMAEv2 disponibile come immagine di riferimento. Da `moduli/videomae/README.md` riga 85-87.
- Sezione da completare: mancano flowchart formali dei tre pipeline adattati (solo descrizioni testuali).

**8. Dettagli implementativi e computazionali**
- Training/inference distribuiti via `utils.get_resources()` con detection torchrun/mpirun/srun e DDP con backend `nccl`. Da `moduli/videomae/docs/distributed_training_summary.md` riga 5-59.
- Batch size e learning rate sono scalati con `world_size`; DDP usato per sincronizzare gradienti. Da `moduli/videomae/docs/distributed_training_summary.md` riga 53-58.
- Layer-wise LR scaling e schedule step‑level in tracking; `lr_scale` per gruppi parametri e passaggio a `train_one_epoch`. Da `moduli/videomae/docs/learning_rate_overview.md` riga 3-62.
- Inferenza distribuita aggrega NPZ per logits/embedding e salva CSV finali. Da `moduli/videomae/docs/inference_classification.md` riga 6-15 e 31-34.
- Requisiti ambiente: PyTorch >=1.12, timm 0.4.12 raccomandato. Da `moduli/videomae/docs/INSTALL.md` riga 14-18.
- Sezione da completare: manca stima dei costi computazionali (GPU‑hours, memoria, throughput).

**9. Parametri utilizzati e loro calibrazione**
- Pretraining: parametri esempio `mask_ratio=0.9`, `decoder_mask_ratio=0.5`, `decoder_depth=4`, `num_frames=16`, `sampling_rate=4`, `batch_size=32`, `lr=6e-4`, `opt=adamw`. Da `moduli/videomae/docs/PRETRAIN.md` riga 42-60.
- Cloud index: dataset “cloudy” con soglia `avg_cloud_idx > 0.2`. Da `moduli/videomae/docs/Cloud_index.md` riga 21-25.
- Tile size implicita 224×224 nella funzione `inside_tile_faster` (usata con coordinate già proiettate). Da `moduli/videomae/docs/pixel_km_conversion.md` riga 26-26.
- Sezione da completare: mancano strategie di calibrazione dei parametri (grid search, validation‑based tuning) e range consigliati.

**10. Fase di addestramento dell’algoritmo**
- Specialization: fine‑tuning auto-supervisionato con logging TensorBoard, scheduler coseno, checkpoint periodici. Da `moduli/videomae/docs/specialization.md` riga 1-18.
- Classificazione: training loop con validazione periodica e salvataggio best checkpoint. Da `moduli/videomae/docs/classification.md` riga 7-18.
- Tracking: training con MSE, valutazione e salvataggio best checkpoint. Da `moduli/videomae/docs/TRACKING.md` riga 18-23 e `moduli/videomae/docs/tracking.md` riga 6-14.
- Pretraining distribuito con script Slurm/torch.distributed e dataset CSV. Da `moduli/videomae/docs/PRETRAIN.md` riga 3-65.
- Sezione da completare: mancano criteri di early stopping e linee guida sui checkpoint da usare per deployment.

**11. Procedure di validazione e test**
- Validazione/metriche: notebook Model_stats calcola accuracy, FPR/FNR/POD/FAR e confusion matrix. Da `moduli/videomae/docs/Model_stats.md` riga 3-26.
- Analisi temporale e per label specifiche con `plot_metrics_over_time` e confusion counts per delta_time. Da `moduli/videomae/docs/Model_stats.md` riga 28-29.
- Inference_classification salva metriche su `inference_metrics.txt` e produce CSV/NPZ. Da `moduli/videomae/docs/inference_classification.md` riga 14-35.
- Validazione visuale su Mediterraneo con merge predizioni/df_video e rendering animazioni. Da `moduli/videomae/docs/View_MED_val_preds.md` riga 6-35.
- Sezione da completare: mancano protocolli quantitativi di split e criteri di accettazione (threshold per metriche).

**12. Limitazioni note dell’algoritmo**
- `specialization.py` ha un `__main__` che solleva `TypeError` per argomento mancante. Da `moduli/videomae/docs/specialization.md` riga 22-24.
- Mixup forzato a `False` e loss pesata commentata in classification. Da `moduli/videomae/docs/classification.md` riga 39-41.
- In make_dataset_from_rgb le flag sono mutuamente esclusive, nessun supporto a combinazioni. Da `moduli/videomae/docs/make_dataset_from_rgb.md` riga 28-31.
- La griglia lat/lon è valida solo per immagini 1290×420; serve rigenerare se cambia dimensione. Da `moduli/videomae/docs/pixel_km_conversion.md` riga 78-78.
- Cloud index: nota TODO su uso canale verde dominante. Da `moduli/videomae/docs/Cloud_index.md` riga 34-34.
- Sezione da completare: mancano limiti quantitativi di generalizzazione e scenari d’uso non coperti.

**13. Analisi degli Errori e Incertezze**
- Metriche di errore per classificazione basate su confusion matrix (TP, FP, FN, TN) e skill scores. Da `moduli/videomae/docs/metrics.md` riga 9-87.
- Model_stats calcola FPR/FNR/POD/FAR e analizza metriche per label temporali. Da `moduli/videomae/docs/Model_stats.md` riga 3-29.
- Per tracking, l’errore può essere espresso in km usando conversione pixel→lat/lon→haversine. Da `moduli/videomae/docs/pixel_km_conversion.md` riga 44-58.
- Sezione da completare: mancano analisi di incertezza statistica (es. intervalli di confidenza, varianza dei risultati).

**14. Casi particolari / screening / filtraggi / case study**
- Filtraggio per nuvolosità con dataset “cloudy/clear‑sky” e soglia cloud index. Da `moduli/videomae/docs/Cloud_index.md` riga 21-25.
- Relabeling basato su distanza temporale dal ciclone e split per classi/anni. Da `moduli/videomae/docs/Experiment_dataset.md` riga 6-21.
- Aggiornamento manuale finestre temporali dei cicloni con `new_cyc_limits.csv`. Da `moduli/videomae/docs/Video_cyclones_cut.md` riga 3-18.
- Case study visivo: generazione di animazioni Mediterraneo con predizioni di validazione. Da `moduli/videomae/docs/View_MED_val_preds.md` riga 29-35.
- Case study tracking: video Mediterraneo con tracce GT vs PRED e errori pixel/km. Da `moduli/videomae/docs/View_MED_tracking_preds.md` riga 33-37 e 45-48.
- Sezione da completare: mancano risultati quantitativi di un case study riportati in forma numerica.

**15. Riferimenti a baseline di codice**
- Entry point principali: `specialization.py`, `classification.py`, `tracking.py`, `inference_classification.py`. Da `moduli/videomae/AGENTS.md` riga 12-27.
- Pipeline dataset e notebook chiave per costruzione e analisi: `Build_dataset_videoMAE.md`, `Experiment_dataset.md`, `Analyze_Manos_tracks.md`. Da `moduli/videomae/AGENTS.md` riga 31-35.
- Sezione da completare: manca un mapping diretto a specifiche versioni/commit baseline.

Contraddizioni: non ho trovato contraddizioni esplicite tra i file di doc del modulo videomae.

**Riassunto finale**
- Sezioni popolate in modo sostanziale: 1, 3, 6, 7, 8, 10, 11, 13, 14, 15 (molte con dettagli operativi e pipeline).
- Sezioni parziali o da completare: 2, 4, 5, 9, 12 (mancano obiettivi formali, formule di loss per MAE/classificazione, criteri di calibrazione/accettazione, limiti quantitativi).
- Integrazioni suggerite (senza inventare dati):
  1) Documentare esplicitamente loss e architetture (MAE, classifier) in forma matematica.
  2) Definire protocolli di validazione (split, early stopping, threshold di performance).
  3) Aggiungere limiti quantitativi attesi e criteri di deployment (es. errori km accettabili per tracking).
