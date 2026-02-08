# ATBD Source Trace - modulo videomae

Questa mappa riepiloga tutti i documenti .md del repository riletti per l'ATBD e descrive, per ciascuno, se e come il contenuto e' confluito nelle sezioni dell'ATBD oppure perche' e' stato escluso. L'ATBD resta focalizzato sul modulo videomae, quindi i documenti di altri moduli vengono segnalati come fuori scope.

## README.md
Uso nell'ATBD: nessun contenuto tecnico specifico e' confluito nell'ATBD, poiche' il file descrive il repository in modo generale.
Esclusioni: escluso perche' non aggiunge dettagli tecnici rispetto al modulo videomae.

## atbd/prompt_for_atbd.md
Uso nell'ATBD: usato come istruzione di struttura e vincoli redazionali, non come fonte tecnica.
Esclusioni: non contiene contenuti scientifici o tecnici da integrare.

## atbd/atbd_start.md
Uso nell'ATBD: non usato come fonte, essendo una bozza preliminare.
Esclusioni: escluso perche' le informazioni sono state sostituite dalla rilettura completa dei documenti originali.

## atbd/atbd_start2.md
Uso nell'ATBD: non usato come fonte, essendo una bozza intermedia.
Esclusioni: escluso perche' l'ATBD finale deriva dai documenti originali del modulo e non da sintesi precedenti.

## atbd/ATBD_videomae.md
Uso nell'ATBD: questo file e' l'output finale dell'ATBD, non una fonte.
Esclusioni: non applicabile.

## atbd/ATBD_videomae_docmap.md
Uso nell'ATBD: questo file e' la mappa di tracciabilita', non una fonte.
Esclusioni: non applicabile.

## moduli/videomae/AGENTS.md
Uso nell'ATBD: confluisce nella sezione di provenance per descrivere entrypoint principali e mappa dei documenti.
Esclusioni: le note operative generali non incidono su algoritmo o assunzioni scientifiche.

## moduli/videomae/README.md
Uso nell'ATBD: alimenta la sezione di scopo e contesto, la panoramica del prodotto e la provenance, oltre alle reference bibliografiche.
Esclusioni: i dettagli di comando per download e installazione sono stati esclusi per evitare contenuti troppo implementativi.

## moduli/videomae/README_MORE.md
Uso nell'ATBD: supporta le sezioni su input e pre-processing, training e verifica, fornendo una mappa ragionata dei notebook.
Esclusioni: le istruzioni di installazione e i dettagli di dipendenze sono stati esclusi perche' troppo legati all'ambiente.

## moduli/videomae/docs/INSTALL.md
Uso nell'ATBD: nessuno, per rispettare il vincolo di non includere dettagli di librerie e setup.
Esclusioni: escluso perche' riguarda solo l'ambiente di installazione.

## moduli/videomae/docs/PRETRAIN.md
Uso nell'ATBD: informazioni su dual masking, formati CSV di pretraining e iperparametri documentati in fase di training.
Esclusioni: lo script di lancio completo e le variabili d'ambiente sono state escluse in quanto operative.

## moduli/videomae/docs/specialization.md
Uso nell'ATBD: descrizione del flusso di specialization, della scalatura degli iperparametri e delle limitazioni note.
Esclusioni: dettagli di API interne esclusi per il vincolo di non citare funzioni o classi.

## moduli/videomae/docs/classification.md
Uso nell'ATBD: workflow di classificazione, gestione di training e validazione, limitazioni su mixup e loss pesata.
Esclusioni: dettagli di implementazione specifici esclusi per coerenza con lo stile ATBD.

## moduli/videomae/docs/TRACKING.md
Uso nell'ATBD: definizione del task di tracking, uso del frame finale, perdita MSE e struttura della head di regressione.
Esclusioni: riferimenti a nomi di classi o file di implementazione.

## moduli/videomae/docs/tracking.md
Uso nell'ATBD: flusso train/val del tracking e output di checkpoint.
Esclusioni: dettagli di API interne esclusi.

## moduli/videomae/docs/inference_classification.md
Uso nell'ATBD: modalita' di inferenza e formati di output per predizioni, logits ed embedding.
Esclusioni: dettagli su formati tecnici NPZ non essenziali all'ATBD.

## moduli/videomae/docs/Build_dataset_videoMAE.md
Uso nell'ATBD: descrizione completa del dataset building supervisionato e non supervisionato, con clip da 16 frame e output CSV.
Esclusioni: note TODO su pulizia CSV non rilevanti all'obiettivo scientifico.

## moduli/videomae/docs/make_dataset_from_rgb.md
Uso nell'ATBD: descritto indirettamente nella sezione di pre-processing come dispatcher per la generazione dei dataset e richiamato nelle limitazioni per le flag mutuamente esclusive.
Esclusioni: dettagli di CLI e percorsi specifici esclusi perche' troppo operativi.

## moduli/videomae/docs/Experiment_dataset.md
Uso nell'ATBD: relabeling temporale, split CL10, split medicanes e dataset full-year.
Esclusioni: TODO non consolidati.

## moduli/videomae/docs/Analyze_Manos_tracks.md
Uso nell'ATBD: unificazione delle tracce, conversione geo->pixel, subset medicanes e aggiornamento finestre temporali.
Esclusioni: dettagli diagnostici riga per riga considerati non essenziali.

## moduli/videomae/docs/Cloud_index.md
Uso nell'ATBD: definizione e uso dell'indice di nuvolosita', dataset cloudy/clear-sky e impiego in verifica.
Esclusioni: dettagli di UI e dipendenze di notebook non pertinenti.

## moduli/videomae/docs/medicane_utils_geo_const.md
Uso nell'ATBD: convenzioni di conversione geo->pixel e inversione asse Y.
Esclusioni: dettagli di librerie e plotting esclusi per evitare riferimenti troppo implementativi.

## moduli/videomae/docs/pixel_km_conversion.md
Uso nell'ATBD: pipeline pixel->lat/lon->km, formula haversine e vincolo della griglia 1290x420.
Esclusioni: riferimenti a file e righe di codice non inclusi.

## moduli/videomae/docs/metrics.md
Uso nell'ATBD: definizione delle metriche POD, FAR, CSI, HSS e Balanced Accuracy.
Esclusioni: immagine della confusion matrix non riportata nel testo.

## moduli/videomae/docs/Model_stats.md
Uso nell'ATBD: workflow di valutazione quantitativa, filtri temporali, analisi per label.
Esclusioni: dettagli di path locali e import non rilevanti.

## moduli/videomae/docs/Predict_general_data.md
Uso nell'ATBD: post-processing da predizioni video a frame, merge con master dataframe e animazioni mediterranee.
Esclusioni: esempi di path specifici.

## moduli/videomae/docs/View_MED_val_preds.md
Uso nell'ATBD: validazione su scala mediterranea e gestione tile mancanti nelle animazioni.
Esclusioni: dettagli su tool di rendering e configurazioni di PATH.

## moduli/videomae/docs/View_test_tiles.md
Uso nell'ATBD: case study qualitativi e animazioni per cicloni specifici.
Esclusioni: sezioni di debug e helper locali.

## moduli/videomae/docs/View_tracking_tiles.md
Uso nell'ATBD: ispezione visiva del dataset di tracking e overlay del centro.
Esclusioni: dettagli di implementazione.

## moduli/videomae/docs/View_MED_tracking_preds.md
Uso nell'ATBD: pipeline di visualizzazione del tracking su scala mediterranea, espansione tile mancanti e rendering finale.
Esclusioni: comandi di rendering dettagliati e variabili interne.

## moduli/videomae/docs/Verifica_patches.md
Uso nell'ATBD: verifica qualitativa delle ricostruzioni MAE e analisi delle maschere.
Esclusioni: dettagli di ricostruzione manuale troppo implementativi.

## moduli/videomae/docs/Plot_train_loss.md
Uso nell'ATBD: confronto di curve di loss e metriche su run diverse.
Esclusioni: liste di file di log specifici.

## moduli/videomae/docs/Plot_compare_metrics.md
Uso nell'ATBD: confronto tra FPR e FNR su esperimenti diversi.
Esclusioni: dettagli di plotting.

## moduli/videomae/docs/learning_rate_overview.md
Uso nell'ATBD: layer-wise learning rate e schedulazione step-level nel tracking.
Esclusioni: snippet di codice e istruzioni operative.

## moduli/videomae/docs/distributed_training_summary.md
Uso nell'ATBD: distribuzione multi-GPU/multi-nodo e scalatura di batch size e learning rate.
Esclusioni: script sbatch completo e dettagli di ambiente.

## moduli/videomae/docs/Video_cyclones_cut.md
Uso nell'ATBD: aggiornamento manuale delle finestre temporali con `new_cyc_limits.csv`.
Esclusioni: dettagli di UI e widget.

## moduli/videomae/docs/training_call_tree.md
Uso nell'ATBD: non utilizzato, file privo di contenuto.
Esclusioni: non applicabile.

## moduli/videomae/misc/todo.md
Uso nell'ATBD: non usato perche' contiene backlog e idee non consolidate.
Esclusioni: tutto il contenuto e' non definitivo.

## moduli/videomae/misc/done.md
Uso nell'ATBD: non usato perche' le definizioni delle metriche sono gia' coperte in `metrics.md`.
Esclusioni: note operative non consolidate.

## moduli/firstpass/AGENTS.md
Uso nell'ATBD: non usato perche' appartiene a un modulo diverso da videomae.
Esclusioni: fuori scope.

## moduli/firstpass/README.md
Uso nell'ATBD: non usato perche' descrive un modulo diverso.
Esclusioni: fuori scope.

## moduli/firstpass/creazione_video_track_preds.md
Uso nell'ATBD: non usato perche' il contenuto e' relativo al modulo firstpass.
Esclusioni: fuori scope.

## moduli/firstpass/pixel_km_conversion.md
Uso nell'ATBD: non usato perche' il contenuto e' relativo al modulo firstpass.
Esclusioni: fuori scope.

## moduli/firstpass/multi-gpu-multi-node_training.md
Uso nell'ATBD: non usato perche' il contenuto e' relativo al modulo firstpass.
Esclusioni: fuori scope.

## moduli/firstpass/vision_cyc_firstpass.md
Uso nell'ATBD: non usato perche' il contenuto e' relativo al modulo firstpass.
Esclusioni: fuori scope.
