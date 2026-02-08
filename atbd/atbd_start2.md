# ATBD - VideoMAE (Medicanes)

## 1. Executive summary
Il modulo VideoMAE estende VideoMAEv2 al dominio dei medicanes su sequenze video satellitari AirmassRGB (SEVIRI MSG), integrando una fase di pretraining auto-supervisionato con fasi supervisionate per detection binaria e tracking del centro ciclone. Il risultato è un sistema capace di apprendere rappresentazioni spaziotemporali mediante mascheramento e ricostruzione di patch, di discriminare la presenza del ciclone su clip temporali e di stimare la posizione del centro in coordinate pixel. Quando necessario, le stime di posizione sono convertite in coordinate geografiche e distanze in km per fornire un errore fisicamente interpretabile. L’intero flusso è progettato come pipeline di costruzione dataset, training e inference distribuiti, affiancata da notebook di validazione, metriche e analisi visive. L’uso di tracce Manos come riferimento consente di definire target coerenti per classificazione e tracking, mentre le sequenze non etichettate alimentano il pretraining MAE.

## 2. Introduzione e scopo del documento
Questo documento segue lo stile ATBD ESA/NASA e intende descrivere in modo completo il fondamento teorico, algoritmico e operativo del modulo VideoMAE per medicanes. L’obiettivo è chiarire cosa viene misurato e prodotto, come i dati vengono trasformati lungo il workflow, quali assunzioni sono implicite e quali limiti sono noti, e con quali procedure si addestrano e si validano i modelli. Il documento integra le informazioni distribuite nei materiali di riferimento, includendo documenti di training, dataset building, inferenza, metriche e visualizzazione, ma riformula i processi in termini algoritmici e scientifici, evitando riferimenti diretti a classi, funzioni o librerie specifiche.

## 3. Panoramica del prodotto / variabile misurata
Il prodotto comprende tre livelli principali. Il primo è un livello di rappresentazioni auto-supervisionate, ottenute tramite masked autoencoding su clip video, finalizzato a specializzare il modello sulle dinamiche tipiche dei medicanes. Il secondo è un livello di classificazione binaria cyclone/no-cyclone, basato su sequenze di tile, con output espresso come label predetta e, opzionalmente, come logit o embedding per analisi successive. Il terzo è un livello di tracking del centro ciclone, espresso come regressione delle coordinate (x, y) in pixel relative alla tile; tali coordinate possono essere convertite in latitudine/longitudine e in distanza chilometrica rispetto al target. La variabile misurata in senso operativo è dunque la presenza del ciclone e la posizione del suo centro, ancorate al sistema di riferimento delle immagini AirmassRGB, con griglie lat/lon precomputate per la trasformazione tra domini.

## 4. Descrizione matematica completa degli algoritmi, equazioni
Il tracking è formulato come un problema di regressione con perdita MSE tra coordinate predette e target in pixel, coerente con un obiettivo continuo di minimizzazione dell’errore spaziale. La valutazione della classificazione si basa su metriche per eventi rari derivate dalla confusion matrix, con definizione di H (Hits), M (Misses), F (False Alarms) e C (Correct Negatives). Le metriche includono POD/Recall, FAR, CSI, HSS e Balanced Accuracy, con formule standard: 

POD = TP / (TP + FN) = H / (H + M)

FAR = FP / (TP + FP) = F / (H + F)

CSI = TP / (TP + FP + FN) = H / (H + F + M)

HSS = 2(TP·TN − FN·FP) / [(TP + FN)(FN + TN) + (TP + FP)(FP + TN)] = 2(H·C − M·F) / [(H+M)(M+C) + (H+F)(F+C)]

BA = 1/2 · [TP/(TP + FN) + TN/(TN + FP)] = 1/2 · [H/(H+M) + C/(C+F)]

Le conversioni pixel→lat/lon e lat/lon→km seguono un percorso coerente basato su una griglia georeferenziata; l’asse verticale viene ribaltato per allineare il sistema di riferimento delle immagini con quello geografico. La distanza geodetica è calcolata con la formula dell’haversine su sfera di raggio medio 6371.0088 km, applicando la sequenza standard: conversione in radianti, calcolo di dlat e dlon, termine a = sin²(dlat/2) + cos(lat1)·cos(lat2)·sin²(dlon/2), angolo centrale c = 2·arcsin(√a), e distanza finale pari a r·c. La documentazione tecnica del modulo non fornisce invece equazioni esplicite per la loss di classificazione e per la loss di ricostruzione MAE, pur descrivendo operativamente il mascheramento e la ricostruzione di patch.

## 5. Assunzioni teoriche e loro giustificazione, semplificazioni esplicite
Il tracking è addestrato esclusivamente su tile positive, assumendo che la stima del centro sia significativa solo quando il ciclone è presente. Il target di regressione è fissato sull’ultimo frame della sequenza, scelta che semplifica la dinamica temporale interna alla clip ma che richiede cautela interpretativa in termini di evoluzione del ciclone. La coerenza temporale tra immagini AirmassRGB e tracce Manos è assunta come garantita; il flusso presuppone inoltre la disponibilità completa dei file su disco. Le conversioni geografiche assumono una risoluzione fissa (1290×420) e una trasformazione coerente con il preprocessing; un’inversione dell’asse Y è applicata sistematicamente per mantenere coerenza tra coordinate immagine e coordinate geografiche. Alcuni comportamenti di training sono fissati a priori, come la disattivazione di mixup nella classificazione e la presenza di una loss pesata commentata; il salvataggio del “best checkpoint” in classificazione e tracking è posticipato a una soglia di epoche per evitare selezioni premature. Rimangono non giustificate in modo esplicito, nei materiali di riferimento, la scelta dell’ultimo frame come target e l’impatto fisico della dimensione delle tile.

## 6. Dati di Input
Le sorgenti dati includono sequenze AirmassRGB, organizzate in tile, tipicamente composte da 16 frame, e le tracce Manos (`TRACKS_CL*.dat`) consolidate su diverse classi (CL2–CL10). Le tracce vengono unificate in un dataframe cumulativo con gestione di identificativi univoci, preservazione della colonna di pressione, selezioni spaziali nel dominio mediterraneo e statistiche di durata; le coordinate lat/lon sono convertite in pixel mediante griglie georeferenziate e ricerca del pixel più vicino, con inversione dell’asse verticale. Sono prodotti subset specifici (CL7, CL10) e subset focalizzati sui medicanes, integrando anche tracce ERA5 per associare nomi noti; vengono inoltre aggiornate finestre temporali tramite un file di limiti rivisti, producendo versioni con intervalli più aderenti alle fasi in cui il ciclone appare ben sviluppato e con rotazione evidente.

I formati di input includono CSV per pretraining e supervisionato. Nel pretraining, le liste dati supportano sia percorsi a video sia cartelle di frame; i record indicano tipicamente il path e indici di start/numero frame, con una variante per training distribuito che aggiunge un campo extra. Nel supervised, i CSV descrivono per ogni video la lista dei frame, gli offset della tile, la label, le coordinate del centro e metadati temporali. Il processo di costruzione del dataset supervisionato prevede la generazione di un master dataframe per ciascun frame, il calcolo degli offset di tile in funzione dello stride, la creazione di sequenze temporali da 16 frame, il bilanciamento opzionale tra esempi positivi e negativi e la produzione di split train/val/test. I principali output includono file come `train_960_UNsupervised.csv`, `all_data_CL7_tracks_complete_fast.csv`, `train_supervised.csv`, `val_supervised.csv`, `test_supervised.csv`, oltre a dataset intermedi e a dataset annuali completi ottenuti senza limitare il dominio ai soli intervalli temporali attorno ai cicloni.

Sono inclusi dataset specializzati per condizioni di nuvolosità: un indice di cloud cover viene calcolato su singoli frame e aggregato per video, con soglie che separano dataset “cloudy” e “clear-sky”. Esempi di output includono `cloudy_train_853.csv` e `cloudy_test_351.csv`, oltre a dataset specifici per eventi come Juliette.

## 7. Logica dell’Algoritmo (workflow, architettura, pseudocodice)
Il workflow complessivo si articola in quattro blocchi principali: costruzione dataset, pretraining/specialization, fine-tuning supervisionato (classificazione e tracking) e inferenza con post-processing. La struttura architetturale è centrata su un backbone tipo Vision Transformer per video, con testate differenti per classificazione o regressione; nel tracking la testata di classificazione viene rimossa e sostituita da una proiezione lineare su due coordinate, preceduta da normalizzazione. La versione di pretraining impiega un decoder con mascheramento aggiuntivo rispetto a varianti precedenti di VideoMAE, secondo il paradigma “dual masking”.

### Pseudocodice sintetico (pretraining/specialization)
```
INPUT: sequenze video non etichettate, parametri di mascheramento, checkpoint iniziale
SETUP: inizializza risorse distribuite e logging, scala iperparametri per world size
MODEL: carica modello MAE, prepara decoder e maschere
DATA: costruisci dataloader con patch size coerente con il modello
FOR ogni epoca:
    aggiorna scheduler (learning rate, weight decay)
    esegui training batch-wise con mascheramento e ricostruzione
    salva checkpoint secondo frequenza stabilita
    se previsto, esegui test periodico e logga metriche
OUTPUT: checkpoint periodici, log di training e test
```

### Pseudocodice sintetico (classificazione)
```
INPUT: dataset supervisionato (train/val/test), checkpoint iniziale opzionale
SETUP: inizializza risorse e logging, scala iperparametri
MODEL: backbone video con testata di classificazione binaria
FOR ogni epoca:
    esegui training su train set
    a intervalli definiti, esegui validazione su val/test
    aggiorna best checkpoint se la metrica migliora dopo una soglia di epoche
OUTPUT: log JSON lines, checkpoint-best, metriche per epoca
```

### Pseudocodice sintetico (tracking)
```
INPUT: dataset con sole tile positive e coordinate target (x, y)
SETUP: inizializza risorse e logging
MODEL: backbone video + head di regressione a 2 dimensioni
FOR ogni epoca:
    esegui training con perdita MSE
    valuta su test/val e aggiorna best checkpoint
OUTPUT: checkpoint tracking best, log per epoca
```

### Pseudocodice sintetico (inferenza classificazione)
```
INPUT: dataset di validazione/test, checkpoint di classificazione
SETUP: modalità inferenza, eventuale distribuzione multi-rank
MODEL: carica modello e checkpoint
IF modalità logits o embedding:
    raccogli output in shard NPZ
    esegui merge e cleanup
ELSE:
    raccogli predizioni standard
OUTPUT: CSV predizioni, metriche inferenza
```

### Workflow di costruzione dataset
Il flusso di costruzione dataset inizia dalla lettura delle immagini e dalla creazione del master dataframe con coordinate, offset e label. Le sequenze sono poi segmentate in gruppi temporali continui, raggruppate per offset di tile e trasformate in clip video da 16 frame; i clip sono salvati in cartelle dedicate e i relativi metadati vengono aggregati in CSV finali. Nei flussi supervisionati, la procedura include bilanciamento opzionale, split temporali, e versioni specifiche per classi di cicloni o per medicanes con finestre temporali aggiornate. Sono disponibili flussi aggiuntivi per dataset annuali completi, per dataset “cloudy” e per relabeling basati sulla distanza temporale dal ciclone.

### Workflow di inferenza e post-processing su scala Mediterranea
Le predizioni su sequenze video possono essere espanse al livello frame, unendo i risultati al master dataframe per recuperare coordinate geografiche e metadati. In caso di tile mancanti, un’espansione basata su offset teorici ricostruisce la griglia completa, marcando le tile mancanti come riempite e colorate in grigio. Le sequenze risultanti possono essere convertite in animazioni MP4 con overlay di predizioni e timestamp, inclusi strumenti per generare GIF o video su cicloni specifici.

## 8. Dettagli implementativi e computazionali
Il sistema supporta l’esecuzione distribuita multi-GPU e multi-nodo, con riconoscimento automatico del contesto di lancio in base a variabili d’ambiente di rank globale/locale e world size. È prevista la scalatura di batch size e learning rate in funzione del numero totale di processi, con inizializzazione del gruppo di comunicazione e sincronizzazione dei gradienti. Nei flussi di training è applicata una schedulazione coseno per learning rate e weight decay, con aggiornamento step-level; nel tracking, una correzione esplicita assicura che i valori schedulati siano applicati a ogni batch. È presente una logica di decadimento “layer-wise” che modula il learning rate per profondità del trasformatore, con possibilità di impostare profili uniformi o personalizzati.

Il pretraining è documentato con uno script di esempio per 64 GPU (8 nodi × 8 GPU), con configurazione di porta master dinamica, numero di thread controllato, e parametri di training dettagliati: mascheramento tipo “tube” con rapporto 0.9, mascheramento decoder “run_cell” con rapporto 0.5, modello “giant” con patch 14 e input 224, profondità decoder 4, batch size 32, sampling rate 4, 4 campioni per clip, 10 worker, ottimizzatore AdamW con lr 6e-4, clip dei gradienti 0.02, betas 0.9/0.95, warmup 30 epoche, salvataggio checkpoint ogni 5 epoche e durata 300 epoche. È fornita anche una variante per training distribuito non Slurm, con master_port esplicito, specifica di nnodes, node_rank e master_addr, e formato esteso della lista dati con un campo aggiuntivo.

Nelle configurazioni Slurm per classificazione, un esempio di job definisce 4 nodi, 4 task per nodo, 4 GPU per nodo, 4 CPU per task, partizione boost_usr_prod, tempo 17:58:00, e usa un avvio multi-processo con mappatura per socket. Il master address è derivato dal nodo principale e la porta è fissata a 12340; sono previsti log di output e di errore dedicati. La documentazione include inoltre una ricetta generale per replicare lo stesso schema in altri repository, con indicazioni su detection del contesto, impostazione del device, inizializzazione del processo distribuito, wrapping del modello, scalatura degli iperparametri e script di lancio.

L’inferenza distribuita produce file intermedi in formato NPZ con schema di naming per rank e batch, seguiti da merge e pulizia; le metriche di inferenza sono salvate in un file dedicato. I log di training sono registrati in formato JSON lines in `log.txt`, con checkpoint specifici come `checkpoint-best.pth` o `checkpoint-tracking-best.pth`.

## 9. Parametri utilizzati e loro calibrazione
I documenti riportano parametri dettagliati per il pretraining e il training, ma non indicano una procedura formale di calibrazione. Oltre ai parametri di mascheramento e decoder già citati, la pipeline di classificazione include la possibilità di pesare le classi in base alla distribuzione del training set, sebbene l’uso effettivo della loss pesata risulti disabilitato. La schedulazione coseno è adottata per learning rate e weight decay, e la scelta del best checkpoint è ritardata di un numero di epoche per evitare selezioni premature.

Nel dominio dei dataset, sono presenti soglie esplicite per il cloud index (ad esempio 0.2), oltre a scelte strutturali come finestre temporali da 16 frame e stride specifici per la griglia di tile. Per il tracking su scala mediterranea sono riportati stride pari a 213 in x e 196 in y, usati per enumerare tutte le tile teoriche e riempire quelle mancanti. È inoltre documentata l’adozione di finestre temporali aggiornate per i medicanes, con filtri basati su distanza temporale dal centro del ciclone (ad esempio 12 ore) che producono relabeling e rimozione delle righe fuori soglia.

## 10. Fase di addestramento dell’algoritmo
La specialization prevede un fine-tuning auto-supervisionato con logging su dashboard interattive, salvataggio periodico di checkpoint e test a intervalli stabiliti. La classificazione supervisionata include training per epoche, validazione periodica con salvataggio del best checkpoint e logging in formato JSON lines; il tracking segue un ciclo analogo con perdita MSE e aggiornamento del best modello in base alla loss di validazione. I flussi sono progettati per scalare su più GPU e nodi, con inizializzazione distribuita e controllo delle risorse. La documentazione segnala che l’entry point di specialization richiede argomenti espliciti e può fallire se lanciato senza parametri; la necessità di un parser CLI completo è indicata come correzione da implementare. Non sono definiti criteri formali di early stopping o soglie quantitative di arresto.

## 11. Procedure di validazione e test
La validazione combina metriche quantitative e valutazioni qualitative. Sul piano quantitativo, i notebook di analisi costruiscono set di validazione a partire dal master dataset, applicano filtri temporali (ad esempio esclusione di frame oltre 12 ore dal ciclone), calcolano indici di nuvolosità, generano dataset “cloudy” e valutano le prestazioni su dataset bilanciati e sbilanciati, includendo casi di anno intero. Le metriche includono accuracy, FPR, FNR, POD, FAR e confusion matrix, con estensioni per valutazioni per label temporali e grafici di andamento delle metriche nel tempo.

Sul piano qualitativo, le predizioni vengono mappate dal livello video al livello frame e visualizzate in animazioni del Mediterraneo con overlay delle predizioni, timestamp e tile. Sono prodotti GIF e MP4 per cicloni specifici, con utility di rendering che supportano sia esecuzione seriale sia parallelizzazione su CPU. I notebook includono strumenti di QA visivo per il tracking, con disegno del centro ciclone sulle tile e ispezione di sequenze animate.

## 12. Limitazioni note dell’algoritmo
Le limitazioni note includono l’assenza di un parser CLI completo nell’entry point di specialization, la disattivazione forzata di mixup nella classificazione, la loss pesata non attiva e l’uso di flag mutuamente esclusivi nella costruzione dataset, che impediscono combinazioni di pipeline. Le conversioni geografiche sono vincolate alla risoluzione 1290×420 e richiedono rigenerazione delle griglie in caso di cambiamento delle dimensioni. Alcune sezioni dei notebook riportano TODO o incertezze operative, come la scelta del canale colore più adatto per la stima della nuvolosità. Non sono presenti analisi formali di generalizzazione su domini nuovi né quantificazioni di performance su casi limite o scenari estremi.

## 13. Analisi degli Errori e Incertezze
L’analisi degli errori si basa su metriche per eventi rari e su tassi di falsi allarmi e mancate detection. Il tracking include una conversione diretta dell’errore in km, utile per interpretazioni geofisiche. Sono disponibili strumenti per analisi per label temporali e per il confronto tra run diverse, con grafici di loss, accuracy e FPR/FNR. Non sono documentate analisi di incertezza statistica come intervalli di confidenza o varianza tra run, né metodi di stima dell’incertezza epistemica o aleatoria.

## 14. Casi particolari / screening / filtraggi / case study
Sono previsti flussi di screening basati su cloud index, con soglie che separano dataset “cloudy” e “clear-sky” e con strumenti di visualizzazione per verificare il comportamento delle maschere su singoli frame e su clip. Il relabeling temporale consente di escludere o rietichettare frame lontani dall’intervallo di validità del ciclone, producendo dataset più puliti per l’addestramento. I casi studio includono la generazione di animazioni per cicloni nominati (es. Apollo, Blas, Daniel, Helios, Juliette) e la visualizzazione di predizioni su scala mediterranea per validation e tracking.

Per il tracking su scala Mediterraneo, il workflow prevede la ricostruzione delle sequenze, il merge con le predizioni di tracking, l’espansione a livello frame, la gestione delle tile mancanti con flag di riempimento grigio, l’inserimento della traccia GT e PRED in ordine coerente, e la generazione di un MP4 con framerate 10, codec H.264, CRF 18 e pixel format yuv420p. Il processo produce cartelle di frame PNG, un file di elenco frame con durata fissa e un video finale, con possibilità di ripetere la conversione se il file non viene generato. È anche prevista la forzatura del colore delle predizioni (rosso) per distinguerle dalle tracce di riferimento.

## 15. Riferimenti a baseline di codice
Le baseline operative sono rappresentate dagli entry point di training e inference (specialization, classificazione, tracking, inferenza classificazione), dalle pipeline di costruzione dataset e relabeling, e dai moduli di utilità per la gestione di risorse distribuite, conversioni geografiche e metriche. Sono disponibili diagrammi di call tree per training, classification, tracking e specialization, che offrono una vista sintetica delle dipendenze tra script; tali diagrammi sono indicati come ancora in via di miglioramento. Il repository esplicita inoltre che l’ambiente è orientato alla ricerca, senza framework di test o linting formalizzati, privilegiando flessibilità e sperimentazione.

## Riassunto finale
Il documento integra tutte le informazioni tecniche presenti nei materiali di riferimento, riformulandole in termini algoritmici e scientifici senza riferimento esplicito a classi, funzioni o librerie specifiche. Le sezioni più dense riguardano dataset building, workflow di training/inference, conversioni geografiche, metriche di valutazione e procedure di visualizzazione. Le principali aree ancora parziali sono quelle non specificate nei documenti originari: formalizzazione della loss di classificazione e della loss MAE, criteri di calibrazione degli iperparametri, stime quantitative dei costi computazionali e analisi di incertezza statistica. Se utile, è possibile estendere il documento con un glossario di variabili e con una sezione di requisiti operativi (risorse minime, tempi di training, capacità di storage) una volta disponibili dati aggiuntivi.
