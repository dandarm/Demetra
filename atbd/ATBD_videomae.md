# ATBD - VideoMAE per Medicanes (modulo videomae)

## 1. Scopo e contesto

### Summary
Il modulo VideoMAE adatta VideoMAE v2 allo studio dei medicanes su sequenze video satellitari AirmassRGB (SEVIRI MSG). Il sistema integra un pretraining auto-supervisionato di tipo masked autoencoding su video e due fasi supervisionate orientate rispettivamente alla classificazione cyclone/no-cyclone e al tracking del centro ciclone come regressione di coordinate. L'architettura viene quindi impiegata lungo un flusso coerente che copre la costruzione dei dataset, il training distribuito, l'inferenza e una verifica che combina metriche quantitative e ispezioni qualitative su scala mediterranea.

### Introduzione, scopo e contesto del prodotto
Lo scopo del prodotto e' fornire una base algoritmica solida per rilevare e localizzare i cicloni mediterranei a partire da sequenze satellitari, mantenendo coerenza con le pratiche di analisi di eventi rari in meteorologia. Il contesto operativo e' definito da immagini AirmassRGB compatibili con la griglia geografica adottata (1290x420) e con la logica di tiling usata nella costruzione dei dataset; entro questi vincoli, il sistema permette di derivare un output interpretabile sia come decisione binaria di presenza del ciclone sia come stima spaziale del centro. Il pubblico di riferimento include ricercatori e operatori con competenze meteorologiche che necessitano di strumenti riproducibili per detection e tracking, affiancati da visualizzazioni e metriche piu' adatte a dataset sbilanciati.

## 2. Panoramica prodotto, definizioni e specifiche

### Target variable
Il prodotto fornisce tre livelli di output, ciascuno legato a un obiettivo differente ma integrato nella stessa pipeline. Nel pretraining, il target e' la ricostruzione di patch mascherate, un compito auto-supervisionato che spinge il modello a catturare pattern spaziotemporali coerenti. Nella classificazione supervisionata, il target e' una label binaria cyclone/no-cyclone associata a ogni clip video. Nel tracking, il target e' una coppia di coordinate (x, y) che rappresenta il centro del ciclone nell'ultimo frame della sequenza e che consente una stima continua della posizione.

### Input data specs, coverage, resolution, cadence
Gli input principali sono immagini satellitari AirmassRGB da SEVIRI MSG, con copertura Mediterranea e griglia lat/lon precomputata per immagini 1290x420. Le immagini vengono suddivise in tile di 224x224 con offset determinati da stride che, nelle visualizzazioni a scala mediterranea, assumono valori tipici di 213 in x e 196 in y. Le sequenze di lavoro sono clip da 16 frame; nel tracking ogni clip rappresenta una finestra di circa 80 minuti con ultimo frame allineato all'ora piena, in modo da rendere coerente il legame tra sequenza e target spaziale. I formati di input sono CSV con righe che puntano ai video o alle cartelle di frame; per il pretraining i record seguono la sintassi `video_path, 0, -1` oppure `frame_folder_path, start_index, total_frames`, con una variante distribuita che aggiunge un campo extra, mentre nel supervisionato il CSV riporta la lista dei frame, gli offset di tile, la label, le coordinate del centro e i metadati temporali.

### Output variables, units, valid ranges, griglia/risoluzione, convenzioni
L'output di classificazione e' una label binaria a livello clip, a cui possono essere associati punteggi continui per analisi successive. L'output di tracking e' una stima (x, y) in pixel relativi alla tile, convertibile in lat/lon tramite la griglia georeferenziata e in distanza geodetica in km mediante la formula dell'haversine. Le convenzioni garantiscono la coerenza tra dominio immagine e dominio geografico: l'asse Y viene invertito per allineare la rappresentazione pixel alla latitudine, e le coordinate globali si ottengono sommando gli offset di tile alle coordinate relative prima della conversione geografica.

### Quality flags / confidence indicators
Il principale indicatore di qualita' documentato e' l'indice medio di nuvolosita' per video, usato per separare dataset cloudy e clear-sky con soglie tipiche superiori a 0.2. In fase di visualizzazione mediterranea e di post-processing, le tile mancanti vengono espanse e marcate con flag di riempimento per distinguere i dati reali dalle ricostruzioni e per evitare interpretazioni spurie nelle animazioni o nei mosaici finali.

## 3. Input data & pre-processing

### Dati di input
I dati di input comprendono le sequenze AirmassRGB e le tracce Manos `TRACKS_CL*.dat` (classi CL2-CL10) con coordinate geografiche e pressione, integrate quando necessario con tracce ERA5 per associare nomi noti ai medicanes. Sono inoltre presenti file di supporto per l'aggiornamento delle finestre temporali, come `new_cyc_limits.csv` e `more_medicanes_time_updated.csv`, che definiscono intervalli piu' coerenti con le fasi di rotazione evidente dei cicloni.

### Pre-processing e labeling
Il pre-processing inizia con l'unificazione delle tracce di ciclone provenienti da classi differenti, la normalizzazione degli identificativi e la conservazione di variabili come la pressione. Il dominio viene poi ristretto al Mediterraneo tramite selezioni spaziali, e la durata dei cicloni e' analizzata per individuare subset piu' affidabili. Le coordinate lat/lon vengono trasformate in pixel usando una griglia georeferenziata coerente con la risoluzione 1290x420; questa fase include il ribaltamento dell'asse Y per mantenere la consistenza tra sistema geografico e immagine, e produce CSV con coordinate `x_pix` e `y_pix`.

A partire da queste tracce si costruisce un master dataframe supervisionato che, per ogni frame, include path dell'immagine, offset della tile, coordinate del centro e label. Il master viene poi segmentato in gruppi temporali contigui, raggruppato per offset di tile e trasformato in clip da 16 frame, che vengono salvati su disco insieme ai CSV finali di train, val e test. Il flusso ammette strategie di bilanciamento tra esempi positivi e negativi, split temporali e split per identificativi di ciclone, inclusi dataset specifici per CL10 o per medicanes nominati.

Un passaggio importante e' il relabeling temporale, che ricalcola la label in funzione della distanza dall'intervallo di validita' del ciclone, ad esempio escludendo i frame oltre 12 ore dal centro. Questa logica produce dataset piu' coerenti con l'obiettivo meteorologico di catturare la fase attiva del ciclone. Sono inoltre presenti pipeline per la generazione di dataset full-year che includono periodi senza cicloni, cosi da valutare il comportamento su distribuzioni realistiche e altamente sbilanciate.

L'indice di nuvolosita' viene calcolato sia a livello frame sia a livello video e consente la creazione di dataset cloudy e clear-sky. Questa informazione e' utile per stratificare la validazione e per verificare se le prestazioni degradano in condizioni di copertura nuvolosa elevata. La generazione dei dataset e' inoltre organizzata da un dispatcher che seleziona una sola pipeline per esecuzione, con modalita' dedicate a dataset supervisionati, relabeling, cloud index, tracking o full-year; questa scelta rende il flusso piu' chiaro ma limita la combinazione di opzioni in una singola run. Infine, per il tracking si costruiscono dataset dedicati che includono solo esempi positivi e associano a ogni clip le coordinate del centro sul frame finale.

## 4. Algoritmo

### Overview
Il nucleo dell'algoritmo e' un backbone video basato su trasformatori, inizialmente specializzato con masked autoencoding per apprendere rappresentazioni spaziotemporali robuste in assenza di etichette. Il masked autoencoding introduce un vincolo di ricostruzione di patch mascherate che obbliga il modello a inferire il contenuto mancante dalle strutture contestuali, un meccanismo utile in meteorologia per catturare pattern dinamici e graduali. La base specializzata viene poi fine-tunata per due compiti distinti: la classificazione binaria e il tracking regressivo del centro, ottenuto sostituendo la head di classificazione con un regressore a due dimensioni.

### Flowchart (testuale)
```
AirmassRGB frames + Manos tracks
  -> costruzione master dataframe (offset, label, x/y pixel)
  -> clip video (16 frame) + CSV train/val/test
     -> (A) pretraining MAE su clip non etichettati
     -> (B) fine-tuning classificazione su clip etichettati
     -> (C) fine-tuning tracking (solo tile positive)
  -> inferenza + post-processing
  -> validazione quantitativa + visualizzazioni
```

### Pseudocodice (sintetico)
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

### Formule, assunzioni, semplificazioni
Il tracking minimizza una loss MSE tra coordinate predette e target (x, y), coerente con l'obiettivo di ridurre l'errore spaziale continuo. Le metriche di classificazione sono quelle tipiche dei problemi di eventi rari in meteorologia, per cui la sola accuracy e' fuorviante in presenza di forte sbilanciamento. Le formule usate si basano sulla confusion matrix con H (hits), M (misses), F (false alarms) e C (correct negatives) e includono:

POD = H / (H + M)
FAR = F / (H + F)
CSI = H / (H + F + M)
HSS = 2(H*C - M*F) / [(H+M)(M+C) + (H+F)(F+C)]
BA = 0.5 * [H/(H+M) + C/(C+F)]

Per il tracking, l'errore in km e' derivato passando dalle coordinate pixel a lat/lon tramite la griglia georeferenziata e applicando la formula dell'haversine con raggio medio terrestre 6371.0088 km. Le assunzioni principali includono l'uso di soli esempi positivi per il tracking e la scelta del frame finale come target, una semplificazione temporale che rende il problema ben definito rispetto alla finestra di osservazione. Un'altra assunzione fondamentale e' la validita' della griglia lat/lon per immagini 1290x420; se la risoluzione cambia, la griglia deve essere rigenerata per mantenere la coerenza delle conversioni.

Le motivazioni operative di queste scelte sono legate alla disponibilita' delle etichette e alla necessita' di un target stabile nel tempo. In un contesto meteorologico, fissare il target al frame finale consente di confrontare in modo riproducibile la posizione stimata con l'osservazione piu' recente, mentre l'uso di soli casi positivi evita di introdurre rumore nei target quando il ciclone non e' presente. Le equazioni della loss di classificazione e della loss MAE non sono esplicitate nei documenti del modulo e restano quindi non formalizzate in questo ATBD.

## 5. Fase di training

Il training segue due percorsi complementari. Nel pretraining, il modello impara a ricostruire patch mascherate su clip non etichettati, con l'obiettivo di specializzarsi sulle dinamiche dei medicanes senza imporre vincoli di classe. Nel supervisionato, la classificazione ottimizza la separazione cyclone/no-cyclone, mentre il tracking ottimizza l'errore MSE sulle coordinate del centro; in entrambi i casi vengono creati set di train, validation e test, con possibilita' di bilanciamento o di split per identita' di ciclone. La loss di classificazione non e' specificata nei documenti e va considerata non documentata.

I principali iperparametri documentati riguardano il pretraining: mascheramento con `mask_ratio=0.9` e `decoder_mask_ratio=0.5`, decoder depth 4, patch 14 e input 224, clip di 16 frame con sampling rate 4 e num_sample 4, batch size 32, learning rate 6e-4, warmup di 30 epoche, gradient clipping a 0.02, training per 300 epoche e checkpoint ogni 5 epoche. Sul versante dei dataset, la soglia tipica dell'indice di nuvolosita' e' 0.2, mentre gli stride per ricostruire il mosaico mediterraneo sono 213 in x e 196 in y. Questi parametri sono riportati come esempi operativi piu' che come configurazioni ottimali.

I dettagli computazionali che influenzano la convergenza includono la scalatura di batch size e learning rate con il world size nel training distribuito, l'uso di scheduler coseno per learning rate e weight decay, l'applicazione di una modulazione layer-wise del learning rate e l'uso di mixed precision con gradient scaling. Il checkpoint del best modello viene ritardato di un numero di epoche per evitare selezioni premature, e le procedure di impostazione dei seed sono impiegate per aumentare la riproducibilita' tra run. La presenza di log in JSON lines e di checkpoint periodici permette di ricostruire l'andamento e di riprendere il training in caso di interruzione.

Practical notes relative alle prestazioni includono esempi di training distribuito su cluster multi-nodo con configurazioni dell'ordine di decine di GPU e l'uso di pipeline di rendering video per le visualizzazioni finali. Questi aspetti migliorano la scalabilita' ma non modificano i risultati scientifici.

## 6. Verifica: validazione, test, incertezze, limiti

La verifica quantitativa combina la confusion matrix con metriche specifiche per eventi rari, in modo da ridurre l'impatto della forte asimmetria tra casi positivi e negativi. Le metriche adottate sono POD, FAR, CSI, HSS e Balanced Accuracy, con formule esplicitate nella sezione precedente; esse permettono di valutare simultaneamente la capacita' di rilevare gli eventi, il tasso di falsi allarmi e la performance rispetto a un baseline casuale. Per il tracking, l'errore e' espresso anche in km tramite conversione geodetica, cosi da fornire una misura fisicamente interpretabile.

Il workflow di validazione prevede la costruzione di set di test e validation attraverso filtri temporali che escludono frame troppo lontani dal ciclone, e la generazione di dataset bilanciati, sbilanciati o annuali completi. Le analisi includono il calcolo di accuracy, FPR, FNR, POD, FAR e confusion matrix, oltre a valutazioni per label temporali che mettono in relazione la performance con la distanza dall'evento. I log storici vengono inoltre impiegati per confrontare esperimenti diversi e analizzare la sensibilita' agli iperparametri.

La verifica qualitativa comprende la generazione di animazioni mediterranee con overlay delle predizioni, la visualizzazione delle tile e delle traiettorie del centro ciclone, e il rendering di GIF o MP4 per cicloni specifici. Nel caso del pretraining, la qualita' delle ricostruzioni viene verificata tramite confronti visivi tra input e output e tramite l'analisi delle maschere di patching. Queste procedure non sostituiscono la validazione quantitativa, ma forniscono un controllo visivo della coerenza spaziale e temporale delle predizioni.

Le principali limitazioni documentate riguardano l'assenza di criteri formali di early stopping, la mancanza di analisi di incertezza statistica e l'assenza di soglie di accettazione quantitative. Inoltre, alcune scelte sono hard-coded, come la disattivazione del mixup in classificazione e la non attivazione di una loss pesata, mentre le pipeline di costruzione dataset utilizzano flag mutuamente esclusivi che impediscono combinazioni di flussi. Le conversioni geografiche restano valide solo per la griglia 1290x420 e l'indice di nuvolosita' si basa su scelte di canale colore che richiedono ulteriori verifiche. L'entry point della specialization necessita di argomenti espliciti e non risulta avviabile in modo standalone senza una corretta configurazione dei parametri.

## 7. Provenance (obbligatoria, compatta)

### Upstream e licenza
Il modulo deriva da VideoMAE v2 e mantiene riferimenti concettuali a VideoMAE v1. La licenza upstream non e' riportata nei documenti del modulo e va verificata nel repository di origine.

### Modifiche locali (upstream vs ours)
L'adattamento locale introduce un dominio applicativo specifico ai medicanes su immagini AirmassRGB, una pipeline di costruzione dataset con tiling e clip da 16 frame, e l'integrazione delle tracce Manos con aggiornamento delle finestre temporali. Viene aggiunto un percorso di tracking del centro con regressione a due coordinate e una conversione sistematica degli errori in km. La pipeline include la stima della nuvolosita' per filtrare dataset cloudy e clear-sky, oltre a flussi di inferenza e post-processing per mosaici mediterranei. Sono presenti notebook per validazione quantitativa, analisi temporale e visualizzazioni di case study, insieme a un supporto esplicito al training distribuito e alla scalatura degli iperparametri. La fase di pretraining e' accompagnata da strumenti di verifica delle maschere e delle ricostruzioni.

### Entrypoint, layout del repo, config
Gli entrypoint principali coprono pretraining, classificazione, tracking e inferenza, mentre la pipeline dati include strumenti per la costruzione dei dataset e per il tracking. Il repository e' organizzato con una cartella `docs/` per notebook e guide, `misc/` per note operative e `scripts/` per job di training, e produce output standardizzati come `log.txt` e checkpoint best. La configurazione si basa su parametri via CLI e su CSV di input per train, validation e test.

## 8. References bibliografiche
Le reference principali sono VideoMAE v1 (NeurIPS 2022, https://arxiv.org/abs/2203.12602), VideoMAE v2 (CVPR 2023, https://arxiv.org/abs/2303.16727) e il repository upstream VideoMAE v2 (https://github.com/OpenGVLab/VideoMAEv2).
