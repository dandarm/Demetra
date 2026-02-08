Agisci come un Senior Technical Writer esperto in documentazione scientifica e spaziale (standard ESA/NASA).
Analizza i file doc del repo (.md) ed estrai tutte le informazioni tecniche presenti. Lo scopo è estrarre tutte le informazioni utili per compilare un documento ATBD (Algorithm Theoretical Basis Document), che dovrà avere grosso modo questa struttura:


STRUTTURA SUGGERITA (modifica liberamente se migliora chiarezza ATBD)

1) Scopo, contesto
    Summary 
    Introduzione, Scopo e contesto del prodotto 
    (perché esiste il prodotto, a chi serve, in quali condizioni è “valido”)

2) Panoramica prodotto, definizioni e specifiche 
   - target variable (cosa si stimi: classe, heatmap, coordinate, ecc.)
   - Data input specs, Coverage, resolution, cadence (quali dati minimi devono esistere per produrre l’output)
   - Output variables, units, valid ranges, griglia/risoluzione, convenzioni (coordinate, timestamp)
   - Quality flags / confidence indicators (se presenti)

3) Input data & pre-processing - tutti gli step di lavorazione dei dati
   - preparazione, labeling, split

4) Algoritmo 
   overview 
   logica e pipeline completa + diagramma flowchart + pseudocodice
   formule-assunzioni/semplificazioni locali e giustificazioni; il resto rimando alla provenienza del modello open source videomae)
   - Dettagli computazionali

5) Fase di Training 
   dataset/split/loss.. parametri e iperparametri chiave + default, calibrazioni e scelte; riproducibilità.
     (doc/README)
   Dettagli implementativi:
      BatchNorm vs GroupNorm / FreezeBN
      AMP (fp16/bf16), gradient scaling
      scheduler, warmup, clipping, EMA
      DDP/multi-GPU e implicazioni (SyncBN, seed)
      strategie anti-OOM: checkpointing, crop/tiling
      determinismo e riproducibilità (seed, cuDNN flags) per training

6) Verifica: validazione, test, incertezze, limiti
   protocollo e metriche 
   risultati (rimando ai risultati/slide/report)
   Analisi errori, incertezze, limitazioni e failure modes 
   case study come dimostrazione qualitativa

7) Provenance (obbligatoria, compatta):
    - repo upstream + licenza 
    - elenco modifiche locali (5–12 bullet o tabella breve “upstream vs ours”)
    - entrypoint, layout del repo, config, struttura cartelle




11. References bibliografiche


Per adesso usiamo solo il modulo videomae.

Estrai tutte le informazioni tecniche presenti, ma evita i dettagli strettamente inerenti python, pytorch, e le varie altre librerie usate, descrivendo tutti i processi senza riferimenti espliciti a classi e funzioni.

Ovviamente non tutte queste sezioni possono essere riempite uniformemente dalle informazioni contenute nel repository, che riguarda il modello e il trattamente dei dati principalmente. Quindi concentrati sulle sezioni che dovrebbero avere il maggior overlap con il contenuto dei docs.
Esponi quindi un logica dettagliata dell'algoritmo/modello (Workflow, architettura , pseudocodice, passi logici) e non evitare nessuna parte presente nei docs .md di riferimento

Regola anti-ridondanza
Se un dettaglio “computazionale” serve a far convergere o cambia le metriche, sta in (5).
Se un dettaglio “computazionale” riguarda solo performance (es. quante immagini/s) ma non cambia risultati, 
o lo ometti oppure lo metti come 2 righe finali in (5) sotto “Practical notes”.
Se è “dove sta il codice / quale commit”, sta solo in (7).

Crea un nuovo documento con il contenuto, più un altro documento che riprende tutti i docs e riporta escplicitamente sezione per sezione quale è stata usata e dove, e soprattutto quale è stata evitata perché non utile per lo scopo di questo ATBD.