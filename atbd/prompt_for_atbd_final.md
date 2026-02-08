Agisci come un Senior Technical Writer esperto in documentazione scientifica e spaziale (standard ESA/NASA).
Analizza i file doc .md (slides, TR_v2, e le referenze bibliografiche AI_4_medicanes e annex_videomaev2) ed estrai tutte le informazioni tecniche presenti. Lo scopo è estrarre tutte le informazioni utili per compilare un documento ATBD (Algorithm Theoretical Basis Document), che dovrà avere grosso modo questa struttura:


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


8) References bibliografiche



Evita elenchi puntati e numerati, espandi il testop in modo discorsivo, la spiegazione deve fluire come in un contesto scientifico, con frasi organizzate con soggetti e verbi.  Espandi pure le nozioni puntuali con le tue conoscenze se nei doc sono date per scontate le motivazioni e le descrizioni dettagliate, per illustrare anche a un pubblico esperto non di machine learning seppur esperto di meteorologia ed eventi rari, quali sono gli scienziati dell'ESA.

Ovviamente non tutte queste sezioni possono essere riempite uniformemente e allo stesso modo dalle informazioni contenute nei docs. Quindi concentrati sulle sezioni che dovrebbero avere il maggior overlap con il contenuto dei docs, e lascia pure meno popolate di testo le sezioni meno sovrapponibili col contenuto dei docs.
Esponi quindi un logica dettagliata delle procedure(Workflow, architettura, passi logici) e non evitare nessuna parte presente nei docs .md di riferimento


Crea un nuovo documento con il contenuto, più un altro documento che riprende tutti i docs e riporta escplicitamente sezione per sezione quale è stata usata e dove, e soprattutto nell'eventualità che qualcosa sia stata evitata, cita quale e perché non utile per lo scopo di questo ATBD.