## Fonte: ATBD_DeMeTrA.md
Paragrafo 1: # ATBD DeMeTrA
Frase 1: "# ATBD DeMeTrA." -> ATBD_final: ATBD_final (sintesi)

Paragrafo 2: ## Scopo, contesto e summary Questo documento descrive la base teorica e la c...
Frase 1: "## Scopo, contesto e summary
Questo documento descrive la base teorica e la catena di produzione del prodotto DeMeTrA, un algoritmo di deep learning per il rilevamento e il tracciamento dei medicanes, cioe cicloni mediterranei con caratteristiche simil tropicali." -> ATBD_final: ATBD_final > 6. Verifica: validazione, test, incertezze, limiti (sintesi)
Frase 2: "Il contesto operativo e il monitoraggio near real time di fenomeni rari, in cui la disponibilita di immagini geostazionarie ad alta cadenza consente di seguire l'evoluzione rapida dei sistemi convettivi." -> ATBD_final: ATBD_final > 1. Scopo e contesto > Summary
Frase 3: "L'obiettivo e fornire uno strumento per individuare la presenza di cicloni, localizzare il centro di rotazione e riconoscere precocemente l'evoluzione verso sistemi tropical-like con occhio chiuso." -> ATBD_final: ATBD_final > 1. Scopo e contesto > Summary

Paragrafo 3: DeMeTrA sfrutta l'imaging infrarosso del sensore Spinning Enhanced Visible In...
Frase 1: "DeMeTrA sfrutta l'imaging infrarosso del sensore Spinning Enhanced Visible InfraRed Imager, SEVIRI, a bordo di Meteosat Second Generation, MSG." -> ATBD_final: ATBD_final > 1. Scopo e contesto > Summary
Frase 2: "Le immagini sono convertite in compositi Airmass RGB, riorganizzate in clip video e usate per specializzare un modello VideoMAEv2 pre-addestrato." -> ATBD_final: ATBD_final > 3. Input data & pre-processing (sintesi)
Frase 3: "Il processo prevede un addestramento auto-supervisionato di specializzazione e due addestramenti supervisionati, uno per la classificazione binaria presenza o assenza di ciclone e uno per la regressione della posizione del centro di rotazione." -> ATBD_final: ATBD_final > 1. Scopo e contesto > Summary
Frase 4: "La detection e preparatoria al tracking, perche consente di filtrare i clip e ridurre il carico computazionale della regressione." -> ATBD_final: ATBD_final > 1. Scopo e contesto > Summary
Frase 5: "Il prodotto e sviluppato nel quadro delle attivita WP2400 per la detection, il monitoraggio e il tracking dei medicanes con dati geostazionari e output utilizzabili in contesto operativo ESA." -> ATBD_final: ATBD_final > 1. Scopo e contesto > Summary

Paragrafo 4: ## Panoramica prodotto, definizioni e specifiche Il prodotto DeMeTrA stima du...
Frase 1: "## Panoramica prodotto, definizioni e specifiche
Il prodotto DeMeTrA stima due variabili target." -> ATBD_final: ATBD_final > 2. Panoramica prodotto, definizioni e specifiche (sintesi)
Frase 2: "La prima e un indicatore di presenza di ciclone mediterraneo in un clip video, ottenuto tramite classificazione binaria." -> ATBD_final: ATBD_final (sintesi)
Frase 3: "La seconda e la coordinata del centro di rotazione, RC, prodotta con una regressione su ciascun clip video contenente un ciclone." -> ATBD_final: ATBD_final (sintesi)
Frase 4: "La definizione operativa di ciclone segue i cataloghi TRACKS_CL7, con successive rifiniture basate su ispezione visiva e selezione di medicanes." -> ATBD_final: ATBD_final > 2. Panoramica prodotto, definizioni e specifiche > Target variable
Frase 5: "TRACKS_CL7 deriva dalla combinazione di sette metodi di tracking applicati a reanalisi, quindi rappresenta un riferimento coerente ma non esente da mismatch rispetto a segnali IR." -> ATBD_final: ATBD_final > 2. Panoramica prodotto, definizioni e specifiche > Target variable

Paragrafo 5: Gli input minimi sono immagini IR del servizio Rapid Scan Service con cadenza...
Frase 1: "Gli input minimi sono immagini IR del servizio Rapid Scan Service con cadenza di 5 minuti e risoluzione spaziale di circa 3 km al sub-satellite point." -> ATBD_final: ATBD_final > 2. Panoramica prodotto, definizioni e specifiche > Input data specs, coverage, resolution, cadence
Frase 2: "Le immagini sono fornite in formato Zarr, in proiezione geostazionaria non riproiettata." -> ATBD_final: ATBD_final > 2. Panoramica prodotto, definizioni e specifiche > Input data specs, coverage, resolution, cadence
Frase 3: "La catena necessita dei canali IR_097, IR_108, WV_062 e WV_073 per la costruzione del composito Airmass RGB." -> ATBD_final: ATBD_final > 3. Input data & pre-processing (sintesi)
Frase 4: "Le immagini vengono ritagliate sull'area mediterranea, con latitudine compresa tra 30 gradi e 48 gradi Nord e longitudine tra -7 e 46 gradi, e risultano in un frame di 1290 per 420 pixel." -> ATBD_final: ATBD_final > 3. Input data & pre-processing (sintesi)

Paragrafo 6: Fig. 5.2 – Air mass RGB example image covering the Mediterranean basin, with ...
Frase 1: "Fig." -> ATBD_final: ATBD_final > 3. Input data & pre-processing > Dati di input
Frase 2: "5.2 – Air mass RGB example image covering the Mediterranean basin, with superimposed borders." -> ATBD_final: ATBD_final > 3. Input data & pre-processing > Dati di input
Frase 3: "Image time: 17th September 2020, 03:40 UTC." -> ATBD_final: ATBD_final > 3. Input data & pre-processing > Dati di input

Paragrafo 7: Questi frame sono poi suddivisi in tile di 224 per 224 pixel, parzialmente so...
Frase 1: "Questi frame sono poi suddivisi in tile di 224 per 224 pixel, parzialmente sovrapposti, e impilati temporalmente in blocchi di 16 frame, corrispondenti a una finestra temporale di 80 minuti." -> ATBD_final: ATBD_final > 3. Input data & pre-processing (sintesi)
Frase 2: "L'overlap temporale consente un campionamento orario dei clip." -> ATBD_final: ATBD_final (sintesi)

Paragrafo 8: L'output della fase di detection e un'etichetta binaria per ciascun video til...
Frase 1: "L'output della fase di detection e un'etichetta binaria per ciascun video tile con uno score di classificazione che, in un contesto operativo, puo essere interpretato come indice di confidenza." -> ATBD_final: ATBD_final > 6. Verifica: validazione, test, incertezze, limiti (sintesi)
Frase 2: "L'output della fase di tracking e una coppia di coordinate del centro di rotazione nel dominio dei pixel del tile." -> ATBD_final: ATBD_final > 3. Input data & pre-processing (sintesi)
Frase 3: "La conversione da coordinate pixel a coordinate geografiche richiede l'inversione della trasformazione applicata durante il crop e il tiling." -> ATBD_final: ATBD_final > 2. Panoramica prodotto, definizioni e specifiche > Output variables, units, valid ranges, griglia/risoluzione, convenzioni
Frase 4: "Il prodotto deve inoltre includere un flag di copertura dati e un indicatore di qualita, per distinguere le situazioni con input incompleti o con segnali IR non coerenti con la dinamica ciclonica." -> ATBD_final: ATBD_final > 2. Panoramica prodotto, definizioni e specifiche > Quality flags / confidence indicators

Paragrafo 9: ## Input data e pre-processing L'addestramento usa dati IR di EUMETSAT Rapid ...
Frase 1: "## Input data e pre-processing
L'addestramento usa dati IR di EUMETSAT Rapid Scan High Rate SEVIRI Level 1.5 Image Data MSG, con alta frequenza di 5 minuti e alta risoluzione spaziale, utili per monitorare fenomeni meteorologici in rapida evoluzione." -> ATBD_final: ATBD_final > 3. Input data & pre-processing (sintesi)
Frase 2: "I dati sono scaricabili da Google Cloud Big Query Public Data, con permesso di redistribuzione, e Google ospita dataset pubblici e di terze parti per conto dei provider, offrendo accesso affidabile e su larga scala senza oneri di storage." -> ATBD_final: ATBD_final > 3. Input data & pre-processing > Dati di input
Frase 3: "I dati sono resi disponibili da Open Climate Fix, che li ha processati ulteriormente usando le librerie satip e SatPy." -> ATBD_final: ATBD_final > 3. Input data & pre-processing > Dati di input
Frase 4: "Il dataset disponibile non contiene i valori numerici originali, perche e stato calibrato con SatPy e linearmene normalizzato per produrre Brightness Temperature, con valori mappati nell'intervallo [0, 1023], cioe 10 bit per canale." -> ATBD_final: ATBD_final > 3. Input data & pre-processing > Dati di input
Frase 5: "Il dataset e in formato Zarr e non e stato riproiettato, quindi rimane in proiezione geostazionaria." -> ATBD_final: ATBD_final > 3. Input data & pre-processing > Dati di input

Paragrafo 10: Il dataset include anche le etichette di occorrenza dei medicanes basate sull...
Frase 1: "Il dataset include anche le etichette di occorrenza dei medicanes basate sull'output WP2300 e le coordinate del centro di rotazione ottenute da osservazioni indipendenti, oltre alle tracce modellistiche disponibili per tutti i cicloni considerati come riferimento." -> ATBD_final: ATBD_final > 3. Input data & pre-processing > Dati di input

Paragrafo 11: Il recupero dei valori fisici richiede la de-normalizzazione secondo la formu...
Frase 1: "Il recupero dei valori fisici richiede la de-normalizzazione secondo la formula BT = v (xmax - xmin) + xmin, dove v e il valore normalizzato in 0 a 1 e xmax e xmin sono specifici per canale." -> ATBD_final: ATBD_final > 3. Input data & pre-processing > Dati di input
Frase 2: "La tabella seguente riporta i valori in uso per i canali richiesti." -> ATBD_final: ATBD_final > 3. Input data & pre-processing > Dati di input

Paragrafo 12: ```text SEVIRI channel    Xmin      Xmax IR_097     2,84         317,87 IR_10...
Frase 1: "```text
SEVIRI channel    Xmin      Xmax
IR_097     2,84         317,87
IR_108     199,10       313,28
WV_062     199,57       249,92
WV_073     198,95       286,96
```." -> ATBD_final: ATBD_final > 3. Input data & pre-processing > Dati di input

Paragrafo 13: Table 5.1 – Maximum and minimum values for the channels of interest.
Frase 1: "Table 5.1 – Maximum and minimum values for the channels of interest." -> ATBD_final: ATBD_final > 3. Input data & pre-processing > Dati di input

Paragrafo 14: I canali infrarossi e di vapor d'acqua sono combinati in un composito Airmass...
Frase 1: "I canali infrarossi e di vapor d'acqua sono combinati in un composito Airmass RGB secondo le linee guida EUMETSAT, RGB Recipes e Best Practices, con riferimento a https://eumetrain.org/sites/default/files/2020-05/RGB_recipes.pdf e https://www-cdn.eumetsat.int/files/2020-04/pdf_using_rgb_best_practices.pdf." -> ATBD_final: ATBD_final > 3. Input data & pre-processing > Dati di input
Frase 2: "Il canale rosso e costruito come differenza tra il canale infrarosso di vapor d'acqua intorno a 6.2 µm, WV_062, e il canale di vapor d'acqua 7.3 µm, WV_073." -> ATBD_final: ATBD_final > 3. Input data & pre-processing > Dati di input
Frase 3: "Il canale verde e ottenuto dalla differenza tra i canali sensibili all'assorbimento dell'ozono IR_097 e IR_108, che evidenziano le intrusioni stratosferiche." -> ATBD_final: ATBD_final > 3. Input data & pre-processing > Dati di input
Frase 4: "Il canale blu e WV_062." -> ATBD_final: ATBD_final > 3. Input data & pre-processing > Dati di input
Frase 5: "Questa combinazione rende visibili masse d'aria, umidita e sistemi frontali, facilitando l'identificazione visiva della dinamica ciclonica e delle strutture cloud." -> ATBD_final: ATBD_final > 3. Input data & pre-processing > Dati di input
Frase 6: "Nel flusso operativo, questo composito e usato per costruire l'intero dataset." -> ATBD_final: ATBD_final > 3. Input data & pre-processing > Dati di input
Frase 7: "R = WV_062 – WV_073, G = IR_097 – IR_108, B = WV_062." -> ATBD_final: ATBD_final > 3. Input data & pre-processing > Dati di input

Paragrafo 15: Il processo di dataset building parte dal presupposto che i modelli di grandi...
Frase 1: "Il processo di dataset building parte dal presupposto che i modelli di grandi dimensioni richiedono grandi dataset." -> ATBD_final: ATBD_final > 3. Input data & pre-processing > Dati di input
Frase 2: "Un dataset precedente di Ifremer risultava non completo per mancanza di continuita temporale e insufficiente per un training adeguato." -> ATBD_final: ATBD_final > 3. Input data & pre-processing > Dati di input
Frase 3: "Per questo si e proceduto al download e alla ricostruzione dei dati, includendo la trasformazione inversa verso valori originali di Brightness Temperature e la costruzione del composito con le corrette finestre di normalizzazione Meteosat." -> ATBD_final: ATBD_final > 3. Input data & pre-processing > Dati di input
Frase 4: "We have the Source Dataset  (fig airmassRGB senza land borders)." -> ATBD_final: ATBD_final > 3. Input data & pre-processing > Dati di input

Paragrafo 16: Big models need big data. Need to download data. Transform back to original B...
Frase 1: "Big models need big data." -> ATBD_final: ATBD_final > 3. Input data & pre-processing > Dati di input
Frase 2: "Need to download data." -> ATBD_final: ATBD_final > 3. Input data & pre-processing > Dati di input
Frase 3: "Transform back to original Brightness Temperature values." -> ATBD_final: ATBD_final > 3. Input data & pre-processing > Dati di input
Frase 4: "Build the composite with proper normalization ranges as in the Meteosat recipe." -> ATBD_final: ATBD_final > 3. Input data & pre-processing > Dati di input

Paragrafo 17: Il dataset sorgente comprende circa 860K frame AirmassRGB di 1290 X 420 pixel...
Frase 1: "Il dataset sorgente comprende circa 860K frame AirmassRGB di 1290 X 420 pixel sull'area del Mediterraneo, per un totale di circa 600 GByte, con copertura temporale ⪞ 7.5 years tra 2010 e 2023." -> ATBD_final: ATBD_final > 3. Input data & pre-processing > Dati di input
Frase 2: "Un processo di download e pre-processing e stato implementato con script Python ottimizzati per garantire efficienza, perche la generazione del working dataset deve essere ripetuta per ogni scelta di parametri." -> ATBD_final: ATBD_final > 3. Input data & pre-processing > Dati di input
Frase 3: "Il dataset risultante e denominato Source Dataset, mentre il dataset prodotto da tiling e labeling e denominato Working Dataset." -> ATBD_final: ATBD_final > 3. Input data & pre-processing > Dati di input

Paragrafo 18: La regione mediterranea e suddivisa in tile 224 x 224 pixel parzialmente sovr...
Frase 1: "La regione mediterranea e suddivisa in tile 224 x 224 pixel parzialmente sovrapposti, con un numero tipico di 12 tile, ma variabile in funzione dell'overlap." -> ATBD_final: ATBD_final > 3. Input data & pre-processing > Pre-processing e labeling
Frase 2: "Ogni tile copre approssimativamente 757 km in latitudine e 805 km in longitudine, con risoluzione di circa 3.38 km per pixel in latitudine e 3.59 km per pixel in longitudine." -> ATBD_final: ATBD_final > 3. Input data & pre-processing > Pre-processing e labeling
Frase 3: "I tile sono impilati in clip di 16 frame distanziati di 5 minuti, per un totale di 80 minuti, con sovrapposizione temporale e generazione di un video tile ogni ora." -> ATBD_final: ATBD_final > 3. Input data & pre-processing > Pre-processing e labeling
Frase 4: "Questo formato e richiesto dal modello pre-addestrato per dimensioni spaziali e numero di frame." -> ATBD_final: ATBD_final > 3. Input data & pre-processing > Pre-processing e labeling
Frase 5: "fig tiles-mediterraneo  - fig video-tiles." -> ATBD_final: ATBD_final > 3. Input data & pre-processing > Pre-processing e labeling

Paragrafo 19: L'etichettatura si basa sul database TracksCL7, con trasformazione delle coor...
Frase 1: "L'etichettatura si basa sul database TracksCL7, con trasformazione delle coordinate del centro di rotazione da geografiche a pixel." -> ATBD_final: ATBD_final > 3. Input data & pre-processing > Pre-processing e labeling
Frase 2: "La procedura e altamente ottimizzata perche il working dataset viene ricostruito molte volte in funzione delle scelte di training." -> ATBD_final: ATBD_final > 3. Input data & pre-processing > Pre-processing e labeling
Frase 3: "Un tile e etichettato come positivo quando il centro di rotazione e contenuto nel tile per almeno 6 frame su 16, altrimenti e etichettato come negativo." -> ATBD_final: ATBD_final > 3. Input data & pre-processing > Pre-processing e labeling
Frase 4: "I tile contenenti il centro sono indicati come label 1 e gli altri come label 0, con rappresentazione grafica del green square e dei punti di traccia." -> ATBD_final: ATBD_final > 3. Input data & pre-processing > Pre-processing e labeling
Frase 5: "TracksCL7 deriva dalla combinazione di sette metodi di tracking applicati a reanalisi." -> ATBD_final: ATBD_final > 3. Input data & pre-processing > Pre-processing e labeling
Frase 6: "fig labeled-tiles." -> ATBD_final: ATBD_final > 3. Input data & pre-processing > Pre-processing e labeling

Paragrafo 20: In the following fig 5.3 it is shown the process of tiling and labeling from ...
Frase 1: "In the following fig 5.3 it is shown the process of tiling and labeling from a source image." -> ATBD_final: ATBD_final > 3. Input data & pre-processing > Pre-processing e labeling
Frase 2: "Fig." -> ATBD_final: ATBD_final > 3. Input data & pre-processing > Dati di input
Frase 3: "5.2 – Airmass RGB source frame: the squares are sliced to form video clips in the same position for 16 frames." -> ATBD_final: ATBD_final > 3. Input data & pre-processing > Pre-processing e labeling
Frase 4: "A green square is shown where the video tile is selected as positive sample, containing a cyclone center track (cyan dots)." -> ATBD_final: ATBD_final > 3. Input data & pre-processing > Pre-processing e labeling

Paragrafo 21: ## Algoritmo DeMeTrA e basato su VideoMAEv2, un transformer per video con app...
Frase 1: "## Algoritmo
DeMeTrA e basato su VideoMAEv2, un transformer per video con apprendimento auto-supervisionato." -> ATBD_final: ATBD_final > 4. Algoritmo (sintesi)
Frase 2: "Il modello e open source e pre-addestrato su grandi dataset video, e usa una architettura autoencoder che non richiede etichette nella fase di pre-training." -> ATBD_final: ATBD_final > 4. Algoritmo > Overview
Frase 3: "Il modello lavora su clip di 16 frame con dimensioni 224×224 pixel, e ogni frame viene suddiviso in patch non sovrapposte di 14x14 pixel nel forward pass." -> ATBD_final: ATBD_final > 4. Algoritmo > Overview
Frase 4: "L'architettura include una struttura encoder-decoder e una strategia di tube masking, in cui patch spatio-temporali sono mascherate per forzare l'apprendimento di rappresentazioni robuste." -> ATBD_final: ATBD_final > 4. Algoritmo > Overview
Frase 5: "DeMeTrA specializza VideoMAEv2 per il riconoscimento e tracking di cicloni tramite fine-tuning sul dataset Airmass RGB." -> ATBD_final: ATBD_final > 4. Algoritmo > Overview

Paragrafo 22: How Transformers Work and Why They Outperform CNNs. La scelta del transformer...
Frase 1: "How Transformers Work and Why They Outperform CNNs." -> ATBD_final: ATBD_final > 4. Algoritmo > Overview
Frase 2: "La scelta del transformer e motivata da proprieta specifiche di self-attention." -> ATBD_final: ATBD_final > 4. Algoritmo > Overview
Frase 3: "I transformer calcolano dipendenze tra tutte le parti di un'immagine o di una sequenza video, permettendo a ogni patch di interagire direttamente con tutte le altre e modellare dipendenze a lungo raggio." -> ATBD_final: ATBD_final > 4. Algoritmo > Overview
Frase 4: "Questo consente di apprendere relazioni contestuali dinamiche, invece di filtri statici locali, e di scalare a dataset e modelli piu grandi, come ViT-Large e ViT-Huge, con prestazioni elevate senza overfitting eccessivo." -> ATBD_final: ATBD_final > 4. Algoritmo > Overview
Frase 5: "Questa architettura e particolarmente efficace per video understanding, dove le relazioni temporali tra frame sono cruciali." -> ATBD_final: ATBD_final > 4. Algoritmo > Overview
Frase 6: "Un riferimento specifico che sintetizza questi aspetti e il lavoro Comparing Vision Transformers and Convolutional Neural Networks for Image Classification: A Literature Review di Maurício, Domingues e Bernardino, con URL https://www.mdpi.com/2225-1154/12/12/220." -> ATBD_final: ATBD_final > 4. Algoritmo > Overview

Paragrafo 23: VideoMAE: A Transformer-Based Model for Video Understanding. VideoMAE e un ap...
Frase 1: "VideoMAE: A Transformer-Based Model for Video Understanding." -> ATBD_final: ATBD_final > 4. Algoritmo > Overview
Frase 2: "VideoMAE e un approccio di self-supervised learning per video che estende i Masked Autoencoders da immagini a sequenze spatio-temporali." -> ATBD_final: ATBD_final > 4. Algoritmo > Overview
Frase 3: "Key Features of VideoMAE." -> ATBD_final: ATBD_final > 4. Algoritmo > Overview
Frase 4: "Il modello usa un masking ratio elevato, tipicamente 90-95%, rispetto al 75% delle MAE per immagini, sfruttando la ridondanza temporale nei video." -> ATBD_final: ATBD_final > 4. Algoritmo > Overview
Frase 5: "Il masking e di tipo tube masking, cioe maschera cubi spatio-temporali continui, rendendo la ricostruzione piu complessa rispetto al mascheramento casuale di singoli patch." -> ATBD_final: ATBD_final > 4. Algoritmo > Overview
Frase 6: "L'architettura e asimmetrica: l'encoder processa solo i token visibili, rendendo l'addestramento efficiente, mentre il decoder ricostruisce i token mancanti con una struttura piu leggera." -> ATBD_final: ATBD_final > 4. Algoritmo > Overview

Paragrafo 24: Why VideoMAE is Effective for Short Video Classification. VideoMAE e particol...
Frase 1: "Why VideoMAE is Effective for Short Video Classification." -> ATBD_final: ATBD_final > 4. Algoritmo > Overview
Frase 2: "VideoMAE e particolarmente efficace per la classificazione di video brevi perche l'auto-supervised pretext task di ricostruzione permette di apprendere rappresentazioni spatio-temporali ricche senza etichette." -> ATBD_final: ATBD_final > 4. Algoritmo > Overview
Frase 3: "La scalabilita e dimostrata su dataset come Kinetics-400 e Something-Something V2 e su modelli grandi come ViT-L, ViT-H e ViT-g, con prestazioni di stato dell'arte." -> ATBD_final: ATBD_final > 4. Algoritmo > Overview
Frase 4: "La trasferibilita e elevata e consente l'uso del backbone su compiti downstream come video classification, action recognition e spatiotemporal localization." -> ATBD_final: ATBD_final > 4. Algoritmo > Overview

Paragrafo 25: VideoMAE Architecture Overview. La struttura interna di VideoMAE puo essere d...
Frase 1: "VideoMAE Architecture Overview." -> ATBD_final: ATBD_final > 4. Algoritmo > Overview
Frase 2: "La struttura interna di VideoMAE puo essere descritta come segue." -> ATBD_final: ATBD_final > 4. Algoritmo > Overview
Frase 3: "Key Components of VideoMAE." -> ATBD_final: ATBD_final > 4. Algoritmo > Overview
Frase 4: "Il modello usa cube embedding per convertire frame in token, con cubi spatio-temporali di dimensione 2 × 16 × 16 pixel, corrispondenti a due frame e una risoluzione di 16×16." -> ATBD_final: ATBD_final > 4. Algoritmo > Overview
Frase 5: "Cube Embedding (Patch Tokenization)." -> ATBD_final: ATBD_final > 4. Algoritmo > Overview
Frase 6: "Ogni patch e embedded in un vettore con position encoding, riducendo la ridondanza spaziale e temporale." -> ATBD_final: ATBD_final > 4. Algoritmo > Overview
Frase 7: "Encoder: Masked Spatiotemporal Transformer." -> ATBD_final: ATBD_final > 4. Algoritmo > Overview
Frase 8: "Joint Space-Time Attention." -> ATBD_final: ATBD_final > 4. Algoritmo > Overview
Frase 9: "Masked Token Strategy." -> ATBD_final: ATBD_final > 4. Algoritmo > Overview
Frase 10: "L'encoder e un Vision Transformer con attenzione congiunta nello spazio e nel tempo, e processa solo circa il 10% dei token di input, mentre il resto viene scartato, aumentando l'efficienza." -> ATBD_final: ATBD_final > 4. Algoritmo > Overview
Frase 11: "Decoder: Lightweight Reconstruction." -> ATBD_final: ATBD_final > 4. Algoritmo > Overview
Frase 12: "Asymmetric Architecture." -> ATBD_final: ATBD_final > 4. Algoritmo > Overview
Frase 13: "Il decoder ricostruisce i token mancanti e opera sul 100% dei token ma con profondita e larghezza ridotte." -> ATBD_final: ATBD_final > 4. Algoritmo > Overview
Frase 14: "Una configurazione tipica usa 12 blocchi per l'encoder ViT-Base e 4 blocchi per il decoder, con un vantaggio computazionale significativo." -> ATBD_final: ATBD_final > 4. Algoritmo > Overview

Paragrafo 26: Why Tube Masking? Il tube masking e motivato dal fatto che, nei video, masche...
Frase 1: "Why Tube Masking? Il tube masking e motivato dal fatto che, nei video, mascherare singoli patch e troppo facile per la ricostruzione a causa della ridondanza temporale." -> ATBD_final: ATBD_final > 4. Algoritmo > Overview
Frase 2: "How It Works? Mascherare interi tubi spatio-temporali aumenta la difficolta e forza l'apprendimento di rappresentazioni robuste." -> ATBD_final: ATBD_final > 4. Algoritmo > Overview
Frase 3: "Dual Masking in VideoMAE V2." -> ATBD_final: ATBD_final > 4. Algoritmo > Overview
Frase 4: "VideoMAEv2 introduce una doppia mascheratura: l'encoder vede solo una frazione dei token, tipicamente con 90% di token rimossi, e il decoder ricostruisce solo un sottoinsieme dei cubi mancanti, riducendo memoria e tempo di training e permettendo di scalare a modelli con miliardi di parametri." -> ATBD_final: ATBD_final > 4. Algoritmo > Overview
Frase 5: "Computational Efficiency." -> ATBD_final: ATBD_final > 4. Algoritmo > Overview
Frase 6: "Asymmetric Architecture: il decoder piu piccolo riduce il costo computazionale pur mantenendo ricostruzioni di qualita." -> ATBD_final: ATBD_final > 4. Algoritmo > Overview
Frase 7: "Efficient Self-Supervised Pre-Training: no labeled data is required—models learn representations by reconstructing missing patches." -> ATBD_final: ATBD_final > 4. Algoritmo > Overview
Frase 8: "Scalability: VideoMAE can scale from small models (ViT-B) to billion-parameter models (ViT-g)." -> ATBD_final: ATBD_final > 4. Algoritmo > Overview

Paragrafo 27: (figura VideoMAEv2_flowchart.png) Fig. 5.1 – Technical details of the VideoMA...
Frase 1: "(figura VideoMAEv2_flowchart.png)
Fig." -> ATBD_final: ATBD_final > 4. Algoritmo > Overview
Frase 2: "5.1 – Technical details of the VideoMAEv2 model: encoder-decoder structure, with separate masking for each one." -> ATBD_final: ATBD_final > 4. Algoritmo > Overview
Frase 3: "The latent representation vector (embedding) is shown between the encoder and the decoder." -> ATBD_final: ATBD_final > 4. Algoritmo > Overview

Paragrafo 28: La pipeline algoritmica per DeMeTrA integra la catena di pre-processing e i t...
Frase 1: "La pipeline algoritmica per DeMeTrA integra la catena di pre-processing e i tre stadi di training." -> ATBD_final: ATBD_final > 4. Algoritmo > Flowchart (testuale)

Paragrafo 29: ```text Dati SEVIRI RSS -> De-normalizzazione BT -> Composito Airmass RGB -> ...
Frase 1: "```text
Dati SEVIRI RSS -> De-normalizzazione BT -> Composito Airmass RGB -> Crop Mediterraneo
-> Tiling 224x224 -> Stack 16 frame -> Etichettatura con TRACKS_CL7
-> Specialization training VideoMAE -> Classificazione presenza ciclone
-> Regressione del centro di rotazione
```." -> ATBD_final: ATBD_final > 4. Algoritmo > Flowchart (testuale)

Paragrafo 30: ```text carica canali IR_097, IR_108, WV_062, WV_073 per ciascun timestamp ca...
Frase 1: "```text
carica canali IR_097, IR_108, WV_062, WV_073 per ciascun timestamp
calcola BT con BT = v (xmax - xmin) + xmin
costruisci Airmass RGB con R = WV_062 - WV_073, G = IR_097 - IR_108, B = WV_062
ritaglia su area mediterranea e normalizza
per ogni tile 224x224 nella griglia mediterranea
    crea un clip con 16 frame separati di 5 minuti
    se la traccia del centro di rotazione cade nel tile per almeno 6 frame
        assegna etichetta positiva
    altrimenti assegna etichetta negativa
addestra VideoMAEv2 con mascheramento a tubi su clip non etichettati
usa il backbone specializzato per addestrare un classificatore binario
usa lo stesso backbone per addestrare un regressore di coordinate RC
```." -> ATBD_final: ATBD_final > 4. Algoritmo > Pseudocodice (sintetico)

Paragrafo 31: ## Fase di training Three stage training: Il training e organizzato in tre st...
Frase 1: "## Fase di training
Three stage training:
Il training e organizzato in tre stadi." -> ATBD_final: ATBD_final > 5. Fase di training (sintesi)
Frase 2: "Le due fasi supervisionate usano le feature apprese nella fase di specializzazione." -> ATBD_final: ATBD_final > 5. Fase di training
Frase 3: "Il primo e un addestramento auto-supervisionato di specializzazione su clip Airmass RGB, senza etichette, con circa 80’000 video in training e 20’000 in test." -> ATBD_final: ATBD_final > 5. Fase di training
Frase 4: "In questa fase si usa il modello pre-addestrato di taglia “giant” per una specializzazione non supervisionata che si aggiunge al pre-training da zero, con l'obiettivo di aumentare la capacita di estrazione delle feature utili ai task successivi." -> ATBD_final: ATBD_final > 5. Fase di training
Frase 5: "Il training e lungo, dell'ordine di giorni, e la curva di loss di training e validazione non mostra overfitting, indicando una specializzazione efficace, per cui il modello viene salvato come checkpoint per i task successivi." -> ATBD_final: ATBD_final > 5. Fase di training
Frase 6: "Training and validation losses versus epochs." -> ATBD_final: ATBD_final > 5. Fase di training

Paragrafo 32: Self-supervised dataset: random videos, con train set ~ 80K video tiles e tes...
Frase 1: "Self-supervised dataset: random videos, con train set ~ 80K video tiles e test set ~ 20K video tiles." -> ATBD_final: ATBD_final > 5. Fase di training
Frase 2: "Long time training (~ days)." -> ATBD_final: ATBD_final > 5. Fase di training
Frase 3: "Results: no overfitting → good! → save this model as a checkpoint for the next stage training." -> ATBD_final: ATBD_final > 5. Fase di training

Paragrafo 33: Il secondo stadio e una classificazione binaria. Il primo trial usa un datase...
Frase 1: "Il secondo stadio e una classificazione binaria." -> ATBD_final: ATBD_final > 5. Fase di training
Frase 2: "Il primo trial usa un dataset bilanciato di circa 3000 video, divisi in due classi, con class 1 che include cicloni in TRACKS_CL7, medicanes e altri cicloni mediterranei, e class 0 senza cicloni." -> ATBD_final: ATBD_final > 5. Fase di training
Frase 3: "Using Tracks CL7: Max." -> ATBD_final: ATBD_final > 5. Fase di training
Frase 4: "Accuracy 80 %." -> ATBD_final: ATBD_final > 5. Fase di training
Frase 5: "Need for increased dataset quality: why? because of mismatch tra tracce e contenuto IR, con casi di cloud absence o assenza di rotazione evidente." -> ATBD_final: ATBD_final > 5. Fase di training
Frase 6: "Mismatch with the tracks: IR images may not capture cyclonic dynamics for certain tracks (cloud absence, no cloud rotation...)." -> ATBD_final: ATBD_final > 5. Fase di training
Frase 7: "Questo ha portato a un percorso di raffinamento del dataset con quattro passi, partendo da TRACKS_CL7, passando a TRACKS_CL10, poi limitando ai soli medicanes e infine restringendo la finestra temporale ai frame con rotazione chiaramente osservabile." -> ATBD_final: ATBD_final > 5. Fase di training
Frase 8: "Supervised binary classification dataset - 4 refining steps: Step 1: Tracks CL7, Step 2: Tracks CL10, Step 3: Only Medicanes *, Step 4: Only Medicanes * and narrow time window with clearly observable rotation." -> ATBD_final: ATBD_final > 5. Fase di training
Frase 9: "* From the Full_List_Medicanes." -> ATBD_final: ATBD_final > 5. Fase di training

Paragrafo 34: Dataset refinement: Cyclones new time boundaries (manual selection). Using On...
Frase 1: "Dataset refinement: Cyclones new time boundaries (manual selection)." -> ATBD_final: ATBD_final > 5. Fase di training
Frase 2: "Using Only Medicanes *." -> ATBD_final: ATBD_final > 5. Fase di training
Frase 3: "Le finestre temporali dei medicanes sono state riviste manualmente, con nuovi start e end times basati su ispezione visiva e presenza di rotazione chiaramente visibile." -> ATBD_final: ATBD_final > 5. Fase di training
Frase 4: "New Start and End times based on visual inspection." -> ATBD_final: ATBD_final > 5. Fase di training
Frase 5: "Clearly visible cyclone clouds rotation." -> ATBD_final: ATBD_final > 5. Fase di training
Frase 6: "(medicane_new_windiws.csv)." -> ATBD_final: ATBD_final > 5. Fase di training

Paragrafo 35: Il dataset finale per la detection include 18 cicloni, con la lista riportata...
Frase 1: "Il dataset finale per la detection include 18 cicloni, con la lista riportata di seguito in forma tabellare, e la partizione train, validation e test e coerente con la tabella descritta nella sezione dati." -> ATBD_final: ATBD_final > 5. Fase di training

Paragrafo 36: ```text ID 		1283	1328	1358	1421	1461		1466	1500	1521 	1542 Name 	Unnamed Rol...
Frase 1: "```text
ID 		1283	1328	1358	1421	1461		1466	1500	1521 	1542
Name 	Unnamed Rolf 	Unnamed Unnamed Qendresa	Unnamed Unnamed Unnamed Trixie
ID 		1575 	1674	1702	1715	1716		-		-		-		-
Name 	Numa 	Ianos 	Unnamed Unnamed Unnamed 	Apollo	Blas	Daniel	Juliette
```." -> ATBD_final: ATBD_final > 5. Fase di training

Paragrafo 37: I primi identificativi derivano da TRACKS_CL7, mentre i casi Apollo, Blas, Da...
Frase 1: "I primi identificativi derivano da TRACKS_CL7, mentre i casi Apollo, Blas, Daniel e Juliette non sono inclusi in Flaounas et al." -> ATBD_final: ATBD_final > 5. Fase di training
Frase 2: "(2023) e derivano dal minimo MSLP ERA5." -> ATBD_final: ATBD_final > 5. Fase di training
Frase 3: "Il dataset e partizionato per evitare leakage tra eventi in training e test." -> ATBD_final: ATBD_final > 5. Fase di training

Paragrafo 38: La partizione delle classi e illustrata nella tabella riportata di seguito, d...
Frase 1: "La partizione delle classi e illustrata nella tabella riportata di seguito, dove la differenza tra i due validation set riguarda il bilanciamento di classi positive e negative." -> ATBD_final: ATBD_final > 5. Fase di training

Paragrafo 39: ```text 			Num. cyclones 	Total time interval 	Num. Video clips Train set   1...
Frase 1: "```text
			Num." -> ATBD_final: ATBD_final > 5. Fase di training
Frase 2: "cyclones 	Total time interval 	Num." -> ATBD_final: ATBD_final > 5. Fase di training
Frase 3: "Video clips
Train set   12 				23 days 8h45’ 			1238 					Balanced
Val  set    3 				7 days 8h5' 			354 					Balanced
Test set    3 				8 days 13h25' 			2400 					201 positives (cyclones)
																			2199 negatives (no cyclones)
```." -> ATBD_final: ATBD_final > 5. Fase di training

Paragrafo 40: In the next tables it is shown the time window for some example medicane, bef...
Frase 1: "In the next tables it is shown the time window for some example medicane, before and after the narrower time window selection: (tabella completa medicanes_new_windows.csv)." -> ATBD_final: ATBD_final > 5. Fase di training

Paragrafo 41: Best results: Max. Accuracy 91 %. Balanced dataset (validation set). Best res...
Frase 1: "Best results: Max." -> ATBD_final: ATBD_final > 5. Fase di training
Frase 2: "Accuracy 91 %." -> ATBD_final: ATBD_final > 5. Fase di training
Frase 3: "Balanced dataset (validation set)." -> ATBD_final: ATBD_final > 5. Fase di training
Frase 4: "Best results: Max." -> ATBD_final: ATBD_final > 5. Fase di training
Frase 5: "Accuracy 89 %." -> ATBD_final: ATBD_final > 5. Fase di training
Frase 6: "UNbalanced dataset (test set)." -> ATBD_final: ATBD_final > 5. Fase di training
Frase 7: "I risultati migliori riportano una Max." -> ATBD_final: ATBD_final > 5. Fase di training
Frase 8: "Accuracy di 91% su validation bilanciata e 89% su test sbilanciato, con confusion matrices come supporto visivo." -> ATBD_final: ATBD_final > 5. Fase di training
Frase 9: "(confusion matrices images)." -> ATBD_final: ATBD_final > 5. Fase di training

Paragrafo 42: Il terzo stadio e una regressione per il tracking del centro di rotazione, co...
Frase 1: "Il terzo stadio e una regressione per il tracking del centro di rotazione, costruita selezionando solo i video tile con cicloni dal dataset di detection." -> ATBD_final: ATBD_final > 5. Fase di training
Frase 2: "Tracking dataset: collecting only video tiles with cyclones from detection dataset." -> ATBD_final: ATBD_final > 5. Fase di training
Frase 3: "Il tracking dataset comprende 12 cicloni per training con 834 video, 3 cicloni per test con 160 video e 3 cicloni per validation con 192 video." -> ATBD_final: ATBD_final > 5. Fase di training
Frase 4: "In un riepilogo alternativo, il dataset di tracking ha 835 campioni in training e 280 in test, e la specifica va consolidata per la riproducibilita." -> ATBD_final: ATBD_final > 5. Fase di training
Frase 5: "The dataset has the following number of samples: Training set samples: 835." -> ATBD_final: ATBD_final > 5. Fase di training
Frase 6: "Test set samples: 280." -> ATBD_final: ATBD_final > 5. Fase di training
Frase 7: "...following activity:  devo completare cosa ho fatto." -> ATBD_final: ATBD_final > 5. Fase di training

Paragrafo 43: In the following figure some dataset samples are shown, along with their labe...
Frase 1: "In the following figure some dataset samples are shown, along with their labeled center." -> ATBD_final: ATBD_final > 5. Fase di training
Frase 2: "Fig." -> ATBD_final: ATBD_final > 3. Input data & pre-processing > Dati di input
Frase 3: "5.4 tracking samples – Example frames from tracking video dataset." -> ATBD_final: ATBD_final > 5. Fase di training
Frase 4: "The red dot represents the center track label." -> ATBD_final: ATBD_final > 5. Fase di training

Paragrafo 44: 16- tracking results  (image plot histogram error in pixel and in km).
Frase 1: "16- tracking results  (image plot histogram error in pixel and in km)." -> ATBD_final: ATBD_final > 5. Fase di training

Paragrafo 45: 17- Cyclone first pass.
Frase 1: "17- Cyclone first pass." -> ATBD_final: ATBD_final > 5. Fase di training

Paragrafo 46: ## Verifica, validazione e limiti La verifica della detection usa accuracy su...
Frase 1: "## Verifica, validazione e limiti
La verifica della detection usa accuracy su validation e test con dataset bilanciati e sbilanciati." -> ATBD_final: ATBD_final > 6. Verifica: validazione, test, incertezze, limiti (sintesi)
Frase 2: "Le confusion matrix permettono di quantificare falsi positivi e falsi negativi e di comprendere la sensibilita al mismatch tra tracce e segnali IR." -> ATBD_final: ATBD_final > 6. Verifica: validazione, test, incertezze, limiti
Frase 3: "L'analisi degli errori evidenzia casi in cui un centro tracciato non corrisponde a rotazione visibile, a causa di cloud absence o strutture non cicloniche." -> ATBD_final: ATBD_final > 6. Verifica: validazione, test, incertezze, limiti
Frase 4: "Per il tracking, l'errore di localizzazione e rappresentato in pixel e in chilometri con istogrammi, utili per valutare precisione e distribuzione degli errori." -> ATBD_final: ATBD_final > 6. Verifica: validazione, test, incertezze, limiti

Paragrafo 47: Il sistema presenta limiti dovuti alla dipendenza dalle tracce, alla sensibil...
Frase 1: "Il sistema presenta limiti dovuti alla dipendenza dalle tracce, alla sensibilita alle condizioni di visibilita IR e alla mancanza di una definizione completa di confidenza e flags di qualita." -> ATBD_final: ATBD_final > 6. Verifica: validazione, test, incertezze, limiti (sintesi)
Frase 2: "La conversione da coordinate pixel a geografiche richiede un modello geometrico coerente con la proiezione geostazionaria e con il crop." -> ATBD_final: ATBD_final (sintesi)
Frase 3: "La fase di cyclone first pass e menzionata come passaggio di workflow ma non e ancora descritta in dettaglio." -> ATBD_final: ATBD_final > 6. Verifica: validazione, test, incertezze, limiti (sintesi)

Paragrafo 48: ## Limiti e lavori futuri La documentazione tecnica deve includere una descri...
Frase 1: "## Limiti e lavori futuri
La documentazione tecnica deve includere una descrizione completa degli iperparametri e delle scelte di ottimizzazione, insieme alla configurazione di training e inference, per garantire riproducibilita." -> ATBD_final: ATBD_final > 6. Verifica: validazione, test, incertezze, limiti (sintesi)
Frase 2: "E necessario definire formalmente i flag di qualita, gli indicatori di confidenza e le regole di validita del prodotto in presenza di input incompleti." -> ATBD_final: ATBD_final > 6. Verifica: validazione, test, incertezze, limiti
Frase 3: "La sezione di tracking richiede l'allineamento definitivo dei numeri di campioni e la definizione di metriche aggregate con confronti in chilometri." -> ATBD_final: ATBD_final > 6. Verifica: validazione, test, incertezze, limiti
Frase 4: "Ulteriori case study con evoluzione temporale e confronto con analisi sinottiche fornirebbero una validazione qualitativa utile al contesto ESA." -> ATBD_final: ATBD_final > 6. Verifica: validazione, test, incertezze, limiti

Paragrafo 49: ## Provenance Il modello di base e VideoMAEv2, un modello open source per vid...
Frase 1: "## Provenance
Il modello di base e VideoMAEv2, un modello open source per video, con repository dedicato al progetto https://github.com/dandarm/VideoMAEv2 e riferimenti alle pubblicazioni originali su Arxiv." -> ATBD_final: ATBD_final > 8. References bibliografiche (sintesi)
Frase 2: "La pipeline qui descritta introduce una specializzazione su dati Airmass RGB del Mediterraneo, un pre-processing con de-normalizzazione BT, compositi RGB, tiling e stacking temporale, oltre a dataset etichettati con TRACKS_CL7 e successive rifiniture." -> ATBD_final: ATBD_final > 5. Fase di training (sintesi)
Frase 3: "La fase supervisionata aggiunge un classificatore binario e un regressore per il centro di rotazione, entrambi basati sullo stesso backbone specializzato." -> ATBD_final: ATBD_final (sintesi)
Frase 4: "L'entrypoint operativo e la struttura delle cartelle devono essere documentati esplicitamente nel repository per garantire tracciabilita." -> ATBD_final: ATBD_final > 7. Provenance (obbligatoria, compatta) (sintesi)

Paragrafo 50: ## References e panorama della letteratura Artificial Intelligence applied to...
Frase 1: "## References e panorama della letteratura
Artificial Intelligence applied to Atmospheric Science is quite a new field, e i lavori recenti indicano un panorama in rapida evoluzione." -> ATBD_final: ATBD_final > 8. References bibliografiche (sintesi)
Frase 2: "A Comprehensive AI Approach for Monitoring and Forecasting Medicanes Development (2024), di J. Martinez-Amaya, V. Nieves e J. Muñoz-Mari, propone un approccio con tracking tramite k-means clustering e forecast basato su CNN e Random Forest." -> ATBD_final: ATBD_final > 8. References bibliografiche (sintesi)
Frase 3: "Il dataset include 58 medicanes tra 1984 e 2023 con immagini IR Meteosat in Brightness Temperature e reanalisi CERRA ed ERA5 con MSLP e vento." -> ATBD_final: ATBD_final > 5. Fase di training (sintesi)
Frase 4: "Lo studio e il primo modello ML dedicato ai medicanes e consente la previsione di eventi estremi fino a due giorni con accuracy 65-80%, offrendo un'alternativa ai modelli tradizionali e adattabilita al cambiamento climatico." -> ATBD_final: ATBD_final > 6. Verifica: validazione, test, incertezze, limiti (sintesi)
Frase 5: "URL https://www.mdpi.com/2225-1154/12/12/220." -> ATBD_final: ATBD_final > 4. Algoritmo > Overview

Paragrafo 51: A Statistical Learning Approach to Mediterranean Cyclones (2025), di L. Rover...
Frase 1: "A Statistical Learning Approach to Mediterranean Cyclones (2025), di L. Roveri, L. Fery, L. Cavicchia e F. Grotto, usa una fase non supervisionata con Latent Dirichlet Allocation per ridurre dimensionalita e una fase supervisionata con classificatore statistico su feature LDA." -> ATBD_final: ATBD_final > 8. References bibliografiche (sintesi)
Frase 2: "Il dataset e ERA5 con campi di vento e pressione combinati con un archivio di tracce mediterranee." -> ATBD_final: ATBD_final > 5. Fase di training (sintesi)
Frase 3: "Il risultato e un workflow di classificazione con accuracy circa 90% nel rilevamento di cicloni mediterranei usando poche feature, superando le difficolta di definire forma e diametro dei medicanes e abilitando precursori temporali di cicloni estremi." -> ATBD_final: ATBD_final > 6. Verifica: validazione, test, incertezze, limiti (sintesi)
Frase 4: "URL https://arxiv.org/pdf/2501.15694v1." -> ATBD_final: ATBD_final > 8. References bibliografiche

Paragrafo 52: Deepti: Deep-Learning-Based Tropical Cyclone Intensity Estimation System (202...
Frase 1: "Deepti: Deep-Learning-Based Tropical Cyclone Intensity Estimation System (2020), di M. Maskey, R. Ramachandran, M. Ramasubramanian, I. Gurung, et al., utilizza una CNN custom per la regressione dell'intensita con vento massimo." -> ATBD_final: ATBD_final > 8. References bibliografiche (sintesi)
Frase 2: "Il dataset include immagini IR GOES ogni 15 minuti su piu satelliti geostazionari tra 2000 e 2019, con labels best-track HURDAT2." -> ATBD_final: ATBD_final > 8. References bibliografiche (sintesi)
Frase 3: "Il sistema automatizza la stima di intensita da IR, emulando l'analisi Dvorak, e ottiene RMSE di 13.24 knots, comparabile a tecniche operative, con stime near real time e un portale di visualizzazione." -> ATBD_final: ATBD_final (sintesi)
Frase 4: "URL https://ieeexplore.ieee.org/document/9149719." -> ATBD_final: ATBD_final > 8. References bibliografiche

Paragrafo 53: A Novel Deep Learning Based Model for Tropical Intensity Estimation and Post-...
Frase 1: "A Novel Deep Learning Based Model for Tropical Intensity Estimation and Post-Disaster Management of Hurricanes (2021), di J. Devaraj, S. Ganesan, R. M. Elavarasan e U. Subramaniam, usa una CNN potenziata con batch normalization e dropout per la stima di intensita, e una fase di transfer learning con VGG-19 per la classificazione di danni ed eventi." -> ATBD_final: ATBD_final > 1. Scopo e contesto (sintesi)
Frase 2: "Il dataset include immagini IR GOES con label HURDAT2 su uragani 2000-2019, oltre a immagini satellitari post-evento, ad esempio Houston, per il danno e video di severe weather per la classificazione." -> ATBD_final: ATBD_final > 8. References bibliografiche (sintesi)
Frase 3: "Il modello riduce l'errore a 7.6 knots RMSE e raggiunge 98% accuracy per il danno e 97% per eventi severi, con un approccio integrato per previsione e gestione impatti." -> ATBD_final: ATBD_final > 6. Verifica: validazione, test, incertezze, limiti (sintesi)
Frase 4: "URL https://www.mdpi.com/2076-3417/11/9/4129." -> ATBD_final: ATBD_final > 8. References bibliografiche

Paragrafo 54: Tropical and Extratropical Cyclone Detection Using Deep Learning (2020), di C...
Frase 1: "Tropical and Extratropical Cyclone Detection Using Deep Learning (2020), di C. Kumler-Bonfanti, J. Stewart, D. Hall e M. Govett, usa una segmentazione con U-Net, in quattro varianti, per identificare regioni cicloniche, con confronto di etichette tra IBTrACS e un algoritmo euristico." -> ATBD_final: ATBD_final > 8. References bibliografiche (sintesi)
Frase 2: "Il dataset include mappe globali di total precipitable water da GFS a 0.5 gradi e immagini satellitari GOES nel canale water vapor, con ground truth da IBTrACS e detection euristica di cicloni extratropicali." -> ATBD_final: ATBD_final > 8. References bibliografiche (sintesi)
Frase 3: "I modelli raggiungono accuracy 80-99% nel marking delle regioni cicloniche con Dice 0.51-0.76, includendo vortici deboli ignorati manualmente, e il modello per cicloni extratropicali e tre volte piu veloce dell'algoritmo operativo di riferimento." -> ATBD_final: ATBD_final > 6. Verifica: validazione, test, incertezze, limiti (sintesi)
Frase 4: "L'uso di input multisorgente consente di scoprire cicloni ambigui che sfuggono ai criteri stretti." -> ATBD_final: ATBD_final > 2. Panoramica prodotto, definizioni e specifiche (sintesi)
Frase 5: "URL https://colab.ws/articles/10.1175%2Fjamc-d-20-0117.1." -> ATBD_final: ATBD_final > 8. References bibliografiche

Paragrafo 55: A Hybrid ML/Physics-Based Modeling Framework for 2-Week Extended Prediction o...
Frase 1: "A Hybrid ML/Physics-Based Modeling Framework for 2-Week Extended Prediction of Tropical Cyclones (2024), di X. Liu et al., combina un modello numerico WRF a circa 2 km con un modello deep learning globale Pangu-Weather a circa 25 km." -> ATBD_final: ATBD_final > 8. References bibliografiche (sintesi)
Frase 2: "Il dataset comprende simulazioni WRF ad alta risoluzione e forecast ML, con test su casi reali come il ciclone Freddy 2023." -> ATBD_final: ATBD_final > 8. References bibliografiche (sintesi)
Frase 3: "Il framework estende la previsione di traiettoria e intensita fino a circa 2 settimane rispetto ai circa 5 giorni dei modelli tradizionali, con miglioramento a 7 giorni di accuratezza e affidabilita lungo 14 giorni." -> ATBD_final: ATBD_final (sintesi)
Frase 4: "URL https://agupubs.onlinelibrary.wiley.com/doi/10.1029/2024JH000207." -> ATBD_final: ATBD_final > 8. References bibliografiche

Paragrafo 56: Tropical cyclone size estimation based on deep learning using infrared and mi...
Frase 1: "Tropical cyclone size estimation based on deep learning using infrared and microwave satellite data (2023), di J. Xu, X. Wang, H. Wang, C. Zhao, H. Wang e J. Zhu, propone TC-ResNet, una variante di ResNet-50 con conv 5×5 sullo shortcut e doppia attenzione canale/spazio per la regressione del raggio del vento." -> ATBD_final: ATBD_final > 8. References bibliografiche (sintesi)
Frase 2: "Il dataset integra immagini IR geostazionarie e mappe microonde passive dal 2003 al 2017 e il dataset globale R34 best-track." -> ATBD_final: ATBD_final > 5. Fase di training (sintesi)
Frase 3: "Il modello ottiene un errore medio di 11.3 nm, circa 21 km, con coefficiente di correlazione 0.907, superiore a metodi empirici tradizionali." -> ATBD_final: ATBD_final (sintesi)
Frase 4: "L'integrazione multisorgente permette di filtrare rumore e blur nelle nubi cicloniche e migliorare la stima del rischio associato a vento e swell." -> ATBD_final: ATBD_final (sintesi)
Frase 5: "URL https://www.frontiersin.org/journals/marine-science/articles/10.3389/fmars.2022.1077901/full." -> ATBD_final: ATBD_final > 8. References bibliografiche

Paragrafo 57: Detecting Extratropical Cyclones of the Northern Hemisphere with Single Shot ...
Frase 1: "Detecting Extratropical Cyclones of the Northern Hemisphere with Single Shot Detector (2022), di M. Shi, P. He, Y. Shi et al., applica un SSD one-stage per rilevare centri ciclonici e fase evolutiva." -> ATBD_final: ATBD_final > 8. References bibliografiche (sintesi)
Frase 2: "Il dataset include immagini satellitari whole-disk visibile e IR dell'emisfero Nord, con etichette manuali per ciclogenesi, maturita e dissipazione basate su criteri Bonfanti 2017." -> ATBD_final: ATBD_final > 5. Fase di training (sintesi)
Frase 3: "Il modello ottiene mAP 86.6% per cicloni maturi e circa 79.3% considerando le tre classi, dimostrando l'efficacia del deep learning per cicloni extratropicali con grande variabilita di forma e dimensione." -> ATBD_final: ATBD_final (sintesi)
Frase 4: "URL https://www.mdpi.com/2072-4292/14/2/254." -> ATBD_final: ATBD_final > 8. References bibliografiche

## Fonte: ATBD_videomae.md
Paragrafo 1: # ATBD - VideoMAE per Medicanes (modulo videomae)
Frase 1: "# ATBD - VideoMAE per Medicanes (modulo videomae)." -> ATBD_final: ATBD_final (sintesi)

Paragrafo 2: ## 1. Scopo e contesto
Frase 1: "## 1." -> ATBD_final: ATBD_final > 1. Scopo e contesto
Frase 2: "Scopo e contesto." -> ATBD_final: ATBD_final > 1. Scopo e contesto

Paragrafo 3: ### Summary Il modulo VideoMAE adatta VideoMAE v2 allo studio dei medicanes s...
Frase 1: "### Summary
Il modulo VideoMAE adatta VideoMAE v2 allo studio dei medicanes su sequenze video satellitari AirmassRGB (SEVIRI MSG)." -> ATBD_final: ATBD_final > 1. Scopo e contesto > Summary
Frase 2: "Il sistema integra un pretraining auto-supervisionato di tipo masked autoencoding su video e due fasi supervisionate orientate rispettivamente alla classificazione cyclone/no-cyclone e al tracking del centro ciclone come regressione di coordinate." -> ATBD_final: ATBD_final > 1. Scopo e contesto > Summary
Frase 3: "L'architettura viene quindi impiegata lungo un flusso coerente che copre la costruzione dei dataset, il training distribuito, l'inferenza e una verifica che combina metriche quantitative e ispezioni qualitative su scala mediterranea." -> ATBD_final: ATBD_final > 1. Scopo e contesto > Summary

Paragrafo 4: ### Introduzione, scopo e contesto del prodotto Lo scopo del prodotto e' forn...
Frase 1: "### Introduzione, scopo e contesto del prodotto
Lo scopo del prodotto e' fornire una base algoritmica solida per rilevare e localizzare i cicloni mediterranei a partire da sequenze satellitari, mantenendo coerenza con le pratiche di analisi di eventi rari in meteorologia." -> ATBD_final: ATBD_final > 1. Scopo e contesto > Introduzione, scopo e contesto del prodotto
Frase 2: "Il contesto operativo e' definito da immagini AirmassRGB compatibili con la griglia geografica adottata (1290x420) e con la logica di tiling usata nella costruzione dei dataset; entro questi vincoli, il sistema permette di derivare un output interpretabile sia come decisione binaria di presenza del ciclone sia come stima spaziale del centro." -> ATBD_final: ATBD_final > 1. Scopo e contesto > Introduzione, scopo e contesto del prodotto
Frase 3: "Il pubblico di riferimento include ricercatori e operatori con competenze meteorologiche che necessitano di strumenti riproducibili per detection e tracking, affiancati da visualizzazioni e metriche piu' adatte a dataset sbilanciati." -> ATBD_final: ATBD_final > 1. Scopo e contesto > Introduzione, scopo e contesto del prodotto

Paragrafo 5: ## 2. Panoramica prodotto, definizioni e specifiche
Frase 1: "## 2." -> ATBD_final: ATBD_final > 2. Panoramica prodotto, definizioni e specifiche
Frase 2: "Panoramica prodotto, definizioni e specifiche." -> ATBD_final: ATBD_final > 2. Panoramica prodotto, definizioni e specifiche

Paragrafo 6: ### Target variable Il prodotto fornisce tre livelli di output, ciascuno lega...
Frase 1: "### Target variable
Il prodotto fornisce tre livelli di output, ciascuno legato a un obiettivo differente ma integrato nella stessa pipeline." -> ATBD_final: ATBD_final > 2. Panoramica prodotto, definizioni e specifiche > Target variable
Frase 2: "Nel pretraining, il target e' la ricostruzione di patch mascherate, un compito auto-supervisionato che spinge il modello a catturare pattern spaziotemporali coerenti." -> ATBD_final: ATBD_final > 2. Panoramica prodotto, definizioni e specifiche > Target variable
Frase 3: "Nella classificazione supervisionata, il target e' una label binaria cyclone/no-cyclone associata a ogni clip video." -> ATBD_final: ATBD_final > 2. Panoramica prodotto, definizioni e specifiche > Target variable
Frase 4: "Nel tracking, il target e' una coppia di coordinate (x, y) che rappresenta il centro del ciclone nell'ultimo frame della sequenza e che consente una stima continua della posizione." -> ATBD_final: ATBD_final > 2. Panoramica prodotto, definizioni e specifiche (sintesi)

Paragrafo 7: ### Input data specs, coverage, resolution, cadence Gli input principali sono...
Frase 1: "### Input data specs, coverage, resolution, cadence
Gli input principali sono immagini satellitari AirmassRGB da SEVIRI MSG, con copertura Mediterranea e griglia lat/lon precomputata per immagini 1290x420." -> ATBD_final: ATBD_final > 2. Panoramica prodotto, definizioni e specifiche > Input data specs, coverage, resolution, cadence
Frase 2: "Le immagini vengono suddivise in tile di 224x224 con offset determinati da stride che, nelle visualizzazioni a scala mediterranea, assumono valori tipici di 213 in x e 196 in y." -> ATBD_final: ATBD_final > 2. Panoramica prodotto, definizioni e specifiche > Input data specs, coverage, resolution, cadence
Frase 3: "Le sequenze di lavoro sono clip da 16 frame; nel tracking ogni clip rappresenta una finestra di circa 80 minuti con ultimo frame allineato all'ora piena, in modo da rendere coerente il legame tra sequenza e target spaziale." -> ATBD_final: ATBD_final > 2. Panoramica prodotto, definizioni e specifiche > Input data specs, coverage, resolution, cadence
Frase 4: "I formati di input sono CSV con righe che puntano ai video o alle cartelle di frame; per il pretraining i record seguono la sintassi `video_path, 0, -1` oppure `frame_folder_path, start_index, total_frames`, con una variante distribuita che aggiunge un campo extra, mentre nel supervisionato il CSV riporta la lista dei frame, gli offset di tile, la label, le coordinate del centro e i metadati temporali." -> ATBD_final: ATBD_final > 2. Panoramica prodotto, definizioni e specifiche > Input data specs, coverage, resolution, cadence

Paragrafo 8: ### Output variables, units, valid ranges, griglia/risoluzione, convenzioni L...
Frase 1: "### Output variables, units, valid ranges, griglia/risoluzione, convenzioni
L'output di classificazione e' una label binaria a livello clip, a cui possono essere associati punteggi continui per analisi successive." -> ATBD_final: ATBD_final > 2. Panoramica prodotto, definizioni e specifiche > Output variables, units, valid ranges, griglia/risoluzione, convenzioni
Frase 2: "L'output di tracking e' una stima (x, y) in pixel relativi alla tile, convertibile in lat/lon tramite la griglia georeferenziata e in distanza geodetica in km mediante la formula dell'haversine." -> ATBD_final: ATBD_final > 2. Panoramica prodotto, definizioni e specifiche > Output variables, units, valid ranges, griglia/risoluzione, convenzioni
Frase 3: "Le convenzioni garantiscono la coerenza tra dominio immagine e dominio geografico: l'asse Y viene invertito per allineare la rappresentazione pixel alla latitudine, e le coordinate globali si ottengono sommando gli offset di tile alle coordinate relative prima della conversione geografica." -> ATBD_final: ATBD_final > 2. Panoramica prodotto, definizioni e specifiche > Output variables, units, valid ranges, griglia/risoluzione, convenzioni

Paragrafo 9: ### Quality flags / confidence indicators Il principale indicatore di qualita...
Frase 1: "### Quality flags / confidence indicators
Il principale indicatore di qualita' documentato e' l'indice medio di nuvolosita' per video, usato per separare dataset cloudy e clear-sky con soglie tipiche superiori a 0.2." -> ATBD_final: ATBD_final > 2. Panoramica prodotto, definizioni e specifiche > Quality flags / confidence indicators
Frase 2: "In fase di visualizzazione mediterranea e di post-processing, le tile mancanti vengono espanse e marcate con flag di riempimento per distinguere i dati reali dalle ricostruzioni e per evitare interpretazioni spurie nelle animazioni o nei mosaici finali." -> ATBD_final: ATBD_final > 2. Panoramica prodotto, definizioni e specifiche > Quality flags / confidence indicators

Paragrafo 10: ## 3. Input data & pre-processing
Frase 1: "## 3." -> ATBD_final: ATBD_final > 3. Input data & pre-processing
Frase 2: "Input data & pre-processing." -> ATBD_final: ATBD_final > 3. Input data & pre-processing

Paragrafo 11: ### Dati di input I dati di input comprendono le sequenze AirmassRGB e le tra...
Frase 1: "### Dati di input
I dati di input comprendono le sequenze AirmassRGB e le tracce Manos `TRACKS_CL*.dat` (classi CL2-CL10) con coordinate geografiche e pressione, integrate quando necessario con tracce ERA5 per associare nomi noti ai medicanes." -> ATBD_final: ATBD_final > 5. Fase di training (sintesi)
Frase 2: "Sono inoltre presenti file di supporto per l'aggiornamento delle finestre temporali, come `new_cyc_limits.csv` e `more_medicanes_time_updated.csv`, che definiscono intervalli piu' coerenti con le fasi di rotazione evidente dei cicloni." -> ATBD_final: ATBD_final > 3. Input data & pre-processing > Dati di input

Paragrafo 12: ### Pre-processing e labeling Il pre-processing inizia con l'unificazione del...
Frase 1: "### Pre-processing e labeling
Il pre-processing inizia con l'unificazione delle tracce di ciclone provenienti da classi differenti, la normalizzazione degli identificativi e la conservazione di variabili come la pressione." -> ATBD_final: ATBD_final > 3. Input data & pre-processing (sintesi)
Frase 2: "Il dominio viene poi ristretto al Mediterraneo tramite selezioni spaziali, e la durata dei cicloni e' analizzata per individuare subset piu' affidabili." -> ATBD_final: ATBD_final > 3. Input data & pre-processing > Pre-processing e labeling
Frase 3: "Le coordinate lat/lon vengono trasformate in pixel usando una griglia georeferenziata coerente con la risoluzione 1290x420; questa fase include il ribaltamento dell'asse Y per mantenere la consistenza tra sistema geografico e immagine, e produce CSV con coordinate `x_pix` e `y_pix`." -> ATBD_final: ATBD_final > 3. Input data & pre-processing > Pre-processing e labeling

Paragrafo 13: A partire da queste tracce si costruisce un master dataframe supervisionato c...
Frase 1: "A partire da queste tracce si costruisce un master dataframe supervisionato che, per ogni frame, include path dell'immagine, offset della tile, coordinate del centro e label." -> ATBD_final: ATBD_final > 3. Input data & pre-processing > Pre-processing e labeling
Frase 2: "Il master viene poi segmentato in gruppi temporali contigui, raggruppato per offset di tile e trasformato in clip da 16 frame, che vengono salvati su disco insieme ai CSV finali di train, val e test." -> ATBD_final: ATBD_final > 3. Input data & pre-processing > Pre-processing e labeling
Frase 3: "Il flusso ammette strategie di bilanciamento tra esempi positivi e negativi, split temporali e split per identificativi di ciclone, inclusi dataset specifici per CL10 o per medicanes nominati." -> ATBD_final: ATBD_final > 3. Input data & pre-processing > Pre-processing e labeling

Paragrafo 14: Un passaggio importante e' il relabeling temporale, che ricalcola la label in...
Frase 1: "Un passaggio importante e' il relabeling temporale, che ricalcola la label in funzione della distanza dall'intervallo di validita' del ciclone, ad esempio escludendo i frame oltre 12 ore dal centro." -> ATBD_final: ATBD_final > 3. Input data & pre-processing > Pre-processing e labeling
Frase 2: "Questa logica produce dataset piu' coerenti con l'obiettivo meteorologico di catturare la fase attiva del ciclone." -> ATBD_final: ATBD_final > 3. Input data & pre-processing > Pre-processing e labeling
Frase 3: "Sono inoltre presenti pipeline per la generazione di dataset full-year che includono periodi senza cicloni, cosi da valutare il comportamento su distribuzioni realistiche e altamente sbilanciate." -> ATBD_final: ATBD_final > 3. Input data & pre-processing > Pre-processing e labeling

Paragrafo 15: L'indice di nuvolosita' viene calcolato sia a livello frame sia a livello vid...
Frase 1: "L'indice di nuvolosita' viene calcolato sia a livello frame sia a livello video e consente la creazione di dataset cloudy e clear-sky." -> ATBD_final: ATBD_final > 3. Input data & pre-processing > Pre-processing e labeling
Frase 2: "Questa informazione e' utile per stratificare la validazione e per verificare se le prestazioni degradano in condizioni di copertura nuvolosa elevata." -> ATBD_final: ATBD_final > 3. Input data & pre-processing > Pre-processing e labeling
Frase 3: "La generazione dei dataset e' inoltre organizzata da un dispatcher che seleziona una sola pipeline per esecuzione, con modalita' dedicate a dataset supervisionati, relabeling, cloud index, tracking o full-year; questa scelta rende il flusso piu' chiaro ma limita la combinazione di opzioni in una singola run." -> ATBD_final: ATBD_final > 3. Input data & pre-processing > Pre-processing e labeling
Frase 4: "Infine, per il tracking si costruiscono dataset dedicati che includono solo esempi positivi e associano a ogni clip le coordinate del centro sul frame finale." -> ATBD_final: ATBD_final > 3. Input data & pre-processing > Pre-processing e labeling

Paragrafo 16: ## 4. Algoritmo
Frase 1: "## 4." -> ATBD_final: ATBD_final > 4. Algoritmo
Frase 2: "Algoritmo." -> ATBD_final: ATBD_final > 4. Algoritmo

Paragrafo 17: ### Overview Il nucleo dell'algoritmo e' un backbone video basato su trasform...
Frase 1: "### Overview
Il nucleo dell'algoritmo e' un backbone video basato su trasformatori, inizialmente specializzato con masked autoencoding per apprendere rappresentazioni spaziotemporali robuste in assenza di etichette." -> ATBD_final: ATBD_final > 4. Algoritmo > Overview
Frase 2: "Il masked autoencoding introduce un vincolo di ricostruzione di patch mascherate che obbliga il modello a inferire il contenuto mancante dalle strutture contestuali, un meccanismo utile in meteorologia per catturare pattern dinamici e graduali." -> ATBD_final: ATBD_final > 4. Algoritmo > Overview
Frase 3: "La base specializzata viene poi fine-tunata per due compiti distinti: la classificazione binaria e il tracking regressivo del centro, ottenuto sostituendo la head di classificazione con un regressore a due dimensioni." -> ATBD_final: ATBD_final > 4. Algoritmo > Overview

Paragrafo 18: ### Flowchart (testuale) ``` AirmassRGB frames + Manos tracks   -> costruzion...
Frase 1: "### Flowchart (testuale)
```
AirmassRGB frames + Manos tracks
  -> costruzione master dataframe (offset, label, x/y pixel)
  -> clip video (16 frame) + CSV train/val/test
     -> (A) pretraining MAE su clip non etichettati
     -> (B) fine-tuning classificazione su clip etichettati
     -> (C) fine-tuning tracking (solo tile positive)
  -> inferenza + post-processing
  -> validazione quantitativa + visualizzazioni
```." -> ATBD_final: ATBD_final > 4. Algoritmo > Flowchart (testuale)

Paragrafo 19: ### Pseudocodice (sintetico) **Pretraining / specialization** ``` INPUT: clip...
Frase 1: "### Pseudocodice (sintetico)
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
```." -> ATBD_final: ATBD_final > 6. Verifica: validazione, test, incertezze, limiti (sintesi)

Paragrafo 20: **Classificazione** ``` INPUT: clip etichettati (train/val/test), checkpoint ...
Frase 1: "**Classificazione**
```
INPUT: clip etichettati (train/val/test), checkpoint iniziale opzionale
SETUP: risorse, logging, scalatura iperparametri
MODEL: backbone video + head binaria
FOR ogni epoca:
  training su train
  validazione periodica
  aggiorna best checkpoint
OUTPUT: best checkpoint + log
```." -> ATBD_final: ATBD_final > 4. Algoritmo > Pseudocodice (sintetico)

Paragrafo 21: **Tracking** ``` INPUT: clip positive con target (x, y) dell'ultimo frame SET...
Frase 1: "**Tracking**
```
INPUT: clip positive con target (x, y) dell'ultimo frame
SETUP: risorse, logging
MODEL: backbone video + regressore 2D
FOR ogni epoca:
  training con perdita MSE
  valutazione su test/val
  aggiorna best checkpoint
OUTPUT: best checkpoint tracking + log
```." -> ATBD_final: ATBD_final > 4. Algoritmo > Pseudocodice (sintetico)

Paragrafo 22: **Inferenza classificazione** ``` INPUT: dataset di test/val, checkpoint MODE...
Frase 1: "**Inferenza classificazione**
```
INPUT: dataset di test/val, checkpoint
MODEL: carica modello
IF richiesta logits/embedding:
  raccogli shard multi-rank, merge, cleanup
ELSE:
  raccogli predizioni standard
OUTPUT: CSV predizioni + metriche
```." -> ATBD_final: ATBD_final > 4. Algoritmo > Pseudocodice (sintetico)

Paragrafo 23: ### Formule, assunzioni, semplificazioni Il tracking minimizza una loss MSE t...
Frase 1: "### Formule, assunzioni, semplificazioni
Il tracking minimizza una loss MSE tra coordinate predette e target (x, y), coerente con l'obiettivo di ridurre l'errore spaziale continuo." -> ATBD_final: ATBD_final > 4. Algoritmo > Formule, assunzioni, semplificazioni
Frase 2: "Le metriche di classificazione sono quelle tipiche dei problemi di eventi rari in meteorologia, per cui la sola accuracy e' fuorviante in presenza di forte sbilanciamento." -> ATBD_final: ATBD_final > 4. Algoritmo > Formule, assunzioni, semplificazioni
Frase 3: "Le formule usate si basano sulla confusion matrix con H (hits), M (misses), F (false alarms) e C (correct negatives) e includono:." -> ATBD_final: ATBD_final > 4. Algoritmo > Formule, assunzioni, semplificazioni

Paragrafo 24: POD = H / (H + M) FAR = F / (H + F) CSI = H / (H + F + M) HSS = 2(H*C - M*F) ...
Frase 1: "POD = H / (H + M)
FAR = F / (H + F)
CSI = H / (H + F + M)
HSS = 2(H*C - M*F) / [(H+M)(M+C) + (H+F)(F+C)]
BA = 0.5 * [H/(H+M) + C/(C+F)]." -> ATBD_final: ATBD_final > 4. Algoritmo > Formule, assunzioni, semplificazioni

Paragrafo 25: Per il tracking, l'errore in km e' derivato passando dalle coordinate pixel a...
Frase 1: "Per il tracking, l'errore in km e' derivato passando dalle coordinate pixel a lat/lon tramite la griglia georeferenziata e applicando la formula dell'haversine con raggio medio terrestre 6371.0088 km." -> ATBD_final: ATBD_final > 4. Algoritmo > Formule, assunzioni, semplificazioni
Frase 2: "Le assunzioni principali includono l'uso di soli esempi positivi per il tracking e la scelta del frame finale come target, una semplificazione temporale che rende il problema ben definito rispetto alla finestra di osservazione." -> ATBD_final: ATBD_final > 4. Algoritmo > Formule, assunzioni, semplificazioni
Frase 3: "Un'altra assunzione fondamentale e' la validita' della griglia lat/lon per immagini 1290x420; se la risoluzione cambia, la griglia deve essere rigenerata per mantenere la coerenza delle conversioni." -> ATBD_final: ATBD_final > 4. Algoritmo > Formule, assunzioni, semplificazioni

Paragrafo 26: Le motivazioni operative di queste scelte sono legate alla disponibilita' del...
Frase 1: "Le motivazioni operative di queste scelte sono legate alla disponibilita' delle etichette e alla necessita' di un target stabile nel tempo." -> ATBD_final: ATBD_final > 2. Panoramica prodotto, definizioni e specifiche (sintesi)
Frase 2: "In un contesto meteorologico, fissare il target al frame finale consente di confrontare in modo riproducibile la posizione stimata con l'osservazione piu' recente, mentre l'uso di soli casi positivi evita di introdurre rumore nei target quando il ciclone non e' presente." -> ATBD_final: ATBD_final > 6. Verifica: validazione, test, incertezze, limiti (sintesi)
Frase 3: "Le equazioni della loss di classificazione e della loss MAE non sono esplicitate nei documenti del modulo e restano quindi non formalizzate in questo ATBD." -> ATBD_final: ATBD_final > 4. Algoritmo > Formule, assunzioni, semplificazioni

Paragrafo 27: ## 5. Fase di training
Frase 1: "## 5." -> ATBD_final: ATBD_final > 5. Fase di training
Frase 2: "Fase di training." -> ATBD_final: ATBD_final > 5. Fase di training

Paragrafo 28: Il training segue due percorsi complementari. Nel pretraining, il modello imp...
Frase 1: "Il training segue due percorsi complementari." -> ATBD_final: ATBD_final > 5. Fase di training
Frase 2: "Nel pretraining, il modello impara a ricostruire patch mascherate su clip non etichettati, con l'obiettivo di specializzarsi sulle dinamiche dei medicanes senza imporre vincoli di classe." -> ATBD_final: ATBD_final > 5. Fase di training
Frase 3: "Nel supervisionato, la classificazione ottimizza la separazione cyclone/no-cyclone, mentre il tracking ottimizza l'errore MSE sulle coordinate del centro; in entrambi i casi vengono creati set di train, validation e test, con possibilita' di bilanciamento o di split per identita' di ciclone." -> ATBD_final: ATBD_final > 5. Fase di training
Frase 4: "La loss di classificazione non e' specificata nei documenti e va considerata non documentata." -> ATBD_final: ATBD_final > 5. Fase di training

Paragrafo 29: I principali iperparametri documentati riguardano il pretraining: mascheramen...
Frase 1: "I principali iperparametri documentati riguardano il pretraining: mascheramento con `mask_ratio=0.9` e `decoder_mask_ratio=0.5`, decoder depth 4, patch 14 e input 224, clip di 16 frame con sampling rate 4 e num_sample 4, batch size 32, learning rate 6e-4, warmup di 30 epoche, gradient clipping a 0.02, training per 300 epoche e checkpoint ogni 5 epoche." -> ATBD_final: ATBD_final > 5. Fase di training
Frase 2: "Sul versante dei dataset, la soglia tipica dell'indice di nuvolosita' e' 0.2, mentre gli stride per ricostruire il mosaico mediterraneo sono 213 in x e 196 in y." -> ATBD_final: ATBD_final > 5. Fase di training
Frase 3: "Questi parametri sono riportati come esempi operativi piu' che come configurazioni ottimali." -> ATBD_final: ATBD_final > 5. Fase di training

Paragrafo 30: I dettagli computazionali che influenzano la convergenza includono la scalatu...
Frase 1: "I dettagli computazionali che influenzano la convergenza includono la scalatura di batch size e learning rate con il world size nel training distribuito, l'uso di scheduler coseno per learning rate e weight decay, l'applicazione di una modulazione layer-wise del learning rate e l'uso di mixed precision con gradient scaling." -> ATBD_final: ATBD_final > 5. Fase di training
Frase 2: "Il checkpoint del best modello viene ritardato di un numero di epoche per evitare selezioni premature, e le procedure di impostazione dei seed sono impiegate per aumentare la riproducibilita' tra run." -> ATBD_final: ATBD_final > 5. Fase di training
Frase 3: "La presenza di log in JSON lines e di checkpoint periodici permette di ricostruire l'andamento e di riprendere il training in caso di interruzione." -> ATBD_final: ATBD_final > 5. Fase di training

Paragrafo 31: Practical notes relative alle prestazioni includono esempi di training distri...
Frase 1: "Practical notes relative alle prestazioni includono esempi di training distribuito su cluster multi-nodo con configurazioni dell'ordine di decine di GPU e l'uso di pipeline di rendering video per le visualizzazioni finali." -> ATBD_final: ATBD_final > 5. Fase di training
Frase 2: "Questi aspetti migliorano la scalabilita' ma non modificano i risultati scientifici." -> ATBD_final: ATBD_final > 5. Fase di training

Paragrafo 32: ## 6. Verifica: validazione, test, incertezze, limiti
Frase 1: "## 6." -> ATBD_final: ATBD_final > 6. Verifica: validazione, test, incertezze, limiti
Frase 2: "Verifica: validazione, test, incertezze, limiti." -> ATBD_final: ATBD_final > 6. Verifica: validazione, test, incertezze, limiti

Paragrafo 33: La verifica quantitativa combina la confusion matrix con metriche specifiche ...
Frase 1: "La verifica quantitativa combina la confusion matrix con metriche specifiche per eventi rari, in modo da ridurre l'impatto della forte asimmetria tra casi positivi e negativi." -> ATBD_final: ATBD_final > 6. Verifica: validazione, test, incertezze, limiti
Frase 2: "Le metriche adottate sono POD, FAR, CSI, HSS e Balanced Accuracy, con formule esplicitate nella sezione precedente; esse permettono di valutare simultaneamente la capacita' di rilevare gli eventi, il tasso di falsi allarmi e la performance rispetto a un baseline casuale." -> ATBD_final: ATBD_final > 6. Verifica: validazione, test, incertezze, limiti
Frase 3: "Per il tracking, l'errore e' espresso anche in km tramite conversione geodetica, cosi da fornire una misura fisicamente interpretabile." -> ATBD_final: ATBD_final > 6. Verifica: validazione, test, incertezze, limiti

Paragrafo 34: Il workflow di validazione prevede la costruzione di set di test e validation...
Frase 1: "Il workflow di validazione prevede la costruzione di set di test e validation attraverso filtri temporali che escludono frame troppo lontani dal ciclone, e la generazione di dataset bilanciati, sbilanciati o annuali completi." -> ATBD_final: ATBD_final > 6. Verifica: validazione, test, incertezze, limiti
Frase 2: "Le analisi includono il calcolo di accuracy, FPR, FNR, POD, FAR e confusion matrix, oltre a valutazioni per label temporali che mettono in relazione la performance con la distanza dall'evento." -> ATBD_final: ATBD_final > 6. Verifica: validazione, test, incertezze, limiti
Frase 3: "I log storici vengono inoltre impiegati per confrontare esperimenti diversi e analizzare la sensibilita' agli iperparametri." -> ATBD_final: ATBD_final > 6. Verifica: validazione, test, incertezze, limiti

Paragrafo 35: La verifica qualitativa comprende la generazione di animazioni mediterranee c...
Frase 1: "La verifica qualitativa comprende la generazione di animazioni mediterranee con overlay delle predizioni, la visualizzazione delle tile e delle traiettorie del centro ciclone, e il rendering di GIF o MP4 per cicloni specifici." -> ATBD_final: ATBD_final > 6. Verifica: validazione, test, incertezze, limiti
Frase 2: "Nel caso del pretraining, la qualita' delle ricostruzioni viene verificata tramite confronti visivi tra input e output e tramite l'analisi delle maschere di patching." -> ATBD_final: ATBD_final > 6. Verifica: validazione, test, incertezze, limiti
Frase 3: "Queste procedure non sostituiscono la validazione quantitativa, ma forniscono un controllo visivo della coerenza spaziale e temporale delle predizioni." -> ATBD_final: ATBD_final > 6. Verifica: validazione, test, incertezze, limiti

Paragrafo 36: Le principali limitazioni documentate riguardano l'assenza di criteri formali...
Frase 1: "Le principali limitazioni documentate riguardano l'assenza di criteri formali di early stopping, la mancanza di analisi di incertezza statistica e l'assenza di soglie di accettazione quantitative." -> ATBD_final: ATBD_final > 6. Verifica: validazione, test, incertezze, limiti
Frase 2: "Inoltre, alcune scelte sono hard-coded, come la disattivazione del mixup in classificazione e la non attivazione di una loss pesata, mentre le pipeline di costruzione dataset utilizzano flag mutuamente esclusivi che impediscono combinazioni di flussi." -> ATBD_final: ATBD_final > 6. Verifica: validazione, test, incertezze, limiti
Frase 3: "Le conversioni geografiche restano valide solo per la griglia 1290x420 e l'indice di nuvolosita' si basa su scelte di canale colore che richiedono ulteriori verifiche." -> ATBD_final: ATBD_final > 6. Verifica: validazione, test, incertezze, limiti
Frase 4: "L'entry point della specialization necessita di argomenti espliciti e non risulta avviabile in modo standalone senza una corretta configurazione dei parametri." -> ATBD_final: ATBD_final > 6. Verifica: validazione, test, incertezze, limiti

Paragrafo 37: ## 7. Provenance (obbligatoria, compatta)
Frase 1: "## 7." -> ATBD_final: ATBD_final > 7. Provenance (obbligatoria, compatta)
Frase 2: "Provenance (obbligatoria, compatta)." -> ATBD_final: ATBD_final > 7. Provenance (obbligatoria, compatta)

Paragrafo 38: ### Upstream e licenza Il modulo deriva da VideoMAE v2 e mantiene riferimenti...
Frase 1: "### Upstream e licenza
Il modulo deriva da VideoMAE v2 e mantiene riferimenti concettuali a VideoMAE v1." -> ATBD_final: ATBD_final > 7. Provenance (obbligatoria, compatta) > Upstream e licenza
Frase 2: "La licenza upstream non e' riportata nei documenti del modulo e va verificata nel repository di origine." -> ATBD_final: ATBD_final > 7. Provenance (obbligatoria, compatta) > Upstream e licenza

Paragrafo 39: ### Modifiche locali (upstream vs ours) L'adattamento locale introduce un dom...
Frase 1: "### Modifiche locali (upstream vs ours)
L'adattamento locale introduce un dominio applicativo specifico ai medicanes su immagini AirmassRGB, una pipeline di costruzione dataset con tiling e clip da 16 frame, e l'integrazione delle tracce Manos con aggiornamento delle finestre temporali." -> ATBD_final: ATBD_final > 7. Provenance (obbligatoria, compatta) > Modifiche locali (upstream vs ours)
Frase 2: "Viene aggiunto un percorso di tracking del centro con regressione a due coordinate e una conversione sistematica degli errori in km." -> ATBD_final: ATBD_final > 7. Provenance (obbligatoria, compatta) > Modifiche locali (upstream vs ours)
Frase 3: "La pipeline include la stima della nuvolosita' per filtrare dataset cloudy e clear-sky, oltre a flussi di inferenza e post-processing per mosaici mediterranei." -> ATBD_final: ATBD_final > 7. Provenance (obbligatoria, compatta) > Modifiche locali (upstream vs ours)
Frase 4: "Sono presenti notebook per validazione quantitativa, analisi temporale e visualizzazioni di case study, insieme a un supporto esplicito al training distribuito e alla scalatura degli iperparametri." -> ATBD_final: ATBD_final > 7. Provenance (obbligatoria, compatta) > Modifiche locali (upstream vs ours)
Frase 5: "La fase di pretraining e' accompagnata da strumenti di verifica delle maschere e delle ricostruzioni." -> ATBD_final: ATBD_final > 7. Provenance (obbligatoria, compatta) > Modifiche locali (upstream vs ours)

Paragrafo 40: ### Entrypoint, layout del repo, config Gli entrypoint principali coprono pre...
Frase 1: "### Entrypoint, layout del repo, config
Gli entrypoint principali coprono pretraining, classificazione, tracking e inferenza, mentre la pipeline dati include strumenti per la costruzione dei dataset e per il tracking." -> ATBD_final: ATBD_final > 7. Provenance (obbligatoria, compatta) > Entrypoint, layout del repo, config
Frase 2: "Il repository e' organizzato con una cartella `docs/` per notebook e guide, `misc/` per note operative e `scripts/` per job di training, e produce output standardizzati come `log.txt` e checkpoint best." -> ATBD_final: ATBD_final > 7. Provenance (obbligatoria, compatta) > Entrypoint, layout del repo, config
Frase 3: "La configurazione si basa su parametri via CLI e su CSV di input per train, validation e test." -> ATBD_final: ATBD_final > 7. Provenance (obbligatoria, compatta) > Entrypoint, layout del repo, config

Paragrafo 41: ## 8. References bibliografiche Le reference principali sono VideoMAE v1 (Neu...
Frase 1: "## 8." -> ATBD_final: ATBD_final > 8. References bibliografiche
Frase 2: "References bibliografiche
Le reference principali sono VideoMAE v1 (NeurIPS 2022, https://arxiv.org/abs/2203.12602), VideoMAE v2 (CVPR 2023, https://arxiv.org/abs/2303.16727) e il repository upstream VideoMAE v2 (https://github.com/OpenGVLab/VideoMAEv2)." -> ATBD_final: ATBD_final > 8. References bibliografiche
