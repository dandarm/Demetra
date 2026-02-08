# ATBD DeMeTrA

## Scopo, contesto e summary
Questo documento descrive la base teorica e la catena di produzione del prodotto DeMeTrA, un algoritmo di deep learning per il rilevamento e il tracciamento dei medicanes, cioe cicloni mediterranei con caratteristiche simil tropicali. Il contesto operativo e il monitoraggio near real time di fenomeni rari, in cui la disponibilita di immagini geostazionarie ad alta cadenza consente di seguire l'evoluzione rapida dei sistemi convettivi. L'obiettivo e fornire uno strumento per individuare la presenza di cicloni, localizzare il centro di rotazione e riconoscere precocemente l'evoluzione verso sistemi tropical-like con occhio chiuso.

DeMeTrA sfrutta l'imaging infrarosso del sensore Spinning Enhanced Visible InfraRed Imager, SEVIRI, a bordo di Meteosat Second Generation, MSG. Le immagini sono convertite in compositi Airmass RGB, riorganizzate in clip video e usate per specializzare un modello VideoMAEv2 pre-addestrato. Il processo prevede un addestramento auto-supervisionato di specializzazione e due addestramenti supervisionati, uno per la classificazione binaria presenza o assenza di ciclone e uno per la regressione della posizione del centro di rotazione. La detection e preparatoria al tracking, perche consente di filtrare i clip e ridurre il carico computazionale della regressione. Il prodotto e sviluppato nel quadro delle attivita WP2400 per la detection, il monitoraggio e il tracking dei medicanes con dati geostazionari e output utilizzabili in contesto operativo ESA.

## Panoramica prodotto, definizioni e specifiche
Il prodotto DeMeTrA stima due variabili target. La prima e un indicatore di presenza di ciclone mediterraneo in un clip video, ottenuto tramite classificazione binaria. La seconda e la coordinata del centro di rotazione, RC, prodotta con una regressione su ciascun clip video contenente un ciclone. La definizione operativa di ciclone segue i cataloghi TRACKS_CL7, con successive rifiniture basate su ispezione visiva e selezione di medicanes. TRACKS_CL7 deriva dalla combinazione di sette metodi di tracking applicati a reanalisi, quindi rappresenta un riferimento coerente ma non esente da mismatch rispetto a segnali IR.

Gli input minimi sono immagini IR del servizio Rapid Scan Service con cadenza di 5 minuti e risoluzione spaziale di circa 3 km al sub-satellite point. Le immagini sono fornite in formato Zarr, in proiezione geostazionaria non riproiettata. La catena necessita dei canali IR_097, IR_108, WV_062 e WV_073 per la costruzione del composito Airmass RGB. Le immagini vengono ritagliate sull'area mediterranea, con latitudine compresa tra 30 gradi e 48 gradi Nord e longitudine tra -7 e 46 gradi, e risultano in un frame di 1290 per 420 pixel.

Fig. 5.2 – Air mass RGB example image covering the Mediterranean basin, with superimposed borders. Image time: 17th September 2020, 03:40 UTC

Questi frame sono poi suddivisi in tile di 224 per 224 pixel, parzialmente sovrapposti, e impilati temporalmente in blocchi di 16 frame, corrispondenti a una finestra temporale di 80 minuti. L'overlap temporale consente un campionamento orario dei clip.

L'output della fase di detection e un'etichetta binaria per ciascun video tile con uno score di classificazione che, in un contesto operativo, puo essere interpretato come indice di confidenza. L'output della fase di tracking e una coppia di coordinate del centro di rotazione nel dominio dei pixel del tile. La conversione da coordinate pixel a coordinate geografiche richiede l'inversione della trasformazione applicata durante il crop e il tiling. Il prodotto deve inoltre includere un flag di copertura dati e un indicatore di qualita, per distinguere le situazioni con input incompleti o con segnali IR non coerenti con la dinamica ciclonica.

## Input data e pre-processing
L'addestramento usa dati IR di EUMETSAT Rapid Scan High Rate SEVIRI Level 1.5 Image Data MSG, con alta frequenza di 5 minuti e alta risoluzione spaziale, utili per monitorare fenomeni meteorologici in rapida evoluzione. I dati sono scaricabili da Google Cloud Big Query Public Data, con permesso di redistribuzione, e Google ospita dataset pubblici e di terze parti per conto dei provider, offrendo accesso affidabile e su larga scala senza oneri di storage. I dati sono resi disponibili da Open Climate Fix, che li ha processati ulteriormente usando le librerie satip e SatPy. Il dataset disponibile non contiene i valori numerici originali, perche e stato calibrato con SatPy e linearmene normalizzato per produrre Brightness Temperature, con valori mappati nell'intervallo [0, 1023], cioe 10 bit per canale. Il dataset e in formato Zarr e non e stato riproiettato, quindi rimane in proiezione geostazionaria.

Il dataset include anche le etichette di occorrenza dei medicanes basate sull'output WP2300 e le coordinate del centro di rotazione ottenute da osservazioni indipendenti, oltre alle tracce modellistiche disponibili per tutti i cicloni considerati come riferimento.

Il recupero dei valori fisici richiede la de-normalizzazione secondo la formula BT = v (xmax - xmin) + xmin, dove v e il valore normalizzato in 0 a 1 e xmax e xmin sono specifici per canale. La tabella seguente riporta i valori in uso per i canali richiesti.

```text
SEVIRI channel    Xmin      Xmax
IR_097     2,84         317,87
IR_108     199,10       313,28
WV_062     199,57       249,92
WV_073     198,95       286,96
```

Table 5.1 – Maximum and minimum values for the channels of interest.

I canali infrarossi e di vapor d'acqua sono combinati in un composito Airmass RGB secondo le linee guida EUMETSAT, RGB Recipes e Best Practices, con riferimento a https://eumetrain.org/sites/default/files/2020-05/RGB_recipes.pdf e https://www-cdn.eumetsat.int/files/2020-04/pdf_using_rgb_best_practices.pdf. Il canale rosso e costruito come differenza tra il canale infrarosso di vapor d'acqua intorno a 6.2 µm, WV_062, e il canale di vapor d'acqua 7.3 µm, WV_073. Il canale verde e ottenuto dalla differenza tra i canali sensibili all'assorbimento dell'ozono IR_097 e IR_108, che evidenziano le intrusioni stratosferiche. Il canale blu e WV_062. Questa combinazione rende visibili masse d'aria, umidita e sistemi frontali, facilitando l'identificazione visiva della dinamica ciclonica e delle strutture cloud. Nel flusso operativo, questo composito e usato per costruire l'intero dataset. R = WV_062 – WV_073, G = IR_097 – IR_108, B = WV_062.

Il processo di dataset building parte dal presupposto che i modelli di grandi dimensioni richiedono grandi dataset. Un dataset precedente di Ifremer risultava non completo per mancanza di continuita temporale e insufficiente per un training adeguato. Per questo si e proceduto al download e alla ricostruzione dei dati, includendo la trasformazione inversa verso valori originali di Brightness Temperature e la costruzione del composito con le corrette finestre di normalizzazione Meteosat. We have the Source Dataset  (fig airmassRGB senza land borders).

Big models need big data. Need to download data. Transform back to original Brightness Temperature values. Build the composite with proper normalization ranges as in the Meteosat recipe.

Il dataset sorgente comprende circa 860K frame AirmassRGB di 1290 X 420 pixel sull'area del Mediterraneo, per un totale di circa 600 GByte, con copertura temporale ⪞ 7.5 years tra 2010 e 2023. Un processo di download e pre-processing e stato implementato con script Python ottimizzati per garantire efficienza, perche la generazione del working dataset deve essere ripetuta per ogni scelta di parametri. Il dataset risultante e denominato Source Dataset, mentre il dataset prodotto da tiling e labeling e denominato Working Dataset.

La regione mediterranea e suddivisa in tile 224 x 224 pixel parzialmente sovrapposti, con un numero tipico di 12 tile, ma variabile in funzione dell'overlap scelto. Ogni tile copre approssimativamente 757 km in latitudine e 805 km in longitudine, con risoluzione di circa 3.38 km per pixel in latitudine e 3.59 km per pixel in longitudine. I tile sono impilati in clip di 16 frame distanziati di 5 minuti, per un totale di 80 minuti, con sovrapposizione temporale e generazione di un video tile ogni ora. Questo formato e richiesto dal modello pre-addestrato per dimensioni spaziali e numero di frame. fig tiles-mediterraneo  - fig video-tiles.
Verifica matematica (da `latcorners = [30, 48]` e `loncorners = [-7, 46]` in `moduli/videomae/medicane_utils/geo_const.py`, con frame 1290 x 420 px): Δlat = 18 deg, Δlon = 53 deg. Con 1 deg lat = 2*pi*R/360 ≈ 111.32 km e 1 deg lon ≈ 111.32*cos(39 deg) = 86.5 km (lat media 39 deg), l'estensione e ~2004 km (N-S) e ~4585 km (E-W). Risoluzione media: 2004/420 ≈ 4.77 km/px in latitudine e 4585/1290 ≈ 3.55 km/px in longitudine. Una tile 224 x 224 copre quindi ~1069 km (lat) e ~796 km (lon).
Verifica via griglia Basemap (`makegrid`): la griglia e in coordinate proiettate, quindi il km/px non e costante. Con la griglia 1290 x 420 si ottiene (mediana, range) ~3.22 km/px (3.07-4.67) in orizzontale e ~4.43 km/px (3.77-7.09) in verticale; una tile 224 x 224 copre ~713 km (687-923) in orizzontale e ~999 km (904-1284) in verticale. Quindi 3.59 km/px in longitudine e plausibile localmente, ma 3.38 km/px e 757 km in latitudine non risultano compatibili con la griglia attuale.

L'etichettatura si basa sul database TracksCL7, con trasformazione delle coordinate del centro di rotazione da geografiche a pixel. La procedura e altamente ottimizzata perche il working dataset viene ricostruito molte volte in funzione delle scelte di training. Un tile e etichettato come positivo quando il centro di rotazione e contenuto nel tile per almeno 6 frame su 16, altrimenti e etichettato come negativo. I tile contenenti il centro sono indicati come label 1 e gli altri come label 0, con rappresentazione grafica del green square e dei punti di traccia. TracksCL7 deriva dalla combinazione di sette metodi di tracking applicati a reanalisi. fig labeled-tiles.

In the following fig 5.3 it is shown the process of tiling and labeling from a source image. Fig. 5.2 – Airmass RGB source frame: the squares are sliced to form video clips in the same position for 16 frames. A green square is shown where the video tile is selected as positive sample, containing a cyclone center track (cyan dots).

## Algoritmo
DeMeTrA e basato su VideoMAEv2, un transformer per video con apprendimento auto-supervisionato. Il modello e open source e pre-addestrato su grandi dataset video, e usa una architettura autoencoder che non richiede etichette nella fase di pre-training. Il modello lavora su clip di 16 frame con dimensioni 224×224 pixel, e ogni frame viene suddiviso in patch non sovrapposte di 14x14 pixel nel forward pass. L'architettura include una struttura encoder-decoder e una strategia di tube masking, in cui patch spatio-temporali sono mascherate per forzare l'apprendimento di rappresentazioni robuste. DeMeTrA specializza VideoMAEv2 per il riconoscimento e tracking di cicloni tramite fine-tuning sul dataset Airmass RGB.

How Transformers Work and Why They Outperform CNNs. La scelta del transformer e motivata da proprieta specifiche di self-attention. I transformer calcolano dipendenze tra tutte le parti di un'immagine o di una sequenza video, permettendo a ogni patch di interagire direttamente con tutte le altre e modellare dipendenze a lungo raggio. Questo consente di apprendere relazioni contestuali dinamiche, invece di filtri statici locali, e di scalare a dataset e modelli piu grandi, come ViT-Large e ViT-Huge, con prestazioni elevate senza overfitting eccessivo. Questa architettura e particolarmente efficace per video understanding, dove le relazioni temporali tra frame sono cruciali. Un riferimento specifico che sintetizza questi aspetti e il lavoro Comparing Vision Transformers and Convolutional Neural Networks for Image Classification: A Literature Review di Maurício, Domingues e Bernardino, con URL https://www.mdpi.com/2225-1154/12/12/220.

VideoMAE: A Transformer-Based Model for Video Understanding. VideoMAE e un approccio di self-supervised learning per video che estende i Masked Autoencoders da immagini a sequenze spatio-temporali. Key Features of VideoMAE. Il modello usa un masking ratio elevato, tipicamente 90-95%, rispetto al 75% delle MAE per immagini, sfruttando la ridondanza temporale nei video. Il masking e di tipo tube masking, cioe maschera cubi spatio-temporali continui, rendendo la ricostruzione piu complessa rispetto al mascheramento casuale di singoli patch. L'architettura e asimmetrica: l'encoder processa solo i token visibili, rendendo l'addestramento efficiente, mentre il decoder ricostruisce i token mancanti con una struttura piu leggera.

Why VideoMAE is Effective for Short Video Classification. VideoMAE e particolarmente efficace per la classificazione di video brevi perche l'auto-supervised pretext task di ricostruzione permette di apprendere rappresentazioni spatio-temporali ricche senza etichette. La scalabilita e dimostrata su dataset come Kinetics-400 e Something-Something V2 e su modelli grandi come ViT-L, ViT-H e ViT-g, con prestazioni di stato dell'arte. La trasferibilita e elevata e consente l'uso del backbone su compiti downstream come video classification, action recognition e spatiotemporal localization.

VideoMAE Architecture Overview. La struttura interna di VideoMAE puo essere descritta come segue. Key Components of VideoMAE. Il modello usa cube embedding per convertire frame in token, con cubi spatio-temporali di dimensione 2 × 16 × 16 pixel, corrispondenti a due frame e una risoluzione di 16×16. Cube Embedding (Patch Tokenization). Ogni patch e embedded in un vettore con position encoding, riducendo la ridondanza spaziale e temporale. Encoder: Masked Spatiotemporal Transformer. Joint Space-Time Attention. Masked Token Strategy. L'encoder e un Vision Transformer con attenzione congiunta nello spazio e nel tempo, e processa solo circa il 10% dei token di input, mentre il resto viene scartato, aumentando l'efficienza. Decoder: Lightweight Reconstruction. Asymmetric Architecture. Il decoder ricostruisce i token mancanti e opera sul 100% dei token ma con profondita e larghezza ridotte. Una configurazione tipica usa 12 blocchi per l'encoder ViT-Base e 4 blocchi per il decoder, con un vantaggio computazionale significativo.

Why Tube Masking? Il tube masking e motivato dal fatto che, nei video, mascherare singoli patch e troppo facile per la ricostruzione a causa della ridondanza temporale. How It Works? Mascherare interi tubi spatio-temporali aumenta la difficolta e forza l'apprendimento di rappresentazioni robuste. Dual Masking in VideoMAE V2. VideoMAEv2 introduce una doppia mascheratura: l'encoder vede solo una frazione dei token, tipicamente con 90% di token rimossi, e il decoder ricostruisce solo un sottoinsieme dei cubi mancanti, riducendo memoria e tempo di training e permettendo di scalare a modelli con miliardi di parametri. Computational Efficiency. Asymmetric Architecture: il decoder piu piccolo riduce il costo computazionale pur mantenendo ricostruzioni di qualita. Efficient Self-Supervised Pre-Training: no labeled data is required—models learn representations by reconstructing missing patches. Scalability: VideoMAE can scale from small models (ViT-B) to billion-parameter models (ViT-g).

(figura VideoMAEv2_flowchart.png)
Fig. 5.1 – Technical details of the VideoMAEv2 model: encoder-decoder structure, with separate masking for each one. The latent representation vector (embedding) is shown between the encoder and the decoder.

La pipeline algoritmica per DeMeTrA integra la catena di pre-processing e i tre stadi di training.

```text
Dati SEVIRI RSS -> De-normalizzazione BT -> Composito Airmass RGB -> Crop Mediterraneo
-> Tiling 224x224 -> Stack 16 frame -> Etichettatura con TRACKS_CL7
-> Specialization training VideoMAE -> Classificazione presenza ciclone
-> Regressione del centro di rotazione
```

```text
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
```

## Fase di training
Three stage training:
Il training e organizzato in tre stadi. Le due fasi supervisionate usano le feature apprese nella fase di specializzazione. Il primo e un addestramento auto-supervisionato di specializzazione su clip Airmass RGB, senza etichette, con circa 80’000 video in training e 20’000 in test. In questa fase si usa il modello pre-addestrato di taglia “giant” per una specializzazione non supervisionata che si aggiunge al pre-training da zero, con l'obiettivo di aumentare la capacita di estrazione delle feature utili ai task successivi. Il training e lungo, dell'ordine di giorni, e la curva di loss di training e validazione non mostra overfitting, indicando una specializzazione efficace, per cui il modello viene salvato come checkpoint per i task successivi. Training and validation losses versus epochs.

Self-supervised dataset: random videos, con train set ~ 80K video tiles e test set ~ 20K video tiles. Long time training (~ days). Results: no overfitting → good! → save this model as a checkpoint for the next stage training.

Il secondo stadio e una classificazione binaria. Il primo trial usa un dataset bilanciato di circa 3000 video, divisi in due classi, con class 1 che include cicloni in TRACKS_CL7, medicanes e altri cicloni mediterranei, e class 0 senza cicloni. Using Tracks CL7: Max. Accuracy 80 %. Need for increased dataset quality: why? because of mismatch tra tracce e contenuto IR, con casi di cloud absence o assenza di rotazione evidente. Mismatch with the tracks: IR images may not capture cyclonic dynamics for certain tracks (cloud absence, no cloud rotation...). Questo ha portato a un percorso di raffinamento del dataset con quattro passi, partendo da TRACKS_CL7, passando a TRACKS_CL10, poi limitando ai soli medicanes e infine restringendo la finestra temporale ai frame con rotazione chiaramente osservabile. Supervised binary classification dataset - 4 refining steps: Step 1: Tracks CL7, Step 2: Tracks CL10, Step 3: Only Medicanes *, Step 4: Only Medicanes * and narrow time window with clearly observable rotation. * From the Full_List_Medicanes.

Dataset refinement: Cyclones new time boundaries (manual selection). Using Only Medicanes *. Le finestre temporali dei medicanes sono state riviste manualmente, con nuovi start e end times basati su ispezione visiva e presenza di rotazione chiaramente visibile. New Start and End times based on visual inspection. Clearly visible cyclone clouds rotation. (medicane_new_windiws.csv).

Il dataset finale per la detection include 18 cicloni, con la lista riportata di seguito in forma tabellare, e la partizione train, validation e test e coerente con la tabella descritta nella sezione dati.

```text
ID 		1283	1328	1358	1421	1461		1466	1500	1521 	1542
Name 	Unnamed Rolf 	Unnamed Unnamed Qendresa	Unnamed Unnamed Unnamed Trixie
ID 		1575 	1674	1702	1715	1716		-		-		-		-
Name 	Numa 	Ianos 	Unnamed Unnamed Unnamed 	Apollo	Blas	Daniel	Juliette
```

I primi identificativi derivano da TRACKS_CL7, mentre i casi Apollo, Blas, Daniel e Juliette non sono inclusi in Flaounas et al. (2023) e derivano dal minimo MSLP ERA5. Il dataset e partizionato per evitare leakage tra eventi in training e test.

La partizione delle classi e illustrata nella tabella riportata di seguito, dove la differenza tra i due validation set riguarda il bilanciamento di classi positive e negative.

```text
			Num. cyclones 	Total time interval 	Num. Video clips
Train set   12 				23 days 8h45’ 			1238 					Balanced
Val  set    3 				7 days 8h5' 			354 					Balanced
Test set    3 				8 days 13h25' 			2400 					201 positives (cyclones)
																			2199 negatives (no cyclones)
```

In the next tables it is shown the time window for some example medicane, before and after the narrower time window selection: (tabella completa medicanes_new_windows.csv).

Best results: Max. Accuracy 91 %. Balanced dataset (validation set). Best results: Max. Accuracy 89 %. UNbalanced dataset (test set). I risultati migliori riportano una Max. Accuracy di 91% su validation bilanciata e 89% su test sbilanciato, con confusion matrices come supporto visivo. (confusion matrices images).

Il terzo stadio e una regressione per il tracking del centro di rotazione, costruita selezionando solo i video tile con cicloni dal dataset di detection. Tracking dataset: collecting only video tiles with cyclones from detection dataset. Il tracking dataset comprende 12 cicloni per training con 834 video, 3 cicloni per test con 160 video e 3 cicloni per validation con 192 video. In un riepilogo alternativo, il dataset di tracking ha 835 campioni in training e 280 in test, e la specifica va consolidata per la riproducibilita. The dataset has the following number of samples: Training set samples: 835. Test set samples: 280. ...following activity:  devo completare cosa ho fatto.

In the following figure some dataset samples are shown, along with their labeled center. Fig. 5.4 tracking samples – Example frames from tracking video dataset. The red dot represents the center track label.

16- tracking results  (image plot histogram error in pixel and in km).

17- Cyclone first pass.

## Verifica, validazione e limiti
La verifica della detection usa accuracy su validation e test con dataset bilanciati e sbilanciati. Le confusion matrix permettono di quantificare falsi positivi e falsi negativi e di comprendere la sensibilita al mismatch tra tracce e segnali IR. L'analisi degli errori evidenzia casi in cui un centro tracciato non corrisponde a rotazione visibile, a causa di cloud absence o strutture non cicloniche. Per il tracking, l'errore di localizzazione e rappresentato in pixel e in chilometri con istogrammi, utili per valutare precisione e distribuzione degli errori.

Il sistema presenta limiti dovuti alla dipendenza dalle tracce, alla sensibilita alle condizioni di visibilita IR e alla mancanza di una definizione completa di confidenza e flags di qualita. La conversione da coordinate pixel a geografiche richiede un modello geometrico coerente con la proiezione geostazionaria e con il crop. La fase di cyclone first pass e menzionata come passaggio di workflow ma non e ancora descritta in dettaglio.

## Limiti e lavori futuri
La documentazione tecnica deve includere una descrizione completa degli iperparametri e delle scelte di ottimizzazione, insieme alla configurazione di training e inference, per garantire riproducibilita. E necessario definire formalmente i flag di qualita, gli indicatori di confidenza e le regole di validita del prodotto in presenza di input incompleti. La sezione di tracking richiede l'allineamento definitivo dei numeri di campioni e la definizione di metriche aggregate con confronti in chilometri. Ulteriori case study con evoluzione temporale e confronto con analisi sinottiche fornirebbero una validazione qualitativa utile al contesto ESA.

## Provenance
Il modello di base e VideoMAEv2, un modello open source per video, con repository dedicato al progetto https://github.com/dandarm/VideoMAEv2 e riferimenti alle pubblicazioni originali su Arxiv. La pipeline qui descritta introduce una specializzazione su dati Airmass RGB del Mediterraneo, un pre-processing con de-normalizzazione BT, compositi RGB, tiling e stacking temporale, oltre a dataset etichettati con TRACKS_CL7 e successive rifiniture. La fase supervisionata aggiunge un classificatore binario e un regressore per il centro di rotazione, entrambi basati sullo stesso backbone specializzato. L'entrypoint operativo e la struttura delle cartelle devono essere documentati esplicitamente nel repository per garantire tracciabilita.

## References e panorama della letteratura
Artificial Intelligence applied to Atmospheric Science is quite a new field, e i lavori recenti indicano un panorama in rapida evoluzione. A Comprehensive AI Approach for Monitoring and Forecasting Medicanes Development (2024), di J. Martinez-Amaya, V. Nieves e J. Muñoz-Mari, propone un approccio con tracking tramite k-means clustering e forecast basato su CNN e Random Forest. Il dataset include 58 medicanes tra 1984 e 2023 con immagini IR Meteosat in Brightness Temperature e reanalisi CERRA ed ERA5 con MSLP e vento. Lo studio e il primo modello ML dedicato ai medicanes e consente la previsione di eventi estremi fino a due giorni con accuracy 65-80%, offrendo un'alternativa ai modelli tradizionali e adattabilita al cambiamento climatico. URL https://www.mdpi.com/2225-1154/12/12/220.

A Statistical Learning Approach to Mediterranean Cyclones (2025), di L. Roveri, L. Fery, L. Cavicchia e F. Grotto, usa una fase non supervisionata con Latent Dirichlet Allocation per ridurre dimensionalita e una fase supervisionata con classificatore statistico su feature LDA. Il dataset e ERA5 con campi di vento e pressione combinati con un archivio di tracce mediterranee. Il risultato e un workflow di classificazione con accuracy circa 90% nel rilevamento di cicloni mediterranei usando poche feature, superando le difficolta di definire forma e diametro dei medicanes e abilitando precursori temporali di cicloni estremi. URL https://arxiv.org/pdf/2501.15694v1.

Deepti: Deep-Learning-Based Tropical Cyclone Intensity Estimation System (2020), di M. Maskey, R. Ramachandran, M. Ramasubramanian, I. Gurung, et al., utilizza una CNN custom per la regressione dell'intensita con vento massimo. Il dataset include immagini IR GOES ogni 15 minuti su piu satelliti geostazionari tra 2000 e 2019, con labels best-track HURDAT2. Il sistema automatizza la stima di intensita da IR, emulando l'analisi Dvorak, e ottiene RMSE di 13.24 knots, comparabile a tecniche operative, con stime near real time e un portale di visualizzazione. URL https://ieeexplore.ieee.org/document/9149719.

A Novel Deep Learning Based Model for Tropical Intensity Estimation and Post-Disaster Management of Hurricanes (2021), di J. Devaraj, S. Ganesan, R. M. Elavarasan e U. Subramaniam, usa una CNN potenziata con batch normalization e dropout per la stima di intensita, e una fase di transfer learning con VGG-19 per la classificazione di danni ed eventi. Il dataset include immagini IR GOES con label HURDAT2 su uragani 2000-2019, oltre a immagini satellitari post-evento, ad esempio Houston, per il danno e video di severe weather per la classificazione. Il modello riduce l'errore a 7.6 knots RMSE e raggiunge 98% accuracy per il danno e 97% per eventi severi, con un approccio integrato per previsione e gestione impatti. URL https://www.mdpi.com/2076-3417/11/9/4129.

Tropical and Extratropical Cyclone Detection Using Deep Learning (2020), di C. Kumler-Bonfanti, J. Stewart, D. Hall e M. Govett, usa una segmentazione con U-Net, in quattro varianti, per identificare regioni cicloniche, con confronto di etichette tra IBTrACS e un algoritmo euristico. Il dataset include mappe globali di total precipitable water da GFS a 0.5 gradi e immagini satellitari GOES nel canale water vapor, con ground truth da IBTrACS e detection euristica di cicloni extratropicali. I modelli raggiungono accuracy 80-99% nel marking delle regioni cicloniche con Dice 0.51-0.76, includendo vortici deboli ignorati manualmente, e il modello per cicloni extratropicali e tre volte piu veloce dell'algoritmo operativo di riferimento. L'uso di input multisorgente consente di scoprire cicloni ambigui che sfuggono ai criteri stretti. URL https://colab.ws/articles/10.1175%2Fjamc-d-20-0117.1.

A Hybrid ML/Physics-Based Modeling Framework for 2-Week Extended Prediction of Tropical Cyclones (2024), di X. Liu et al., combina un modello numerico WRF a circa 2 km con un modello deep learning globale Pangu-Weather a circa 25 km. Il dataset comprende simulazioni WRF ad alta risoluzione e forecast ML, con test su casi reali come il ciclone Freddy 2023. Il framework estende la previsione di traiettoria e intensita fino a circa 2 settimane rispetto ai circa 5 giorni dei modelli tradizionali, con miglioramento a 7 giorni di accuratezza e affidabilita lungo 14 giorni. URL https://agupubs.onlinelibrary.wiley.com/doi/10.1029/2024JH000207.

Tropical cyclone size estimation based on deep learning using infrared and microwave satellite data (2023), di J. Xu, X. Wang, H. Wang, C. Zhao, H. Wang e J. Zhu, propone TC-ResNet, una variante di ResNet-50 con conv 5×5 sullo shortcut e doppia attenzione canale/spazio per la regressione del raggio del vento. Il dataset integra immagini IR geostazionarie e mappe microonde passive dal 2003 al 2017 e il dataset globale R34 best-track. Il modello ottiene un errore medio di 11.3 nm, circa 21 km, con coefficiente di correlazione 0.907, superiore a metodi empirici tradizionali. L'integrazione multisorgente permette di filtrare rumore e blur nelle nubi cicloniche e migliorare la stima del rischio associato a vento e swell. URL https://www.frontiersin.org/journals/marine-science/articles/10.3389/fmars.2022.1077901/full.

Detecting Extratropical Cyclones of the Northern Hemisphere with Single Shot Detector (2022), di M. Shi, P. He, Y. Shi et al., applica un SSD one-stage per rilevare centri ciclonici e fase evolutiva. Il dataset include immagini satellitari whole-disk visibile e IR dell'emisfero Nord, con etichette manuali per ciclogenesi, maturita e dissipazione basate su criteri Bonfanti 2017. Il modello ottiene mAP 86.6% per cicloni maturi e circa 79.3% considerando le tre classi, dimostrando l'efficacia del deep learning per cicloni extratropicali con grande variabilita di forma e dimensione. URL https://www.mdpi.com/2072-4292/14/2/254.
