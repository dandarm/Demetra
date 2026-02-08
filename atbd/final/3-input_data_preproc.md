## 3. Input data & pre-processing

### Dati di input
L'addestramento usa dati IR di EUMETSAT Rapid Scan High Rate SEVIRI Level 1.5 Image Data MSG, con alta frequenza di 5 minuti e alta risoluzione spaziale, utili per monitorare fenomeni meteorologici in rapida evoluzione. I dati sono scaricabili da Google Cloud Big Query Public Data, con permesso di redistribuzione, (Google ospita dataset pubblici e di terze parti per conto dei provider, offrendo accesso affidabile e su larga scala senza oneri di storage). I dati sono resi disponibili da Open Climate Fix, che li ha processati ulteriormente usando le librerie satip e SatPy. Il dataset disponibile non contiene i valori numerici originali, perché e stato calibrato con SatPy e linearmene normalizzato per produrre Brightness Temperature, con valori mappati nell'intervallo [0, 1023], cioè 10 bit per canale. Il dataset e in formato Zarr e non e stato riproiettato, quindi rimane in proiezione geostazionaria.

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



Fig. 5.2 – Air mass RGB example image covering the Mediterranean basin, with superimposed borders. Image time: 17th September 2020, 03:40 UTC


Il processo di dataset building quindi è costituito da: download, ricostruzione dei dati, (includendo la trasformazione inversa verso valori originali di Brightness Temperature), e la costruzione del composito (con le corrette finestre di normalizzazione Meteosat). Queste operazioni portano al cosiddetto "Source Dataset".  Questi processo di download e pre-processing sono stati implementati con script Python ottimizzati per garantire velocità ed efficienza, e sono disponibili insieme al codice sorgente.  (fig airmassRGB senza land borders).


Questo Source Dataset sorgente comprende circa 860K frame AirmassRGB di 1290 X 420 pixel sull'area del Mediterraneo, per un totale di circa 600 GByte, con copertura temporale ⪞ 7.5 years tra 2010 e 2023.


I dati di input comprendono le sequenze AirmassRGB e le tracce Manos `TRACKS_CL*.dat` (classi CL2-CL10) con coordinate geografiche e pressione, integrate quando necessario con tracce ERA5 per i medicane più recenti. Sono inoltre presenti file di supporto che contengono finestre temporali modificate (re-labeling manuale dopo ispezione visiva) per definire la presenza del ciclone (nascita e dissoluzione) , come `medicanes_new_windows.csv`, che definiscono intervalli piu' coerenti con le fasi di rotazione evidente dei cicloni.



### Pre-processing e labeling

La regione mediterranea è suddivisa in tile 224 x 224 pixel parzialmente sovrapposti, con un numero tipico di 12 tile, ma variabile in funzione dell'overlap scelto per le tiles. Ogni tile copre approssimativamente 757 km in latitudine e 805 km in longitudine, con risoluzione di circa 3.38 km per pixel in latitudine e 3.59 km per pixel in longitudine. I tile sono impilati in clip di 16 frame distanziati di 5 minuti, per un totale di 80 minuti, con sovrapposizione temporale e generazione di un video tile ogni ora. Questo formato e richiesto dal modello pre-addestrato per dimensioni spaziali e numero di frame. <fig tiles-mediterraneo  - fig video-tiles.>

L'etichettatura si basa sul database TracksCL7, con trasformazione delle coordinate del centro di rotazione da geografiche a pixel. La procedura è altamente ottimizzata perche il working dataset viene ricostruito molte volte in funzione delle scelte di training. Un tile e etichettato come positivo quando il centro di rotazione e contenuto nel tile per almeno 6 frame su 16, altrimenti e etichettato come negativo. I tile contenenti il centro sono indicati come label 1 e gli altri come label 0. TracksCL7 deriva dalla combinazione di sette metodi di tracking applicati a reanalisi. 
<fig labeled-tiles.>

In the following <fig 5.3> it is shown the process of tiling and labeling from a source image. Fig. 5.2 – Airmass RGB source frame: the squares are sliced to form video clips in the same position for 16 frames. A green square is shown where the video tile is selected as positive sample, containing a cyclone center track (cyan dots).

Le coordinate lat/lon vengono trasformate in pixel usando una griglia georeferenziata coerente con la risoluzione 1290x420; viene inoltre ribaltato l'asse Y per mantenere la consistenza tra sistema geografico e immagine.

A partire da queste tracce si costruisce un 'master'dataframe che, per ogni frame, include path dell'immagine, offset della tile, coordinate del centro del ciclone se presente, e label. Il master dataframe viene poi segmentato in gruppi temporali contigui, raggruppato rispetto all'offset della tile e trasformato in videoclip da 16 frame, e viene infine salvato come file CSV di train, val e test necessari per il training. Il flusso include un bilanciamento tra esempi positivi e negativi. Lo split per train, val e test avviene rispetto agli identificativi di ciclone. Sono inclusi anche altri dataset specifici per TRACKS_CL10, o per medicanes che costituiscono i casi studio.
Sono inoltre presenti pipeline per la generazione di dataset full-year che includono periodi senza cicloni, cosi da valutare il comportamento su distribuzioni realistiche e altamente sbilanciate.

Infine, per il tracking si costruiscono dataset dedicati che includono solo esempi positivi e associano a ogni clip le coordinate del centro sul frame finale.

Il dataset così costituito dai manifest CSV e dai video tiles salvati su disco prodotto da queste procedure di tiling e labeling è denominato "Working Dataset".

Il codice per queste procedure di labeling è stato ottimizzato e reso veloce perché la generazione del working dataset deve essere ripetuta più di una volta, per ogni diversa scelta di parametri e strategia di training.