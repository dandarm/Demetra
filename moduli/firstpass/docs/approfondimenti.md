prompt
rieccoci! ora ti chiedo di farmi un riassunto di tutte le nostre idee, cioè di dare una descrizione dettagliata di cosa fa questo repo che abbiamo creato, spiegando motivazioni e obiettivi, tralasciando le prove intermedie e concentrandoti solo sulle scelte finali che si sono rivelate efficaci. non usare liste ed elenchi, orienta il riassunto a un documento tecnico



- 
- 1

il repository realizza un “primo passaggio” sul Mediterraneo che opera su dei frame a ratio alterato, addestra un modello leggero di rilevazione/localizzazione del centro del ciclone e produce predizioni standardizzate, con un contratto d’uscita pensato per alimentare in modo pulito uno stadio HR successivo. La scelta architetturale è minimale ma robusta: ogni immagine full-basin viene pre-processata a dimensione quadrata S×S (S=224), quindi non preserva il rapporto d’aspetto, e annotando per ciascun file i parametri di scala. Questo step rende deterministiche le coordinate durante training/inferenza. Le etichette di presenza vengono derivate unicamente da un file di finestre temporali (medicanes_new_windows.csv): un frame è positivo se il suo timestamp cade nell’intervallo [start_time, end_time] e negativo altrimenti. Quando nel CSV sono disponibili le coordinate del centro a risoluzione originale, esse vengono proiettate in modo coerente nello spazio S×S utilizzando gli stessi parametri dello stretch. Per arricchire in modo controllato il set di negativi, si può pre-selezionare un insieme di file immagine da una fascia temporale di buffer pre/post rispetto alle finestre temporali di ground truth, garantendo così contesti meteorologici realistici e bilanciati.



L’inferenza opera nello stesso spazio S×S per evitare conversioni non necessarie: dal logit si ricava la probabilità di presenza, dalla heatmap si estrae il picco con argmax.  Le coordinate del centro restano definite nei pixel della canvas S×S; quando occorre riportarle ai pixel originali o generare ritagli per il secondo stadio HR, si attiva esplicitamente la retro-proiezione che usa i metadati del stretch per invertire scala, e salva le ROI centrate con una tile avente  lato solito per videomae . 

si riportano AUPRC e ROC-AUC come metriche soglia-indipendenti, oltre a Precision/Recall/F1 alla soglia scelta e all’errore di centro sui soli positivi




-2 modello x3d_m

Il primo passo è consistito nel riconoscere che l’utilizzo di una singola immagine satellitare limita fortemente la capacità del modello di distinguere sistemi convettivi che possiedono circolazione reale da strutture nuvolose che presentano simmetria solo istantanea. La dinamica temporale fornisce informazioni cruciali: la rotazione organizzata delle bande, la coerenza con cui evolve la spirale del ciclone e la sua persistenza sono segnali che emergono solo osservando più fotogrammi distribuiti su un intervallo temporale significativo. Per questo motivo è stato introdotto un modello video come X3D-M e un sistema di input che fornisce sequenze di sedici frame distanziati nel tempo attraverso una stride di quattro unità, pari a venti minuti ciascuna grazie alla risoluzione temporale originaria di cinque minuti. Questa struttura genera una finestra di osservazione che copre diverse ore e consente al backbone 3D di intercettare la rotazione come una trasformazione spaziotemporale continua.

### Background: X3D e la variante X3D_M

X3D (“**Expanding Architectures for Efficient Video Recognition**”) è una famiglia di reti neurali per video introdotta da Christoph Feichtenhofer (CVPR 2020) con l’obiettivo esplicito di massimizzare il rapporto accuratezza/complessità nei compiti di video understanding, in particolare **action recognition** e, più in generale, classificazione e detection su sequenze video. L’idea chiave è partire da una base estremamente piccola, concettualmente vicina a un backbone 2D “economico”, e trasformarla in un modello spaziotemporale efficiente mediante una procedura di **espansione progressiva** lungo assi distinti della capacità del modello: larghezza dei canali (bottleneck width), risoluzione temporale e durata del clip, risoluzione spaziale, profondità della rete. La procedura è stepwise: a ogni step si espande un solo asse, misurando il guadagno di accuratezza rispetto al costo computazionale, ottenendo una traiettoria di modelli con budget crescente (XS, S, M, L, …) e prestazioni competitive.

Dal punto di vista architetturale, X3D si può leggere come un backbone di tipo ResNet “3D”, ma progettato per essere estremamente parsimonioso in parametri: il paper sottolinea un risultato importante e controintuitivo, cioè che **un’alta risoluzione spazio-temporale può essere compatibile con un numero di parametri molto contenuto**, purché la rete rimanga stretta (pochi canali) e la capacità venga distribuita in modo efficiente. La famiglia X3D viene inoltre riportata come efficace non solo per classificazione video ma anche per compiti di detection su video, con un profilo di costo favorevole rispetto a molte alternative dell’epoca.

#### La variante X3D_M: profilo computazionale e impostazioni d’ingresso

X3D_M rappresenta la configurazione “medium” nella famiglia. Nel paper originale, X3D_M viene riportato con circa **3.76M parametri** e un costo di circa **4.73 GFLOPs per clip** (misurato su single-clip center-crop; il costo per video dipende dal numero di clip/crop usati in test), con accuratezza top-1 nell’intorno di **74.6%** su Kinetics-400 (val) nel setting descritto dagli autori.

Nell’implementazione e benchmark forniti da PyTorchVideo/PyTorch Hub, X3D_M è distribuito come modello **pre-addestrato su Kinetics-400** e viene tipicamente usato con clip di **16 frame** e sampling rate **5**, con parametri di trasformazione che includono short-side scale e crop (nel tutorial viene mostrato un crop 256×256 per X3D_M, coerente con una pipeline d’inferenza standardizzata). In quel contesto vengono riportati anche valori di FLOPs e accuratezza che possono differire da quelli del paper perché dipendono dal protocollo di valutazione (numero di viste/clip, risoluzione effettiva del crop e preprocessing).

A livello di “vincoli” del builder del modello, la documentazione PyTorchVideo esplicita che, per le varianti X3D, la risoluzione spaziale attesa dall’architettura è indicata tramite `input_crop_size` e la lunghezza temporale tramite `input_clip_length`; per X3D_M questi valori sono **224** (crop size) e **16** (clip length) nella definizione del modello, mentre varianti diverse usano risoluzioni differenti (ad esempio X3D-L aumenta la crop size). Questo è rilevante in fine-tuning perché il modello e la sua testa di pooling globale sono costruiti assumendo coerenza tra stride complessivi e dimensioni del clip/crop.

#### Casi d’uso tipici e ragioni della scelta

X3D_M viene impiegato soprattutto quando si vuole un backbone video “general purpose” con un compromesso solido tra costo e accuratezza, adatto a pipeline operative dove la computazione è un vincolo reale: analisi massiva di flussi video, inference multi-clip con budget moderato, oppure fine-tuning su domini specifici (meteorologia satellitare, videosorveglianza, robotica, sport analytics) dove l’obiettivo è sfruttare un backbone pre-addestrato che abbia già appreso primitive spaziotemporali generiche. La famiglia X3D è stata concepita proprio per rendere praticabile l’uso di modelli video in scenari in cui reti 3D “larghe” risultano proibitive; il paper e la descrizione dell’autore enfatizzano il risultato di ottenere prestazioni competitive con **molte meno operazioni e parametri** rispetto a precedenti approcci, mantenendo un design relativamente semplice e trasferibile



- 3 

Parallelamente si è compreso che la supervision introdotta dal dataset è affetta da un rumore non trascurabile, perché il centro di un ciclone non è un punto matematico ma un’entità meteorologica complessa, con incertezza intrinseca sia nella definizione fisica sia nel processo di labeling. Le heatmap gaussiane tradizionalmente usate nei modelli di keypoint detection — basate su una regressione per pixel — si sono rivelate inadeguate, poiché l’errore tende a essere dominato da picchi spuri anche molto lontani dal blob principale. La presenza di un solo pixel erroneamente alto può compromettere la previsione complessiva, rendendo la loss estremamente sensibile al rumore e impedendo alla validation loss di riflettere adeguatamente i miglioramenti del modello. La regressione puntuale basata su argmax è risultata ancora più fragile, poiché l’operatore non è derivabile e soffre di instabilità quando la struttura della heatmap non è ben definita.

Per affrontare questo problema è stata introdotta la trasformazione DSNT (Differentiable Spatial-to-Numerical Transform), una tecnica che converte la heatmap generata dal modello in una coppia di coordinate continue attraverso un processo completamente differenziabile. Invece di considerare il valore massimo, DSNT normalizza l’intera heatmap tramite una softmax bidimensionale, interpretandola come una distribuzione di probabilità spaziale. Le coordinate predette diventano l’aspettativa di questa distribuzione e non sono quindi influenzate da picchi isolati con massa trascurabile. Una caratteristica centrale di DSNT è che il gradiente della loss sulle coordinate si propaga a tutta la heatmap, consentendo al modello di apprendere una rappresentazione morbida e coerente della posizione del centro. Inoltre questo approccio riduce drasticamente la sensibilità a imperfezioni del label, perché non richiede di replicare esattamente la forma della gaussiana target, ma solo di concentrare la massa attorno al punto corretto. In un contesto in cui il target reale può essere incerto di diversi pixel, DSNT si rivela molto più stabile rispetto alle perdite per pixel tradizionali.



