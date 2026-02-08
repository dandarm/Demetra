## 5. Fase di training


Il training è suddiviso due parti, una self-supervised e la successiva supervised. La prima parte unsupervised è un post-pretraining, ovvero un training di specializzazione nel quale il modello impara a ricostruire patch mascherate su videoclip non etichettate, con l'obiettivo di specializzarsi sulle dinamiche dei medicanes senza imporre vincoli di classe. Nel supervisionato, la classificazione ottimizza la separazione cyclone/no-cyclone, mentre il tracking ottimizza l'errore MSE sulle coordinate del centro; in entrambi i casi vengono creati set di train, validation e test, con possibilita' di bilanciamento o di split per identita' di ciclone.

Three stage training:
Il training e organizzato in tre stadi. Le due fasi supervisionate usano le feature apprese nella fase di specializzazione. Il primo è un addestramento auto-supervisionato di specializzazione su clip Airmass RGB, senza etichette, con circa 80’000 video in training e 20’000 in test. In questa fase si usa il modello pre-addestrato di taglia “giant” per post pre-training che si aggiunge al pre-training da zero, con l'obiettivo di aumentare la capacita di estrazione delle feature utili ai task successivi. Il training ha una durata dell'ordine di giorni, la curva di loss di training e validazione non mostra overfitting, indicando una specializzazione efficace, per cui il modello viene salvato come checkpoint per i task successivi. <fig. Training and validation losses versus epochs>. 

I principali iperparametri riguardanti il post pre-training: mascheramento con `mask_ratio=0.9` e `decoder_mask_ratio=0.5`, decoder depth 4, patch 14 e input size 224, clip di 16 frame con sampling rate 4 e num_sample 4, batch size 32, learning rate 6e-4, warmup di 30 epoche, gradient clipping a 0.02, il training è stato fermato a 150 epoche. Sul versante dei dataset, la stride per ricostruire il mosaico mediterraneo sono 213 in x e 196 in y, per un totale di 12 tiles su tutta l'area selezionata del Mediterraneo con sovrapposizione spaziale piccola, e nessuna sovrapposizione temporale.

Il secondo stadio è una classificazione binaria. Il trial definitvo usa un dataset bilanciato di circa x video, divisi in due classi, con class 1 che include solo cicloni medicanes (From the Full_List_Medicanes), e class 0 senza cicloni. Per la definizione dei samples con classe 1 è stata usatil riferimento temporale e spaziale delle tracce contenute in TRACKS_CL7, ma restringendo la finestra temporale ai frame con rotazione chiaramente osservabile, a causa del mismatch tra le tracce e il contenuto IR, dove avvengono casi di cloud absence o assenza di rotazione evidente con una frequenza non trascurabile in tutto il dataset osservato.
Allo scopo di correggere questo mismatch tra immagini e labeling, è stato applicato un refinement, trovando new cyclones time boundaries attraverso una approfondita ispezione e  selezione manuale. Sono stati assegnati nuovi start e end times in corrispondenza di rotazione chiaramente visibile. 

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
Train set   12 				23 days 8h45’ 			1238 				Balanced
Val  set    3 				7 days 8h5' 			354 				Balanced
Test set    3 				8 days 13h25' 			2400 				201 positives (cyclones)
											2199 negatives (no cyclones)
```

In the next tables it is shown the time window for each medicane, before and after the narrower time window selection: <tabella completa medicanes_new_windows.csv>.


Il terzo stadio e una regressione per il tracking del centro di rotazione, costruita selezionando solo i video tile con cicloni dal dataset di detection. Il tracking dataset comprende 12 cicloni per training con 835 video, 3 cicloni per test con 280 video e 3 cicloni per validation con 192 video. 
Era necessario includere solo esempi positivi per il tracciamento, poiché durante l'addestramento non è ammesso un output indefinito. Ciò comporta la necessità di una classificazione adeguata prima del tracciamento in fase di inferenza.  
È stata fatta l'assunzione di considerare la posizione del centro di rotazione nel frame finale come obiettivo, per questo motivo l'ultimo frame di ogni campione di dati del videoclip deve avere una posizione di traccia corrispondente dai dati TRACKS_CL.  
Queste semplificazioni temporali rendono il problema del posizionamento del centro sempre definito rispetto alla finestra di osservazione. 




In the following figure some dataset samples are shown, along with their labeled center. Fig. 5.4 tracking samples – Example frames from tracking video dataset. The red dot represents the center track label.


I dettagli computazionali che influenzano la convergenza includono la scalatura di batch size e learning rate con il world size nel training distribuito, l'uso di scheduler coseno per learning rate e weight decay, l'applicazione di una modulazione layer-wise del learning rate e l'uso di mixed precision con gradient scaling. L'impostazione dei seed è impiegate per aumentare la riproducibilita' tra run. La presenza di log e di checkpoint periodici permette di ricostruire l'andamento e di riprendere il training in caso di interruzione. 
Per raggiungere le prestazioni mostrate inseguito è stato fondamentale usare un training distribuito su cluster multi-nodo con configurazioni di 16 GPU.


(TODO: Cyclone first pass)