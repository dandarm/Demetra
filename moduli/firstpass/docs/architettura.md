# ATBD sintetico dello stato attuale (modalita energy)


## 1) Descrizione dell'algoritmo

L'algoritmo prende in ingresso una clip temporale di lunghezza `T` e la processa con un backbone spatio-temporale `x3d_m`. Dal backbone si diramano due uscite interne: una mappa di logits spaziali della heatmap e un logit scalare della testa di presenza separata (`z_head`). La mappa heatmap non viene usata solo come supporto visivo, ma rappresenta la base della localizzazione del centro ciclone e di una parte della stima di presenza.

La localizzazione avviene in modo differenziabile con DSNT: i logits della heatmap vengono normalizzati con softmax spaziale (temperatura `dsnt_tau`) e da questa distribuzione si ottiene la coordinata continua `(x, y)` tramite aspettativa. In questo assetto il centro non dipende da un argmax duro, e quindi il gradiente resta informativo anche quando il picco non e netto.

Per la presenza, in modalita `energy`, il sistema costruisce uno score fuso `z_tot` che combina informazione geometrica e informazione semantica. La parte heatmap contribuisce con due feature: `E`, definita come 'E = mean(sigmoid(topk(H_logits, K)))'  e `C = 1 - H(P)/log(H*W)`, P = softmax2d(H_logits / tau_dsnt) , concentrazione della distribuzione spaziale ottenuta dall'entropia normalizzata della softmax DSNT. Queste due quantita vengono fuse con il logit della head separata secondo la forma:

`z_tot = b0 + wE * E + wC * C + wH * z_head`

dove `b0`, `wE`, `wC` e `wH` sono parametri scalari addestrabili. La probabilita di presenza usata dal sistema e `p_tot = sigmoid(z_tot)`, e la decisione binaria finale e ottenuta confrontando `p_tot` con la soglia di inferenza configurata (`presence_threshold`).

Nella versione attuale il decoder della heatmap non utilizza piu layer di deconvoluzione (ConvTranspose2d), ma una sequenza di blocchi resize-conv, cioe upsampling deterministico (bilineare) seguito da convoluzione 3x3: questa scelta e stata introdotta per ridurre gli artefatti a scacchiera osservati nelle mappe predette, migliorare la continuita spaziale del segnale e rendere piu stabile l’apprendimento del picco in presenza di stride diversi, mantenendo invariata la logica di localizzazione DSNT e la compatibilita con il resto della pipeline.

Nel sistema attuale DSNT e il meccanismo che trasforma la heatmap in una stima di centro continua e differenziabile: invece di prendere il massimo discreto della mappa, i logits spaziali vengono convertiti in una distribuzione di probabilita con softmax 2D (controllata da una temperatura), e la coordinata finale e calcolata come valore atteso della distribuzione stessa. Questo approccio evita la discontinuita dell’argmax, mantiene un gradiente informativo anche quando il picco non e ancora netto e rende la localizzazione piu stabile durante il training, specialmente nelle fasi in cui la rete produce blob larghi o multi-modali; in pratica, DSNT separa il problema “dove sta il centro” dal problema “quanto alto e il picco”, permettendo di ottenere coordinate affidabili anche quando l’ampiezza assoluta della heatmap non e ancora ben calibrata per la detection binaria.





## 2) Flusso delle operazioni

Il flusso operativo parte dal manifest e dalla costruzione delle clip temporali con gli stride configurati, includendo il mapping coerente tra coordinate nel dominio immagine e dominio heatmap. Ogni clip passa nel modello `x3d_m`, che produce heatmap logits e `z_head`. A questo punto il ramo di localizzazione calcola la posizione continua del centro con DSNT, mentre il ramo di detection calcola `E` e `C` dalla heatmap e costruisce `z_tot` usando anche `z_head`.

Nella fase di inferenza la presenza non e dedotta dal solo picco massimo della heatmap, ma dal valore fuso `z_tot`. Questo punto e importante perche evita che la decisione binaria dipenda in modo fragile da un singolo pixel o da oscillazioni locali del picco. In uscita il sistema produce quindi una coordinata centro continua e una probabilita di presenza `p_tot` coerente con la stessa formula usata in training.

La coerenza train/infer e mantenuta anche nel caricamento dei checkpoint: quando il run include i parametri del blocco `energy_fusion`, il modello viene istanziato con lo stesso blocco per riusare esattamente la stessa definizione di score.

## 3) Fase di training

Nel training corrente la loss totale combina tre contributi: localizzazione DSNT, regolarizzazione strutturale della heatmap e classificazione presenza sul logit fuso. In forma compatta:

`L_total = w_heatmap * L_dsnt + w_heatmap_focal_reg * L_focal_reg + w_peak_bce * BCEWithLogits(z_tot, y)`

Il termine `L_dsnt` ottimizza la posizione del centro; `L_focal_reg` agisce come vincolo di forma della heatmap per ridurre rumore e strutture spurie; il termine BCE opera su `z_tot`, cioe sullo stesso score che viene poi usato in inferenza. In modalita `energy` questa scelta rende esplicito che la detection finale deve essere spiegata dalla combinazione tra energia top-K, concentrazione spaziale e logit della testa separata, senza introdurre disallineamenti tra score di training e score di deployment.

I parametri `b0`, `wE`, `wC`, `wH` vengono inizializzati da configurazione (`energy_init_*`) e aggiornati con l'ottimizzatore insieme agli altri pesi del modello. Questo consente al sistema di imparare automaticamente quanto fidarsi del ramo heatmap rispetto alla head di presenza separata, invece di fissare a priori pesi statici della fusione.


Distinguere chiaramente due stride che prima tendevano a sovrapporsi concettualmente: manifest_stride, che controlla ogni quante righe del manifest viene scelto un nuovo frame centrale (quindi quante clip quasi duplicate entrano davvero nel dataset), e temporal_stride, che invece definisce la distanza temporale tra i frame interni alla singola clip. Nella pratica, ridurre la ridondanza con manifest_stride ha eliminato molte clip quasi identiche che saturavano il training con segnale ripetitivo, migliorando sia la stabilita dell’ottimizzazione sia il tempo per epoca; allo stesso tempo, mantenere un temporal_stride coerente con la dinamica fisica del fenomeno ha preservato l’informazione evolutiva utile dentro ogni clip, evitando finestre troppo “ferme” o troppo sparse. La combinazione finale di queste due scelte ha aumentato la qualita del gradiente (meno rumore da duplicazione, piu contenuto temporale informativo), con effetti concreti su convergenza, robustezza in validazione/test e migliore separazione tra i casi realmente difficili e quelli semplicemente ridondanti.


Nella configurazione corrente la presenza non viene stimata da un singolo indicatore isolato, ma da uno score finale unificato z_tot, che fonde il logit della head di classificazione con due descrittori della heatmap (E, energia top-K, e C, concentrazione spaziale): questo rende la decisione piu robusta perche combina evidenza semantica globale e coerenza spaziale locale in un unico logit ottimizzato direttamente in training. In inferenza si applica la sigmoid a z_tot ottenendo p_tot, interpretata come probabilita di presenza, e la classificazione binaria si ottiene confrontando p_tot con una soglia tau (presence_threshold): se p_tot >= tau il frame/clip e considerato positivo, altrimenti negativo. Il punto importante e che la stessa quantita (z_tot) viene usata sia per la loss di training sia per la decisione finale, riducendo i mismatch tra ottimizzazione e deployment; la soglia, quindi, non cambia la qualita intrinseca del ranking ma regola il trade-off operativo tra recall e precision in base all’obiettivo applicativo.