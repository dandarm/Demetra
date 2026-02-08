## 1. Scopo e contesto

### Summary


Questo documento descrive la base teorica e la catena di produzione del prodotto DeMeTrA, un algoritmo di deep learning per il rilevamento e il tracciamento dei medicanes, cioe cicloni mediterranei con caratteristiche simil tropicali. Il contesto operativo e il monitoraggio near real time di fenomeni rari, in cui la disponibilita di immagini geostazionarie ad alta frequenza consente di seguire l'evoluzione rapida dei sistemi ciclonici. L'obiettivo è fornire un modello per individuare la presenza di cicloni e localizzarne il centro di rotazione. DeMeTrA sfrutta l'imaging infrarosso del sensore Spinning Enhanced Visible InfraRed Imager, SEVIRI, a bordo di Meteosat Second Generation, MSG. Il processo prevede vari addestramenti finalizzati a detection e tracking di cicloni nel mar Mediterraneo.

Il prodotto DeMeTrA è composto da due moduli, quello principale è derivato dal modello VideoMAE, che si occupa sia della rivelazione che del tracking ad alta risoluzione dei cicloni, mentre un altro modulo più leggero first-guess si occupa di migliorare la rivelazione e abbassare i falsi positivi, fornendo una stima rozza di posizione del centro usando immagini a più bassa risoluzione, da fornire al modulo VideoMAE che raffina e migliora la stima con un margine di errore inferiore.

Il modulo VideoMAE adatta il codice open source VideoMAE v2 allo studio dei medicanes su sequenze video satellitari derivate da compositi AirmassRGB dalle immagini SEVIRI MSG. Il sistema integra un post-pretraining auto-supervisionato di tipo masked autoencoding su video e due fasi supervisionate orientate rispettivamente alla classificazione ciclone/no-ciclone e al tracking del centro ciclone come regressione di coordinate. La detection di cicloni è preparatoria al tracking, perché consente di filtrare i dati video e ridurre il carico computazionale della regressione. 

Il prodotto viene quindi impiegato lungo un flusso che copre la costruzione dei dataset, il training distribuito, l'inferenza e una verifica che combina metriche quantitative e ispezioni qualitative su scala mediterranea.