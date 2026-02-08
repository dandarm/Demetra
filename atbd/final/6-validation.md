## 6. Verifica: validazione, test, incertezze, limiti
La verifica quantitativa combina la confusion matrix con metriche specifiche per eventi rari, in modo da ridurre l'impatto della forte asimmetria tra casi positivi e negativi. Le metriche adottate sono POD, FAR, CSI, HSS e Balanced Accuracy, con formule esplicitate di seguito.

### Metriche
Le metriche di classificazione sono quelle tipiche dei problemi di eventi rari in meteorologia, per cui la sola accuracy e' fuorviante in presenza di forte sbilanciamento. Le formule usate si basano sulla confusion matrix con H (hits), M (misses), F (false alarms) e C (correct negatives) 

##### Classification
|                    | Predicted: Event Yes | Predicted: Event No |
|--------------------|----------------------|---------------------|
| True: Event Yes    | H (hits)             | M (misses)          |
| True: Event No     | F (false alarms)     | C (correct negatives) |

e includono:

POD = H / (H + M)
FAR = F / (H + F)
CSI = H / (H + F + M)
HSS = 2(H*C - M*F) / [(H+M)(M+C) + (H+F)(F+C)]
BA = 0.5 * [H/(H+M) + C/(C+F)]

Le analisi includono il calcolo di accuracy, FPR, FNR, e F1 score, calcolate nell'esecuzione del codice ma non riportate quì perché ridondanti rispetto alle suddette metriche.
Tutte queste misure permettono di valutare la capacita' di rilevare gli eventi, il tasso di falsi allarmi e la performance rispetto a un modello di baseline casuale.
La verifica della detection usa metriche su set di validation bilanciati e sbilanciati. 

Best results: Max. Accuracy 91 %. Balanced dataset (validation set). Best results: Max. Accuracy 89 %. UNbalanced dataset (test set). I risultati migliori riportano una Max. Accuracy di 91% su validation bilanciata e 89% su test sbilanciato, con confusion matrices come supporto visivo. 

<fig. confusion matrices>



##### Tracking
Per il tracking l'errore di localizzazione è rappresentato in pixel e in chilometri. Nei plot a istogrammi seguenti sono riportate le distribuzioni, utili per valutare precisione e valor medio/mediana degli errori.
L'errore in km e' derivato passando dalle coordinate pixel a lat/lon tramite la griglia georeferenziata e applicando la formula di haversine con raggio medio terrestre 6371.0088 km. In codice, la distanza geodetica e' calcolata come dlat = lat2_rad - lat1_rad, dlon = lon2_rad - lon1_rad, a = sin(dlat/2)^2 + cos(lat1_rad) * cos(lat2_rad) * sin(dlon/2)^2, c = 2 * arcsin(sqrt(a)), distance_km = EARTH_RADIUS_KM * c.  

tracking results  <image plot histogram error in pixel and in km>




Il workflow di validazione prevede la costruzione di set di test e validation attraverso selezioni che escludono frame temporalmente lontani dal ciclone, e la generazione di dataset bilanciati, sbilanciati o annuali completi. 
L'analisi degli errori ha evidenziato che i casi in cui il centro tracciato (ground truth label) non corrisponde a rotazione visibile, sono spesso dovuti a cloud absence o strutture non cicloniche o in fase precoce/finale. 





Sono effettuate anche verifiche qualitative visive, comprendenti la generazione di animazioni di immagini del mediterraneo con overlay delle predizioni, la visualizzazione delle tile e delle posizioni del centro ciclone, e il rendering di file MP4 per cicloni specifici o per lunghi intervalli temporali. 


< visualizzazione di alcuni case study >