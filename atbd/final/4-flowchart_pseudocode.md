


### Pseudocodice (sintetico)
```text
carica canali IR_097, IR_108, WV_062, WV_073 per ciascun timestamp
calcola BT con BT = v (xmax - xmin) + xmin
costruisci Airmass RGB con R = WV_062 - WV_073, G = IR_097 - IR_108, B = WV_062
ritaglia su area mediterranea (latcorners = [30, 48], loncorners = [-7, 46]) -> si ottengono immagini da 1290x420 px
dividi l'area mediterranea in tiles 224x224 con stride di 213 in x e 196 in y
per ogni tile nella griglia mediterranea
    crea un clip con 16 frame separati di 5 minuti
    se la traccia del centro di rotazione cade nel tile per almeno 6 frame
        assegna etichetta positiva
    altrimenti assegna etichetta negativa
addestra VideoMAEv2 unsupervised su clip non etichettati per feature representation learning
salva il modello addestrato (checkpoint) come backbone per i successivi training
usa il backbone specializzato per addestrare un classificatore binario
usa lo stesso backbone per addestrare un regressore di coordinate RC



**Algorithm 1:** Mediterranean Cyclone Detection \& Tracking Pipeline
**Input:** Raw Satellite Channels $Ch_{IR097}, Ch_{IR108}, Ch_{WV062}, Ch_{WV073}$, Timestamps $T$
**Output:** Binary Classifier $M_{cls}$, Coordinate Regressor $M_{reg}$

**// Phase 1: Data Preprocessing**
1. **for each** timestamp $t \in T$ **do**
2. $\quad$ Load channels $Ch_{IR097}, Ch_{IR108}, Ch_{WV062}, Ch_{WV073}$
3. $\quad$ Compute Brightness Temperature: $BT \leftarrow v \cdot (x_{max} - x_{min}) + x_{min}$
4. $\quad$ Construct Airmass RGB:
$\quad \quad R \leftarrow BT_{WV062} - BT_{WV073}$
$\quad \quad G \leftarrow BT_{IR097} - BT_{IR108}$
$\quad \quad B \leftarrow BT_{WV062}$
5. $\quad$ Crop to Mediterranean area: $Lat \in [30, 48], Lon \in [-7, 46]$
$\quad \rightarrow$ Yields image $I_t$ of size $1290 \times 420$ px
6. **end for**

**// Phase 2: Dataset Generation**
7. Partition area into tiles of size $224 \times 224$ (Stride: $S_x=213, S_y=196$)
8. **for each** tile location $(x, y)$ in Grid **do**
9. $\quad$ Create video clip $C$ with 16 frames (temporal stride: 5 min)
10. $\quad$ **if** Rotation Center track $\in$ tile for $\ge 6$ frames **then**
11. $\quad \quad Label_C \leftarrow 1$ (Positive)
12. $\quad$ **else**
13. $\quad \quad Label_C \leftarrow 0$ (Negative)
14. **end for**

**// Phase 3: Model Training**
15. Train **VideoMAEv2** (unsupervised) on unlabeled clips $\rightarrow$ Learn Feature Representation
16. Save model checkpoint as **Backbone** $B$
17. Train Binary Classifier using $B$ (finetuning) $\rightarrow M_{cls}$
18. Train Coordinate Regressor (RC) using $B$ (finetuning) $\rightarrow M_{reg}$




