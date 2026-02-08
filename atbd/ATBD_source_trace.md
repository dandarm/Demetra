# Tracciamento fonti ATBD DeMeTrA

Questo file contiene una verifica incrociata riga per riga dei file sorgente e indica dove ogni riga e stata incorporata nel documento ATBD. Per le righe con tabelle o segnaposti, la nota indica che sono riportati verbatim nel testo ATBD.

## slides.md -> ATBD_DeMeTrA_ATBD.md
L1: "1- Main objective" -> ATBD: Scopo, contesto e summary.
L2: [blank] -> separatore, mantenuto come separazione logica.
L3: "Deep-learning Medicane Tracking Algorithm (DeMeTrA)" -> ATBD: Scopo, contesto e summary.
L4: "Exploits IR images by the Spinning Enhanced Visible InfraRed Imager SEVIRI" -> ATBD: Scopo, contesto e summary; Input data e pre-processing.
L5: "on board the Meteosat Second Generation (MSG)." -> ATBD: Scopo, contesto e summary.
L6: [blank] -> separatore.
L7: "DeMeTrA main objectives" -> ATBD: Scopo, contesto e summary.
L8: "• To detect the presence of Medicanes" -> ATBD: Scopo, contesto e summary.
L9: "- by means of Binary Classification" -> ATBD: Scopo, contesto e summary.
L10: "• To identify and track the position of the medicane’s RC in NRT" -> ATBD: Scopo, contesto e summary.
L11: "- by means of Regression Analysis" -> ATBD: Scopo, contesto e summary.
L12: "Detection is preparatory to tracking" -> ATBD: Scopo, contesto e summary.
L13: [blank] -> separatore.
L14: [blank] -> separatore.
L15: [blank] -> separatore.
L16: "2- Vision Transformers for DeMeTrA: VideoMAE" -> ATBD: Algoritmo.
L17: [blank] -> separatore.
L18: "Vision Transformers (ViTs) have quickly become state of the art in image recognition and video analysis (1)" -> ATBD: Algoritmo.
L19: "Video Masked Autoencoder (VideoMAE (2)) is one of the most advanced ViT architectures for video understanding:" -> ATBD: Algoritmo.
L20: "• Open source pre-trained model on huge video data" -> ATBD: Algoritmo.
L21: "• AutoEncoder architecture for self-supervised learning (do not need labels)" -> ATBD: Algoritmo.
L22: "(1) Mauricio et al., 2023" -> ATBD: Algoritmo e References e panorama della letteratura.
L23: "(2) Tong et al., 2022, Wang et al. 2023" -> ATBD: Algoritmo e References e panorama della letteratura.
L24: [blank] -> separatore.
L25: "DeMeTrA AI-based algorithm is a specialization of" -> ATBD: Algoritmo.
L26: "the pre-trained VideoMAEv2 model" -> ATBD: Algoritmo.
L27: "for cyclone recognition and tracking" -> ATBD: Algoritmo.
L28: "through fine-tuning on our dataset" -> ATBD: Algoritmo.
L29: [blank] -> separatore.
L30: "3- DeMeTrA input data (Airmass RGB)" -> ATBD: Input data e pre-processing.
L31: "Input data consists of Airmass RGB composite video clips, built from the" -> ATBD: Input data e pre-processing.
L32: "SEVIRI IR measurements from the MSG Rapid Scan Service (RSS)" -> ATBD: Input data e pre-processing.
L33: "Multiple IR and water vapor channels are combined into an Airmass RGB composite by integrating:" -> ATBD: Input data e pre-processing.
L34: "• infrared water vapour channel ~6.2 µm (WV_062) & upper-level water vapor 7.3 µm (WV_073) → red channel;" -> ATBD: Input data e pre-processing.
L35: "• channels sensitive to ozone absorption (IR_097 & IR_108) highlighting stratospheric air intrusions" -> ATBD: Input data e pre-processing.
L36: "→ green channel;" -> ATBD: Input data e pre-processing.
L37: "• WV_062 → blue channel." -> ATBD: Input data e pre-processing.
L38: "Airmass RGB enhances visualization of air masses, atmospheric moisture and frontal systems facilitating the" -> ATBD: Input data e pre-processing.
L39: "visual identification of cyclone dynamics, making it highly effective for medicane detection and analysis." -> ATBD: Input data e pre-processing.
L40: "R = WV_062 – WV_073" -> ATBD: Input data e pre-processing.
L41: "G = IR_097 – IR_108" -> ATBD: Input data e pre-processing.
L42: "B = WV_062" -> ATBD: Input data e pre-processing.
L43: [blank] -> separatore.
L44: [blank] -> separatore.
L45: "4- DeMeTrA dataset building" -> ATBD: Input data e pre-processing.
L46: "• Big models need big data" -> ATBD: Input data e pre-processing.
L47: "• Available dataset from Ifremer was not complete (lack of temporal continuity) and" -> ATBD: Input data e pre-processing.
L48: "not enough for a proper training" -> ATBD: Input data e pre-processing.
L49: "• We downloaded the EUMETSAT Rapid Scan High Rate SEVIRI Level 1.5 Image Data MSG" -> ATBD: Input data e pre-processing.
L50: "data made available by Google Cloud Storage «Big Query Public Data»." -> ATBD: Input data e pre-processing.
L51: "Google hosts public and third-party datasets on behalf of their providers, giving users" -> ATBD: Input data e pre-processing.
L52: "reliable, large-scale access without the burden of data storage." -> ATBD: Input data e pre-processing.
L53: "• This dataset does not contain the original numerical values," -> ATBD: Input data e pre-processing.
L54: "but it is transformed by Open Climate Fix* (* openclimatefix.org):" -> ATBD: Input data e pre-processing.
L55: "pixel values have been calibrated by SatPy to produce" -> ATBD: Input data e pre-processing.
L56: "normalized Brightness Temperatures values" -> ATBD: Input data e pre-processing.
L57: [blank] -> separatore.
L58: "5- DeMeTrA dataset building" -> ATBD: Input data e pre-processing.
L59: "• Need to download data" -> ATBD: Input data e pre-processing.
L60: "• Transform back to original Brightness Temperature values" -> ATBD: Input data e pre-processing.
L61: "• Build the composite with proper normalization ranges as in the Meteosat recipe" -> ATBD: Input data e pre-processing.
L62: "We have the Source Dataset  (fig airmassRGB senza land borders)" -> ATBD: Input data e pre-processing; segnaposto verbatim.
L63: [blank] -> separatore.
L64: "~ 860K AirmassRGB image frames (1290 X 420 pixels) over Mediterranean sea area" -> ATBD: Input data e pre-processing.
L65: "~ 600 GByte total size" -> ATBD: Input data e pre-processing.
L66: "⪞ 7.5 years total time, between 2010 and 2023" -> ATBD: Input data e pre-processing.
L67: [blank] -> separatore.
L68: [blank] -> separatore.
L69: "6- DeMeTrA dataset building - tiling" -> ATBD: Input data e pre-processing.
L70: "Building the Working Dataset" -> ATBD: Input data e pre-processing.
L71: "- Slicing Mediterranean region into 12 partially overlapping tiles - 224 x 224 pixel each" -> ATBD: Input data e pre-processing.
L72: "- Stacking 16 frame tiles to build a single video tile (~ 1h 20' time span)" -> ATBD: Input data e pre-processing.
L73: "- One video tile each hour: overlapping in time" -> ATBD: Input data e pre-processing.
L74: "(pixel size and number of frames are Required by the pre-trained model)" -> ATBD: Input data e pre-processing.
L75: "fig tiles-mediterraneo  - fig video-tiles" -> ATBD: Input data e pre-processing; segnaposto verbatim.
L76: [blank] -> separatore.
L77: [blank] -> separatore.
L78: [blank] -> separatore.
L79: "7- DeMeTrA dataset building - labeling" -> ATBD: Input data e pre-processing.
L80: "• Tracks are used to label each tile frame (static) using TRACKS_CL7* (Flaounas et al., 2023)" -> ATBD: Input data e pre-processing.
L81: "• Transform track position from geo coordinates (lon, lat) to pixel domain (x,y)" -> ATBD: Input data e pre-processing.
L82: "• Higly computationally optimized process:" -> ATBD: Input data e pre-processing.
L83: "the Working Dataset has to be rebuilt many times, depending on training design choices" -> ATBD: Input data e pre-processing.
L84: "The tile where the track is contained" -> ATBD: Input data e pre-processing.
L85: "is marked as label 1 (cyclone)" -> ATBD: Input data e pre-processing.
L86: "green square" -> ATBD: Input data e pre-processing.
L87: "All other tiles: label 0 (no cyclone)" -> ATBD: Input data e pre-processing.
L88: "• Label 1 videos: at least 6 tile frames with cyclone" -> ATBD: Input data e pre-processing.
L89: "• Label 0 videos: otherwise" -> ATBD: Input data e pre-processing.
L90: "* Mediterranean storms produced through the combination of 7 different cyclone-tracking methods applied to reanalysis (Flaounas et al., 2023)" -> ATBD: Input data e pre-processing.
L91: [blank] -> separatore.
L92: "fig labeled-tiles" -> ATBD: Input data e pre-processing; segnaposto verbatim.
L93: [blank] -> separatore.
L94: [blank] -> separatore.
L95: "8- DeMeTrA training" -> ATBD: Fase di training.
L96: "Three stage training:" -> ATBD: Fase di training.
L97: "• Self-supervised \"specialization\" training, for data feature learning," -> ATBD: Fase di training.
L98: "to leverage unlabeled big data available" -> ATBD: Fase di training.
L99: "• Supervised classification training, in order to detect the presence of a cyclone" -> ATBD: Fase di training.
L100: "labels of Mediterranean cyclones occurrence using TRACKS_CL7*" -> ATBD: Fase di training.
L101: "• Supervised regression training, to learn the cyclone center coordinates" -> ATBD: Fase di training.
L102: "the lst two training: Using learned features" -> ATBD: Fase di training.
L103: [blank] -> separatore.
L104: [blank] -> separatore.
L105: "9- Working Dataset - details" -> ATBD: Fase di training.
L106: [blank] -> separatore.
L107: "• Self-supervised dataset: random videos" -> ATBD: Fase di training.
L108: "train set ~ 80K video tiles" -> ATBD: Fase di training.
L109: "test set ~ 20K video tiles" -> ATBD: Fase di training.
L110: "• Supervised binary classification dataset - 4 refining steps:" -> ATBD: Fase di training.
L111: "o Step 1: Tracks CL7" -> ATBD: Fase di training.
L112: "o Step 2: Tracks CL10" -> ATBD: Fase di training.
L113: "o Step 3: Only Medicanes *" -> ATBD: Fase di training.
L114: "o Step 4: Only Medicanes * and narrow time window with clearly observable rotation" -> ATBD: Fase di training.
L115: [blank] -> separatore.
L116: "* From the Full_List_Medicanes" -> ATBD: Fase di training.
L117: [blank] -> separatore.
L118: [blank] -> separatore.
L119: "10- Self-supervised \"specialization\" training" -> ATBD: Fase di training.
L120: "(masking - reconstruction)" -> ATBD: Fase di training.
L121: [blank] -> separatore.
L122: "Training and validation losses versus epochs" -> ATBD: Fase di training; segnaposto verbatim.
L123: "Long time training (~ days)" -> ATBD: Fase di training.
L124: "Results: no overfitting → good!" -> ATBD: Fase di training.
L125: "→ save this model as a checkpoint for the next stage training" -> ATBD: Fase di training.
L126: [blank] -> separatore.
L127: [blank] -> separatore.
L128: "11- Supervised classification training results" -> ATBD: Fase di training.
L129: "Using Tracks CL7: Max. Accuracy 80 %" -> ATBD: Fase di training.
L130: "Need for increased dataset quality: why? because of" -> ATBD: Fase di training.
L131: "Mismatch with the tracks: IR images may not capture cyclonic dynamics for certain tracks (cloud absence, no cloud rotation...)" -> ATBD: Fase di training e Verifica, validazione e limiti.
L132: [blank] -> separatore.
L133: [blank] -> separatore.
L134: "12- Dataset refinement: Cyclones new time boundaries (manual selection)" -> ATBD: Fase di training.
L135: "Using Only Medicanes *" -> ATBD: Fase di training.
L136: "• New Start and End times based on visual inspection" -> ATBD: Fase di training.
L137: "• Clearly visible cyclone clouds rotation" -> ATBD: Fase di training.
L138: "(medicane_new_windiws.csv)" -> ATBD: Fase di training; segnaposto verbatim.
L139: [blank] -> separatore.
L140: "13- Final dataset for detection" -> ATBD: Fase di training.
L141: "• 18 cyclones" -> ATBD: Fase di training.
L142: "splitting train/val/test (same shown in TR_v2.md)" -> ATBD: Fase di training.
L143: [blank] -> separatore.
L144: "14-" -> ATBD: Fase di training.
L145: "Best results: Max. Accuracy 91 %" -> ATBD: Fase di training.
L146: "Balanced dataset (validation set)" -> ATBD: Fase di training.
L147: [blank] -> separatore.
L148: "Best results: Max. Accuracy 89 %" -> ATBD: Fase di training.
L149: "UNbalanced dataset (test set)" -> ATBD: Fase di training.
L150: "(confusion matrices images)" -> ATBD: Fase di training; segnaposto verbatim.
L151: [blank] -> separatore.
L152: [blank] -> separatore.
L153: [blank] -> separatore.
L154: "15- Medicane Tracking training" -> ATBD: Fase di training.
L155: [blank] -> separatore.
L156: "Tracking dataset:" -> ATBD: Fase di training.
L157: "collecting only video tiles with cyclones" -> ATBD: Fase di training.
L158: "from detection dataset" -> ATBD: Fase di training.
L159: [blank] -> separatore.
L160: "Train set: 12 cyclones, 834 videos" -> ATBD: Fase di training.
L161: "Test set: 3 cyclones, 160 videos" -> ATBD: Fase di training.
L162: "Validation set: 3 cyclones, 192 videos" -> ATBD: Fase di training.
L163: [blank] -> separatore.
L164: [blank] -> separatore.
L165: [blank] -> separatore.
L166: "16- tracking results" -> ATBD: Fase di training; segnaposto verbatim.
L167: "(image plot histogram error in pixel and in km)" -> ATBD: Fase di training; segnaposto verbatim.
L168: [blank] -> separatore.
L169: [blank] -> separatore.
L170: "17- Cyclone first pass" -> ATBD: Fase di training e Verifica, validazione e limiti; segnaposto verbatim.

## TR_v2.md -> ATBD_DeMeTrA_ATBD.md
L1: "5.NRT detection and tracking (WP2400)" -> ATBD: Scopo, contesto e summary.
L2: "5.1.Introduction" -> ATBD: Scopo, contesto e summary.
L3: "WP 2400 is dedicated to the development of Deep Learning techniques for the detection," -> ATBD: Scopo, contesto e summary.
L4: "monitoring and tracking of medicanes. The main focus of the WP2400 is the development, testing" -> ATBD: Scopo, contesto e summary.
L5: "and validation of the Deep-learning Medicane Tracking Algorithm (DeMeTrA),exploiting IR" -> ATBD: Scopo, contesto e summary.
L6: "images derived from geostationary satellites observing the Mediterranean area (i.e. Meteosat" -> ATBD: Scopo, contesto e summary.
L7: "Second Generation (MSG) – Spinning Enhanced Visible InfraRed Imager SEVIRI). The main" -> ATBD: Scopo, contesto e summary.
L8: "capability of the DeMeTrA algorithm will be:" -> ATBD: Scopo, contesto e summary.
L9: "1. to detect the presence of Mediterranean cyclones, and to identify their center of rotation (RC);" -> ATBD: Scopo, contesto e summary.
L10: "2. to track the position of the cyclone’s RC;" -> ATBD: Scopo, contesto e summary.
L11: "3. to early identify the cyclones evolving in tropical-like cyclone (detection of the closed eye)." -> ATBD: Scopo, contesto e summary.
L12: "The training dataset has been built from IR measurements available from the Spinning Enhanced" -> ATBD: Input data e pre-processing.
L13: "Visible InfraRed Imager SEVIRI sensor of the MSG Rapid Scan Service (RSS), collecting data with 5" -> ATBD: Input data e pre-processing.
L14: "minutes frequency, useful for monitoring rapidly evolving meteorological phenomena. The" -> ATBD: Input data e pre-processing.
L15: "dataset includes also labels of medicane’s occurrence (based on WP2300 output) and the coordinates of the cyclones’ center of rotation (RC) obtained from independent observations, as" -> ATBD: Scopo, contesto e summary e Input data e pre-processing.
L16: "well as the model-based tracks available for all cyclones considered in the study used as" -> ATBD: Scopo, contesto e summary e Input data e pre-processing.
L17: "reference." -> ATBD: Scopo, contesto e summary.
L18: [blank] -> separatore.
L19: "5.2." -> ATBD: Algoritmo.
L20: "Deep learning for NRT tracking: DeMeTrA" -> ATBD: Algoritmo.
L21: "The DeMeTrA ML-based tracking algorithm by means of specialization in cyclone recognition is" -> ATBD: Algoritmo.
L22: "currently based on VideoMAEv2." -> ATBD: Algoritmo.
L23: "The VideoMAEv2 model is a transformer-based self-supervised learning architecture specifically" -> ATBD: Algoritmo.
L24: "designed for video data. It efficiently captures spatio-temporal features by masking input patches" -> ATBD: Algoritmo.
L25: "and reconstructing the missing content. The pre-trained version of the model processes video" -> ATBD: Algoritmo.
L26: "clips consisting of 16 frames, each with dimensions of 224×224 pixels. Each frame is then" -> ATBD: Algoritmo.
L27: "subdivided, during the forward pass, into non-overlapping patches of size 14x14 pixels, forming" -> ATBD: Algoritmo.
L28: "the basic units for input to the transformer network. The architecture includes:" -> ATBD: Algoritmo.
L29: "•an encoder-decoder transformer structure." -> ATBD: Algoritmo.
L30: "•tube masking strategy, where spatial and temporal patches (tubelets) are masked to encourage the model to learn robust video representations." -> ATBD: Algoritmo.
L31: "Technical details and deeper insights can be found in ANNEX A, and in the related VideoMAEv2" -> ATBD: Algoritmo e Provenance.
L32: "repository on GitHub (https://github.com/dandarm/VideoMAEv2) specifically built for this project, and the original research paper on Arxiv." -> ATBD: Provenance.
L33: [blank] -> separatore.
L34: "(figura VideoMAEv2_flowchart.png)" -> ATBD: Algoritmo; segnaposto verbatim.
L35: "Fig. 5.1 – Technical details of the VideoMAEv2 model: encoder-decoder structure, with separate masking for" -> ATBD: Algoritmo; segnaposto verbatim.
L36: "each one. The latent representation vector (embedding) is shown between the encoder and the decoder." -> ATBD: Algoritmo; segnaposto verbatim.
L37: [blank] -> separatore.
L38: "5.2.1. VideoMAE training Dataset" -> ATBD: Input data e pre-processing.
L39: "The training dataset has been built from IR measurements by EUMETSAT Rapid Scan High Rate" -> ATBD: Input data e pre-processing.
L40: "SEVIRI Level 1.5 Image Data MSG (Rapid Scan Service: RSS) with high frequency availability (every" -> ATBD: Input data e pre-processing.
L41: "5 minutes) at high spatial resolution (3 km at s.s.p.), useful for monitoring rapidly evolving" -> ATBD: Input data e pre-processing.
L42: "meteorological phenomena. Data is downloadable from the Google Cloud platform “Big Query" -> ATBD: Input data e pre-processing.
L43: "Public data”, with permission by EUMETSAT to redistribute data, provided by Open Climate Fix," -> ATBD: Input data e pre-processing.
L44: "that processed the data further using satip and SatPy library." -> ATBD: Input data e pre-processing.
L45: "The original data have been slightly transformed: the numerical values produced by the sensor" -> ATBD: Input data e pre-processing.
L46: "have been rescaled with SatPy python library to produce calibrated Brightness Temperatures" -> ATBD: Input data e pre-processing.
L47: "values, and the pixel values have been linearly mapped to the range [0, 1023] (i.e. 10 bits per" -> ATBD: Input data e pre-processing.
L48: "channel).. The dataset was built using the Zarr format, and has not been spatially reprojected" -> ATBD: Input data e pre-processing.
L49: "(and so is in \"geostationary\" projection)." -> ATBD: Input data e pre-processing.
L50: "Therefore data needs to de-normalized by means the following formula:" -> ATBD: Input data e pre-processing.
L51: "𝐵𝑇 = v (𝑥𝑚𝑎𝑥 − 𝑥𝑚𝑖𝑛) + 𝑥𝑚𝑖𝑛" -> ATBD: Input data e pre-processing.
L52: [blank] -> separatore.
L53: "where v is the rescaled value ranging in the [0, 1] domain, while xmax and xmin are the arrays of" -> ATBD: Input data e pre-processing.
L54: "maximum and minimum values, respectively, and they are specific for each channel, as shown in" -> ATBD: Input data e pre-processing.
L55: "the following Table 5.1:" -> ATBD: Input data e pre-processing.
L56: [blank] -> separatore.
L57: "SEVIRI channel    Xmin      Xmax" -> ATBD: Input data e pre-processing; tabella verbatim.
L58: "IR_097     2,84         317,87" -> ATBD: Input data e pre-processing; tabella verbatim.
L59: "IR_108     199,10       313,28" -> ATBD: Input data e pre-processing; tabella verbatim.
L60: "WV_062     199,57       249,92" -> ATBD: Input data e pre-processing; tabella verbatim.
L61: "WV_073     198,95       286,96" -> ATBD: Input data e pre-processing; tabella verbatim.
L62: [blank] -> separatore.
L63: "Table 5.1 – Maximum and minimum values for the channels of interest." -> ATBD: Input data e pre-processing; tabella verbatim.
L64: [blank] -> separatore.
L65: [blank] -> separatore.
L66: [blank] -> separatore.
L67: [blank] -> separatore.
L68: "These infrared and water vapor channels are further pre-processed, and are combined into the" -> ATBD: Input data e pre-processing.
L69: "Airmass RGB composite. According to EUMETSAT guidelines (RGB Recipes https://eumetrain.org/sites/default/files/2020-05/RGB_recipes.pdf and Best Practices  https://www-cdn.eumetsat.int/files/2020-04/pdf_using_rgb_best_practices.pdf)," -> ATBD: Input data e pre-processing.
L70: "Airmass RGB enhances visualization of air masses and frontal systems, facilitating the visual" -> ATBD: Input data e pre-processing.
L71: "identification of cyclone dynamics and atmospheric moisture, as well as cloud features, by" -> ATBD: Input data e pre-processing.
L72: "integrating:" -> ATBD: Input data e pre-processing.
L73: "•Red channel: infrared water vapour channel around 6.2 µm (WV_062) – upper-level water" -> ATBD: Input data e pre-processing.
L74: "vapor 7.3 µm (WV_073);" -> ATBD: Input data e pre-processing.
L75: "• Green channel: difference between channels sensitive to ozone absorption (IR_097 -" -> ATBD: Input data e pre-processing.
L76: "IR_108), highlighting stratospheric air intrusions;" -> ATBD: Input data e pre-processing.
L77: "• Blue channel: WV_062." -> ATBD: Input data e pre-processing.
L78: "This combination is highly effective for medicane detection and analysis, and so it is used to build" -> ATBD: Input data e pre-processing.
L79: "up the entire dataset" -> ATBD: Input data e pre-processing.
L80: "Then, spatial coordinates were cropped to a geostationary projection covering the" -> ATBD: Input data e pre-processing.
L81: "Mediterranean area (Fig. 5.2), specifically:" -> ATBD: Input data e pre-processing.
L82: [blank] -> separatore.
L83: [blank] -> separatore.
L84: "• Latitude range: 30°– 48° N" -> ATBD: Input data e pre-processing.
L85: "• Longitude range: from -7° – 46°" -> ATBD: Input data e pre-processing.
L86: [blank] -> separatore.
L87: "Fig. 5.2 – Air mass RGB example image covering the Mediterranean basin, with superimposed borders. Image time: 17th September 2020, 03:40 UTC" -> ATBD: Input data e pre-processing; segnaposto verbatim.
L88: [blank] -> separatore.
L89: [blank] -> separatore.
L90: "By using this spatial selection, each image of the Mediterranean region consists of 1290×420 pixels" -> ATBD: Input data e pre-processing.
L91: [blank] -> separatore.
L92: "A python script was developed and refined for efficient downloading and processing of the data" -> ATBD: Input data e pre-processing.
L93: "as described, and the resulting collected images sum up for a total of nearly 860’000," -> ATBD: Input data e pre-processing.
L94: "corresponding to 600 Gbyte of data, spanning more the 7 years and half of time interval, in a" -> ATBD: Input data e pre-processing.
L95: "wider range between 2010 and 2023 years." -> ATBD: Input data e pre-processing.
L96: [blank] -> separatore.
L97: "This dataset is named ‘Source Dataset’ to distinguish it from the further processing that enables" -> ATBD: Input data e pre-processing.
L98: "it to become input for the model as training set and validation sets. The procedures and code" -> ATBD: Input data e pre-processing.
L99: "responsible for this next step, must be efficient as well; in order to be executed every time a" -> ATBD: Input data e pre-processing.
L100: "parameter update is needed to build it. This last dataset is named ‘Working Dataset’, there are" -> ATBD: Input data e pre-processing.
L101: "many of them, one for each trial design choice." -> ATBD: Input data e pre-processing.
L102: [blank] -> separatore.
L103: "These procedures split the entire images into ‘tiles', small squares of 224×224 pixel, with partial" -> ATBD: Input data e pre-processing.
L104: "overlapping that results into 12 tiles, this number can change depending on the overlapping" -> ATBD: Input data e pre-processing.
L105: "parameters choice; after that, 16 tiles are stacked to form a single video clip, each frame being" -> ATBD: Input data e pre-processing.
L106: "equally spaced temporally by 5 minutes. This resulting video is an input sample to the model. The" -> ATBD: Input data e pre-processing.
L107: "side length and the total number of frames are fixed parameters needed by the pretrained" -> ATBD: Input data e pre-processing.
L108: "model. A single tile covers an area of approximately 757 km latitude by 805 km longitude The" -> ATBD: Input data e pre-processing.
L109: "spatial resolution of each pixel is about 3.38 km in latitude and 3.59 km in longitude. Each video" -> ATBD: Input data e pre-processing.
L110: "clip spans a time window of 80 minutes, and there can be an overlap in time among video tiles," -> ATBD: Input data e pre-processing.
L111: "too." -> ATBD: Input data e pre-processing.
L112: [blank] -> separatore.
L113: "Further labeling process is required to build the working dataset. For this purpose, the TracksCL7" -> ATBD: Input data e pre-processing.
L114: "database is used (Flaounas et al., 2023), from which cyclone center coordinates are taken to" -> ATBD: Input data e pre-processing.
L115: "determine the presence of cyclones in each video tiles given its pixel offset respect to the cropped" -> ATBD: Input data e pre-processing.
L116: "angles." -> ATBD: Input data e pre-processing.
L117: [blank] -> separatore.
L118: "A proper coordinate transformation from geospatial latitude and longitude to pixel domain is" -> ATBD: Input data e pre-processing.
L119: "performed for this task, and the positive labels are therefore assigned where and when the" -> ATBD: Input data e pre-processing.
L120: "center cyclone track is included into the tile, for at least 6 frames out of 16. All other videos are" -> ATBD: Input data e pre-processing.
L121: "marked with a negative label, for this binary classification task. A further subdivision in more" -> ATBD: Input data e pre-processing.
L122: "classes could be made for cyclone detection task, but the binary classification is the simplest trial" -> ATBD: Input data e pre-processing.
L123: "to start with." -> ATBD: Input data e pre-processing.
L124: [blank] -> separatore.
L125: "This labeling process is needed for the supervised training, while the self-supervised training" -> ATBD: Input data e pre-processing.
L126: "does not need labels, so the information regarding cyclones position is not considered." -> ATBD: Input data e pre-processing.
L127: [blank] -> separatore.
L128: "This ’tiling’ and labeling algorithms applied to the source dataset produce the working dataset." -> ATBD: Input data e pre-processing.
L129: "In the following fig 5.3 it is shown the process of tiling and labeling from a source image." -> ATBD: Input data e pre-processing; segnaposto verbatim.
L130: [blank] -> separatore.
L131: "Fig. 5.2 – Airmass RGB source frame: the squares are sliced to form video clips in the same position for 16 frames." -> ATBD: Input data e pre-processing; segnaposto verbatim.
L132: "A green square is shown where the video tile is selected as positive sample, containing a cyclone center track" -> ATBD: Input data e pre-processing; segnaposto verbatim.
L133: "(cyan dots)." -> ATBD: Input data e pre-processing; segnaposto verbatim.
L134: [blank] -> separatore.
L135: "A two-stage training is chosen to exploit the big dataset available: the first stage is self-" -> ATBD: Fase di training.
L136: "supervised, and includes a wide variety of Airmass RGB images, with the purpose of building a" -> ATBD: Fase di training.
L137: "specialized model on such kind of images. We used the ‘giant’ pre-trained model for this" -> ATBD: Fase di training.
L138: "unsupervised “specialization” pre-training, which is a training in addition to the pre-training from" -> ATBD: Fase di training.
L139: "scratch, in order to increase the model feature extraction capability for the second stage" -> ATBD: Fase di training.
L140: "trainings, necessary for detection and tracking tasks." -> ATBD: Fase di training.
L141: "The dataset for the first stage self-supervised training is very big since it doesn’t need labels or" -> ATBD: Fase di training.
L142: "balancing, with a total number of samples:" -> ATBD: Fase di training.
L143: "80’000 for the training set" -> ATBD: Fase di training.
L144: "20’000 for the test set" -> ATBD: Fase di training.
L145: [blank] -> separatore.
L146: "After that, the second stagesupervised classification training is performed," -> ATBD: Fase di training.
L147: "using the model trained in the first stage, with a balanced dataset having size of nearly 3000" -> ATBD: Fase di training.
L148: "videos, divided evenly between two classes:" -> ATBD: Fase di training.
L149: [blank] -> separatore.
L150: [blank] -> separatore.
L151: "•class 1: includes cyclones recorded in the TRACKS_CL7 database, including both" -> ATBD: Fase di training.
L152: "medicanes and other Mediterranean cyclones." -> ATBD: Fase di training.
L153: "•class 0: no cyclones occurred" -> ATBD: Fase di training.
L154: [blank] -> separatore.
L155: "This was only the first trial. After that, further dataset processing was needed, due to noise and" -> ATBD: Fase di training.
L156: "labeling mismatch with respect to the information contained in the airmassRGB images." -> ATBD: Fase di training.
L157: "Such a mismatch was found in many occurrences, where a cyclone center was provided without" -> ATBD: Fase di training e Verifica, validazione e limiti.
L158: "visible clouds, or without rotating cyclone formations, only translatory moving clouds." -> ATBD: Fase di training e Verifica, validazione e limiti.
L159: "For this reason, a more limited dataset consisting of TrackCL10 cyclones was taken into account" -> ATBD: Fase di training.
L160: "for a second trial training. Other refinements were necessary to increase the training and" -> ATBD: Fase di training.
L161: "generalization performance, with accurate visual inspection for the entire dataset." -> ATBD: Fase di training.
L162: "This analysis led to another version of the dataset including only clearer medicanes in a third trial" -> ATBD: Fase di training.
L163: "of trainings, excluding almost all tracks from CL10 not identified as medicanes. Moreover the" -> ATBD: Fase di training.
L164: "time window of the medicane occurrences was reduced by eliminating the initial and final stages" -> ATBD: Fase di training.
L165: "with respect to the CL10 track, maintaining the frames when a visible clear cloud rotation can be" -> ATBD: Fase di training.
L166: "observed. This was necessary to achieve a satisfactory performance for the detection task." -> ATBD: Fase di training.
L167: [blank] -> separatore.
L168: "The final dataset consists of the following 18 cyclones:" -> ATBD: Fase di training.
L169: "ID \t\t1283\t1328\t1358\t1421\t1461\t\t1466\t1500\t1521 \t1542" -> ATBD: Fase di training; tabella verbatim.
L170: "Name \tUnnamed Rolf \tUnnamed Unnamed Qendresa\tUnnamed Unnamed Unnamed Trixie" -> ATBD: Fase di training; tabella verbatim.
L171: "ID \t\t1575 \t1674\t1702\t1715\t1716\t\t-\t\t-\t\t-\t\t-" -> ATBD: Fase di training; tabella verbatim.
L172: "Name \tNuma \tIanos \tUnnamed Unnamed Unnamed \tApollo\tBlas\tDaniel\tJuliette" -> ATBD: Fase di training; tabella verbatim.
L173: [blank] -> separatore.
L174: "ids are from TRACKS_CL7 file, and the case studies not included in Flaounas et al. (2023) Apollo," -> ATBD: Fase di training.
L175: "Blas, Daniel, Juliette, are based on the minimum MSLP ERA5 reanalysis." -> ATBD: Fase di training.
L176: "One training set and two validation sets were built, the difference between the two validations" -> ATBD: Fase di training.
L177: "sets being the balancing of positive and negative classes, as shown in the following table:" -> ATBD: Fase di training.
L178: [blank] -> separatore.
L179: "\t\t\tNum. cyclones \tTotal time interval \tNum. Video clips" -> ATBD: Fase di training; tabella verbatim.
L180: "Train set   12 \t\t\t\t23 days 8h45’ \t\t\t1238 \t\t\t\tBalanced" -> ATBD: Fase di training; tabella verbatim.
L181: "Val  set    3 \t\t\t\t7 days 8h5' \t\t\t354 \t\t\t\tBalanced" -> ATBD: Fase di training; tabella verbatim.
L182: "Test set    3 \t\t\t\t8 days 13h25' \t\t\t2400 \t\t\t\t201 positives (cyclones)" -> ATBD: Fase di training; tabella verbatim.
L183: "\t\t\t\t\t\t\t\t\t\t\t2199 negatives (no cyclones)" -> ATBD: Fase di training; tabella verbatim.
L184: [blank] -> separatore.
L185: [blank] -> separatore.
L186: "In the next tables it is shown the time window for some example medicane, before and after the" -> ATBD: Fase di training.
L187: "narrower time window selection:" -> ATBD: Fase di training.
L188: "(tabella completa medicanes_new_windows.csv)" -> ATBD: Fase di training; segnaposto verbatim.
L189: [blank] -> separatore.
L190: "The dataset is also partitioned. This means that cyclones’ events belonging in the training dataset" -> ATBD: Fase di training.
L191: "are keep distinct from cyclones considered in the testing dataset to avoid data leakage and" -> ATBD: Fase di training.
L192: "ensure robustness of evaluation." -> ATBD: Fase di training.
L193: [blank] -> separatore.
L194: [blank] -> separatore.
L195: [blank] -> separatore.
L196: "Tracking:" -> ATBD: Fase di training.
L197: "Another second stage training was performed for the regression task of tracking the cyclone" -> ATBD: Fase di training.
L198: "center. For this training another working dataset is built from the source dataset, simpler than" -> ATBD: Fase di training.
L199: "the classification dataset, since it consists only of video tiles containing cyclone with their rotation" -> ATBD: Fase di training.
L200: "center clearly visibile, and the center track labeled based on TracksCL7. The model used is the" -> ATBD: Fase di training.
L201: "same trained in the first stage." -> ATBD: Fase di training.
L202: [blank] -> separatore.
L203: "The dataset has the following number of samples:" -> ATBD: Fase di training.
L204: "Training set samples: 835" -> ATBD: Fase di training; segnaposto verbatim.
L205: "Test set samples: 280" -> ATBD: Fase di training; segnaposto verbatim.
L206: [blank] -> separatore.
L207: "...following activity:  devo completare cosa ho fatto" -> ATBD: Fase di training; segnaposto verbatim.
L208: [blank] -> separatore.
L209: "In the following figure some dataset samples are shown, along with their labeled center." -> ATBD: Fase di training; segnaposto verbatim.
L210: "Fig. 5.4 tracking samples – Example frames from tracking video dataset. The red dot represents the center track label." -> ATBD: Fase di training; segnaposto verbatim.
L211: [blank] -> separatore.
L212: [blank] -> separatore.
L213: [blank] -> separatore.

## annex_videomaev2.md -> ATBD_DeMeTrA_ATBD.md
L1: "In the following a few key elements of the VideoMAE transformer-based model are provided." -> ATBD: Algoritmo.
L2: [blank] -> separatore.
L3: "- How Transformers Work and Why They Outperform CNNs" -> ATBD: Algoritmo.
L4: [blank] -> separatore.
L5: "Transformers rely on the self-attention mechanism, which computes dependencies between" -> ATBD: Algoritmo.
L6: "different parts of an image or video sequence. Unlike CNNs, which are constrained to local" -> ATBD: Algoritmo.
L7: "feature extraction, transformers can:" -> ATBD: Algoritmo.
L8: "1. Model Long-Range Dependencies: Each patch in an image or frame in a video interacts" -> ATBD: Algoritmo.
L9: "directly with all other patches." -> ATBD: Algoritmo.
L10: "2. Learn Contextual Relationships: Instead of static filters, transformers dynamically adjust" -> ATBD: Algoritmo.
L11: "feature weights based on the entire input." -> ATBD: Algoritmo.
L12: "3. Scale Effectively: Larger datasets and models (e.g., ViT-Large, ViT-Huge) lead to improved" -> ATBD: Algoritmo.
L13: "performance without excessive overfitting." -> ATBD: Algoritmo.
L14: "This architecture has been particularly effective for video understanding, where temporal" -> ATBD: Algoritmo.
L15: "relationships between frames are crucial." -> ATBD: Algoritmo.
L16: [blank] -> separatore.
L17: "A review that highlights this topic is well reported by:" -> ATBD: Algoritmo.
L18: "Comparing Vision Transformers and Convolutional Neural Networks for Image" -> ATBD: Algoritmo.
L19: "Classification: A Literature Review, J. Maurício, I. Domingues, J. Bernardino -" -> ATBD: Algoritmo.
L20: "https://www.mdpi.com/2225-1154/12/12/220" -> ATBD: Algoritmo e References e panorama della letteratura.
L21: [blank] -> separatore.
L22: "- VideoMAE: A Transformer-Based Model for Video Understanding" -> ATBD: Algoritmo.
L23: [blank] -> separatore.
L24: "Video Masked Autoencoders (VideoMAE) are a self-supervised learning approach designed for" -> ATBD: Algoritmo.
L25: "efficient video representation learning. They follow the success of Masked Autoencoders (MAE)" -> ATBD: Algoritmo.
L26: "for images, but introduce a novel tube masking strategy to handle video data." -> ATBD: Algoritmo.
L27: "Key Features of VideoMAE" -> ATBD: Algoritmo.
L28: "1. High Masking Ratio (90-95%): Unlike image-based MAEs (which mask ~75% of tokens)," -> ATBD: Algoritmo.
L29: "VideoMAE can mask up to 95% of the input video tokens due to the redundancy in video" -> ATBD: Algoritmo.
L30: "frames." -> ATBD: Algoritmo.
L31: "2. Tube Masking: Instead of randomly dropping individual pixels or patches, VideoMAE" -> ATBD: Algoritmo.
L32: "masks entire spatiotemporal cubes (tubes) across frames, making the reconstruction task" -> ATBD: Algoritmo.
L33: "more challenging." -> ATBD: Algoritmo.
L34: "3. Asymmetric Encoder-Decoder Architecture:" -> ATBD: Algoritmo.
L35: "• The encoder processes only the visible (unmasked) tokens, making training efficient." -> ATBD: Algoritmo.
L36: "• The decoder reconstructs missing patches using a lightweight architecture, reducing" -> ATBD: Algoritmo.
L37: "computational costs." -> ATBD: Algoritmo.
L38: [blank] -> separatore.
L39: "- Why VideoMAE is Effective for Short Video Classification" -> ATBD: Algoritmo.
L40: [blank] -> separatore.
L41: [blank] -> separatore.
L42: "• Efficient Representation Learning: By training on a self-supervised pretext task" -> ATBD: Algoritmo.
L43: "(reconstructing missing video frames), the model learns rich spatiotemporal features without" -> ATBD: Algoritmo.
L44: "requiring labeled data." -> ATBD: Algoritmo.
L45: "• Scalability: VideoMAE can be scaled to large datasets (e.g., Kinetics-400, Something-" -> ATBD: Algoritmo.
L46: "Something V2) and larger architectures (ViT-L, ViT-H, ViT-g) to achieve state-of-the-art" -> ATBD: Algoritmo.
L47: "performance." -> ATBD: Algoritmo.
L48: "• Transferability: Pre-trained VideoMAE models perform well on various downstream tasks," -> ATBD: Algoritmo.
L49: "including video classification, action recognition, and spatiotemporal localization." -> ATBD: Algoritmo.
L50: [blank] -> separatore.
L51: "- VideoMAE Architecture Overview" -> ATBD: Algoritmo.
L52: [blank] -> separatore.
L53: "The Video Masked Autoencoder (VideoMAE) is a self-supervised learning framework designed to" -> ATBD: Algoritmo.
L54: "efficiently pre-train vision transformers for video understanding. Its tube masking strategy, joint" -> ATBD: Algoritmo.
L55: "space-time attention, and asymmetric encoder-decoder architecture make it one of the most" -> ATBD: Algoritmo.
L56: "effective models for learning spatiotemporal representations. Below is an in-depth look at its" -> ATBD: Algoritmo.
L57: "architecture based on the papers you provided." -> ATBD: Algoritmo.
L58: "- 1. Key Components of VideoMAE" -> ATBD: Algoritmo.
L59: "VideoMAE consists of three major components:" -> ATBD: Algoritmo.
L60: "1. Cube Embedding – Converts input frames into a sequence of tokens." -> ATBD: Algoritmo.
L61: "2. Encoder (Masked ViT) – Processes only visible tokens to extract high-level features." -> ATBD: Algoritmo.
L62: "3. Decoder (Masked Reconstruction) – Reconstructs missing patches using a lightweight" -> ATBD: Algoritmo.
L63: "architecture." -> ATBD: Algoritmo.
L64: "- 2. Cube Embedding (Patch Tokenization)" -> ATBD: Algoritmo.
L65: "Before processing, VideoMAE divides each video into small spatiotemporal cubes (patches)." -> ATBD: Algoritmo.
L66: "These cubes are converted into tokens and serve as input for the transformer." -> ATBD: Algoritmo.
L67: "Patch Size: Each cube is 2 × 16 × 16 pixels (2 frames, 16×16 resolution)." -> ATBD: Algoritmo.
L68: "Feature Representation: Each patch is embedded into a vector and assigned a position" -> ATBD: Algoritmo.
L69: "encoding." -> ATBD: Algoritmo.
L70: "Reduction of Spatial and Temporal Redundancy: Helps eliminate unnecessary information" -> ATBD: Algoritmo.
L71: "from static parts of videos" -> ATBD: Algoritmo.
L72: "- 3. Encoder: Masked Spatiotemporal Transformer" -> ATBD: Algoritmo.
L73: "The encoder follows a Vision Transformer (ViT) backbone with a joint space-time attention" -> ATBD: Algoritmo.
L74: "mechanism, which allows it to model complex video dynamics." -> ATBD: Algoritmo.
L75: "Joint Space-Time Attention: Unlike CNNs, which process spatial features independently," -> ATBD: Algoritmo.
L76: "VideoMAE attends over both spatial and temporal dimensions simultaneously." -> ATBD: Algoritmo.
L77: "Masked Token Strategy: Only 10% of input tokens are processed in the encoder, while the" -> ATBD: Algoritmo.
L78: "rest are dropped, significantly improving efficiency" -> ATBD: Algoritmo.
L79: "- 4. Tube Masking Strategy" -> ATBD: Algoritmo.
L80: "A key innovation in VideoMAE is the Tube Masking Strategy, which significantly differs from" -> ATBD: Algoritmo.
L81: "conventional patch masking in images." -> ATBD: Algoritmo.
L82: "Why Tube Masking?" -> ATBD: Algoritmo.
L83: "In images, missing patches can be inferred from surrounding areas." -> ATBD: Algoritmo.
L84: "In videos, due to temporal redundancy, masking individual patches is too easy for" -> ATBD: Algoritmo.
L85: "reconstruction models." -> ATBD: Algoritmo.
L86: "How It Works?" -> ATBD: Algoritmo.
L87: "Instead of randomly masking individual pixels or patches, entire spatiotemporal tubes" -> ATBD: Algoritmo.
L88: "(continuous regions in time and space) are masked." -> ATBD: Algoritmo.
L89: "This makes the task harder and forces the model to learn robust video representations" -> ATBD: Algoritmo.
L90: "- 5. Decoder: Lightweight Reconstruction" -> ATBD: Algoritmo.
L91: "The decoder attempts to reconstruct the missing patches given only a fraction of the original" -> ATBD: Algoritmo.
L92: "input." -> ATBD: Algoritmo.
L93: "•" -> ATBD: Algoritmo.
L94: "•" -> ATBD: Algoritmo.
L95: "Small and Efficient: Unlike the encoder, which operates on only 10% of the patches, the" -> ATBD: Algoritmo.
L96: "decoder operates on 100% of the tokens but is significantly smaller in depth and width." -> ATBD: Algoritmo.
L97: "Asymmetric Architecture:" -> ATBD: Algoritmo.
L98: "    Encoder: 12 blocks (ViT-Base)" -> ATBD: Algoritmo.
L99: "    Decoder: 4 blocks (lighter, but sufficient for reconstruction)" -> ATBD: Algoritmo.
L100: "    This asymmetry drastically reduces computation time" -> ATBD: Algoritmo.
L101: [blank] -> separatore.
L102: "- 6. Dual Masking in VideoMAE V2" -> ATBD: Algoritmo.
L103: "The second iteration of VideoMAE (VideoMAE V2) introduces a dual masking approach, further" -> ATBD: Algoritmo.
L104: "improving efficiency:" -> ATBD: Algoritmo.
L105: "    Encoder Masking: 90% of tokens are dropped before entering the encoder." -> ATBD: Algoritmo.
L106: "    Decoder Masking: Instead of reconstructing the entire frame, only a subset of missing cubes is reconstructed." -> ATBD: Algoritmo.
L107: "    Benefit: This further reduces memory usage and speeds up training, allowing VideoMAE to scale to billion-parameter video models" -> ATBD: Algoritmo.
L108: [blank] -> separatore.
L109: "- 7. Computational Efficiency" -> ATBD: Algoritmo.
L110: "One of the most significant advantages of VideoMAE is its ability to train efficiently on large-scale" -> ATBD: Algoritmo.
L111: "video datasets." -> ATBD: Algoritmo.
L112: "- Asymmetric Architecture: The small decoder reduces computational overhead while still" -> ATBD: Algoritmo.
L113: "providing high-quality reconstructions." -> ATBD: Algoritmo.
L114: "- Efficient Self-Supervised Pre-Training: No labeled data is required—models learn" -> ATBD: Algoritmo.
L115: "representations by reconstructing missing patches." -> ATBD: Algoritmo.
L116: "- Scalability: VideoMAE can scale from small models (ViT-B) to billion-parameter models" -> ATBD: Algoritmo.
L117: "(ViT-g)" -> ATBD: Algoritmo.
L118: [blank] -> separatore.

## AI_4_medicanes.md -> ATBD_DeMeTrA_ATBD.md
L1: "Artificial Intelligence applied to Atmospheric Science is quite a new field, here it is a brief list of" -> ATBD: References e panorama della letteratura.
L2: "recent works in this field." -> ATBD: References e panorama della letteratura.
L3: [blank] -> separatore.
L4: "A Comprehensive AI Approach for Monitoring and Forecasting Medicanes Development (2024)," -> ATBD: References e panorama della letteratura.
L5: "J. Martinez-Amaya; V. Nieves; J. Muñoz-Mari" -> ATBD: References e panorama della letteratura.
L6: "Methods: Tracking: k-means clustering; Forecast: CNN + Random Forest" -> ATBD: References e panorama della letteratura.
L7: "Dataset: 58 medicanes (1984–2023); Meteosat IR immagini (BT); reanalysis CERRA & ERA5" -> ATBD: References e panorama della letteratura.
L8: "(MSLP, wind)" -> ATBD: References e panorama della letteratura.
L9: "- First ML model dedicated to Medicanes. Detects cyclone location and structural features" -> ATBD: References e panorama della letteratura.
L10: "from IR images, with data-driven prediction of intensification. It can predict extreme" -> ATBD: References e panorama della letteratura.
L11: "medicanes events up to 2 days in advance with 65-80% accuracy, offering an innovative" -> ATBD: References e panorama della letteratura.
L12: "alternative to traditional models and adaptable to climate change." -> ATBD: References e panorama della letteratura.
L13: "https://www.mdpi.com/2225-1154/12/12/220" -> ATBD: References e panorama della letteratura.
L14: [blank] -> separatore.
L15: "•A Statistical Learning Approach to Mediterranean Cyclones (2025), L. Roveri; L. Fery; L." -> ATBD: References e panorama della letteratura.
L16: "Cavicchia; F. Grotto" -> ATBD: References e panorama della letteratura.
L17: "Methods: Unsupervised: Latent Dirichlet Allocation (LDA) for dimensional reduction;" -> ATBD: References e panorama della letteratura.
L18: "Supervised: Statistical classifier on LDA feature" -> ATBD: References e panorama della letteratura.
L19: "Dataset: ERA5 reanalysis (wind fields and pressure) + Mediterranean cyclone track archive" -> ATBD: References e panorama della letteratura.
L20: "- New Mediterranean cyclone classification workflow. LDA extracts cyclonic wind patterns by" -> ATBD: References e panorama della letteratura.
L21: "drastically reducing data size, then a classifier identifies cyclones and tracks. Result: ~90%" -> ATBD: References e panorama della letteratura.
L22: "accuracy in detecting Mediterranean cyclones using a few key features. The approach" -> ATBD: References e panorama della letteratura.
L23: "overcomes difficulties in defining shape/diameter of medicanes, implicitly identifying typical" -> ATBD: References e panorama della letteratura.
L24: "structures and paving the way for temporal precursors of extreme cyclones." -> ATBD: References e panorama della letteratura.
L25: "https://arxiv.org/pdf/2501.15694v1" -> ATBD: References e panorama della letteratura.
L26: [blank] -> separatore.
L27: [blank] -> separatore.
L28: "Deepti: Deep-Learning-Based Tropical Cyclone Intensity Estimation System (2020), M. Maskey;" -> ATBD: References e panorama della letteratura.
L29: "R. Ramachandran; M. Ramasubramanian; I. Gurung; et al." -> ATBD: References e panorama della letteratura.
L30: "Methods: CNN (custom model) for intensity regression (max wind)" -> ATBD: References e panorama della letteratura.
L31: "Dataset: IR GOES satellite images (15-min, multi-geostaz. 2000-2019); Best-track HURDAT2" -> ATBD: References e panorama della letteratura.
L32: "(intensity)" -> ATBD: References e panorama della letteratura.
L33: "- Automated intensity estimation from IR images. CNN learns the relationship between cloud" -> ATBD: References e panorama della letteratura.
L34: "patterns and maximum wind, emulating human (Dvorak) analysis. It obtains an RMSE error of" -> ATBD: References e panorama della letteratura.
L35: "13.24 knots on wind speed, comparable to operational techniques. Provides objective" -> ATBD: References e panorama della letteratura.
L36: "estimates in near-real time and a visualization portal, representing one of the first fully data-" -> ATBD: References e panorama della letteratura.
L37: "driven systems for cyclone intensity assessment" -> ATBD: References e panorama della letteratura.
L38: "https://ieeexplore.ieee.org/document/9149719" -> ATBD: References e panorama della letteratura.
L39: [blank] -> separatore.
L40: "A Novel Deep Learning Based Model for Tropical Intensity Estimation and Post-Disaster" -> ATBD: References e panorama della letteratura.
L41: "Management of Hurricanes (2021), J. Devaraj; S. Ganesan; R. M. Elavarasan; U. Subramaniam" -> ATBD: References e panorama della letteratura.
L42: "Methods: Enhanced CNN (batch normalization + dropout) for intensity; Transfer Learning:" -> ATBD: References e panorama della letteratura.
L43: "VGG-19 fine-tuning for damage and event classification" -> ATBD: References e panorama della letteratura.
L44: "Dataset: IR GOES satellite images + HURDAT2 label (hurricanes 2000-2019); Post-event" -> ATBD: References e panorama della letteratura.
L45: "satellite images (e.g., Houston) for damage; Severe weather video for classification" -> ATBD: References e panorama della letteratura.
L46: "- Improves the state of the art in intensity and impact. Optimized CNN reduces intensity" -> ATBD: References e panorama della letteratura.
L47: "estimation error to 7.6 knots RMSE (about half that of previous models), accurately" -> ATBD: References e panorama della letteratura.
L48: "distinguishing the hurricane category. Moreover, by fine-tuning VGG-19 on impact data, it" -> ATBD: References e panorama della letteratura.
L49: "achieves 98% accuracy in predicting hurricane damage and 97% accuracy in classifying severe" -> ATBD: References e panorama della letteratura.
L50: "weather events. The integrated approach provides both more accurate forecasts and post-" -> ATBD: References e panorama della letteratura.
L51: "event information for disaster management." -> ATBD: References e panorama della letteratura.
L52: "https://www.mdpi.com/2076-3417/11/9/4129" -> ATBD: References e panorama della letteratura.
L53: [blank] -> separatore.
L54: [blank] -> separatore.
L55: "Tropical and Extratropical Cyclone Detection Using Deep Learning (2020), C. Kumler-Bonfanti; J." -> ATBD: References e panorama della letteratura.
L56: "Stewart; D. Hall; M. Govett" -> ATBD: References e panorama della letteratura.
L57: "Methods: Segmentation (U-Net) - 4 U-Net variants to identify cyclone regions; label" -> ATBD: References e panorama della letteratura.
L58: "comparison from IBTrACS vs. heuristic algorithm" -> ATBD: References e panorama della letteratura.
L59: "Dataset: Input: Global maps of total precipitable water (GFS 0.5°) and GOES satellite images" -> ATBD: References e panorama della letteratura.
L60: "(water vapor channel); Ground truth: IBTrACS (official TCs tracks) + heuristic detection of" -> ATBD: References e panorama della letteratura.
L61: "extratropical cyclones." -> ATBD: References e panorama della letteratura.
L62: "- Automated detection of cyclones on a global scale. U-Net models highlight cyclonic ROIs" -> ATBD: References e panorama della letteratura.
L63: "more quickly and accurately than traditional methods. They achieve 80-99% accuracy in" -> ATBD: References e panorama della letteratura.
L64: "marking cyclone regions (Dice coefficient 0.51-0.76), including many weak vortices ignored by" -> ATBD: References e panorama della letteratura.
L65: "manual approaches. The model for extratropical cyclones is 3 times faster than the operational" -> ATBD: References e panorama della letteratura.
L66: "reference algorithm. The use of multisource inputs (model + satellite) allows AI to discover" -> ATBD: References e panorama della letteratura.
L67: "“ambiguous” cyclones that escaped the strict criteria, improving coverage of alerted systems." -> ATBD: References e panorama della letteratura.
L68: "https://colab.ws/articles/10.1175%2Fjamc-d-20-0117.1" -> ATBD: References e panorama della letteratura.
L69: [blank] -> separatore.
L70: [blank] -> separatore.
L71: "A Hybrid ML/Physics-Based Modeling Framework for 2-Week Extended Prediction of Tropical" -> ATBD: References e panorama della letteratura.
L72: "Cyclones (2024), X. Liu; et al. (Studio JGR)" -> ATBD: References e panorama della letteratura.
L73: "Methods: Hybrid model: coupling of a numerical model (WRF, ~2 km) with a global deep" -> ATBD: References e panorama della letteratura.
L74: "learning model (Pangu-Weather, ~25 km)" -> ATBD: References e panorama della letteratura.
L75: "Dataset: High-resolution WRF simulations + ML forecasts (Pangu); tests on real cases such as" -> ATBD: References e panorama della letteratura.
L76: "Cyclone Freddy (2023)" -> ATBD: References e panorama della letteratura.
L77: "- Extends the forecast horizon for cyclones. By combining high-resolution physics with AI, the" -> ATBD: References e panorama della letteratura.
L78: "framework correctly anticipates cyclone trajectory and intensity up to ~2 weeks, compared to" -> ATBD: References e panorama della letteratura.
L79: "~5 days for traditional models. In the case study on Freddy, the hybrid solution dramatically" -> ATBD: References e panorama della letteratura.
L80: "improves track and intensity over WRF or Pangu alone, extending accuracy from ~5 to 7 days" -> ATBD: References e panorama della letteratura.
L81: "and maintaining reliable forecasts over the entire 14-day period. This pioneering approach" -> ATBD: References e panorama della letteratura.
L82: "demonstrates that AI can enhance physics models, moving forward the limit of the state of the" -> ATBD: References e panorama della letteratura.
L83: "art in long-term forecasting." -> ATBD: References e panorama della letteratura.
L84: "https://agupubs.onlinelibrary.wiley.com/doi/10.1029/2024JH000207" -> ATBD: References e panorama della letteratura.
L85: [blank] -> separatore.
L86: "Tropical cyclone size estimation based on deep learning using infrared and microwave satellite" -> ATBD: References e panorama della letteratura.
L87: "data (2023), J. Xu; X. Wang; H. Wang; C. Zhao; H. Wang; J. Zhu" -> ATBD: References e panorama della letteratura.
L88: "Methods: TC-ResNet: modified ResNet-50 (conv 5×5 on shortcut + dual attention canal/space)" -> ATBD: References e panorama della letteratura.
L89: "for wind radius regression" -> ATBD: References e panorama della letteratura.
L90: "Dataset: Multi-satellite data: geostationary IR images and passive microwave maps (2003-" -> ATBD: References e panorama della letteratura.
L91: "2017) + global R34 wind ray dataset (best-track)" -> ATBD: References e panorama della letteratura.
L92: "- Objective estimation of cyclone size (wind radius 34 kt). By integrating IR and microwave information," -> ATBD: References e panorama della letteratura.
L93: "the model learns to filter out background noise and blurring in cyclonic clouds." -> ATBD: References e panorama della letteratura.
L94: "Result: mean error 11.3 nm (≈21 km) on radius R34, with correlation coefficient 0.907, higher" -> ATBD: References e panorama della letteratura.
L95: "than traditional methods based on empirical parameters. The dual focus and multi-source data" -> ATBD: References e panorama della letteratura.
L96: "allow TC-ResNet to better capture the true wind extent, improving the risk assessment (wind," -> ATBD: References e panorama della letteratura.
L97: "swell) related to cyclone size compared to the previous state of the art." -> ATBD: References e panorama della letteratura.
L98: "https://www.frontiersin.org/journals/marine-" -> ATBD: References e panorama della letteratura.
L99: "science/articles/10.3389/fmars.2022.1077901/full" -> ATBD: References e panorama della letteratura.
L100: [blank] -> separatore.
L101: "Detecting Extratropical Cyclones of the Northern Hemisphere with Single Shot Detector (2022)," -> ATBD: References e panorama della letteratura.
L102: "M. Shi; P. He; Y. Shi; et al." -> ATBD: References e panorama della letteratura.
L103: "Methods: Object detection (One-Stage): CNN Single Shot Detector (SSD) adapted to detect" -> ATBD: References e panorama della letteratura.
L104: "cyclone centers and phase" -> ATBD: References e panorama della letteratura.
L105: "Dataset: Whole-disk (visible/IR) satellite images of the Northern Hemisphere; manually labeled" -> ATBD: References e panorama della letteratura.
L106: "dataset with cyclogenesis, maturity, and dissipation (criteria based on Bonfanti 2017)" -> ATBD: References e panorama della letteratura.
L107: "- First application of SSD for extratropical cyclones. Innovative labeling and training pipeline" -> ATBD: References e panorama della letteratura.
L108: "enables the model to automatically detect low pressure centers in satellite images and classify" -> ATBD: References e panorama della letteratura.
L109: "their evolutionary stage. Achieves high performance: mAP 86.6% in recognizing mature" -> ATBD: References e panorama della letteratura.
L110: "cyclones and ~79.3% considering all three classes (developing, mature, fading). This" -> ATBD: References e panorama della letteratura.
L111: "demonstrates the ability of deep learning to reliably identify large-scale extratropical cyclones," -> ATBD: References e panorama della letteratura.
L112: "which has been a complex challenge so far due to shape/size variability. The SSD model shows" -> ATBD: References e panorama della letteratura.
L113: "great potential for automating the monitoring of these storms in future weather operations." -> ATBD: References e panorama della letteratura.
L114: "https://www.mdpi.com/2072-4292/14/2/254" -> ATBD: References e panorama della letteratura.
