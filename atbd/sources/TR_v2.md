5.NRT detection and tracking (WP2400)
5.1.Introduction
WP 2400 is dedicated to the development of Deep Learning techniques for the detection,
monitoring and tracking of medicanes. The main focus of the WP2400 is the development, testing
and validation of the Deep-learning Medicane Tracking Algorithm (DeMeTrA),exploiting IR
images derived from geostationary satellites observing the Mediterranean area (i.e. Meteosat
Second Generation (MSG) – Spinning Enhanced Visible InfraRed Imager SEVIRI). The main
capability of the DeMeTrA algorithm will be:
1. to detect the presence of Mediterranean cyclones, and to identify their center of rotation (RC);
2. to track the position of the cyclone’s RC;
3. to early identify the cyclones evolving in tropical-like cyclone (detection of the closed eye).
The training dataset has been built from IR measurements available from the Spinning Enhanced
Visible InfraRed Imager SEVIRI sensor of the MSG Rapid Scan Service (RSS), collecting data with 5
minutes frequency, useful for monitoring rapidly evolving meteorological phenomena. The
dataset includes also labels of medicane’s occurrence (based on WP2300 output) and the coordinates of the cyclones’ center of rotation (RC) obtained from independent observations, as
well as the model-based tracks available for all cyclones considered in the study used as
reference.

5.2.
Deep learning for NRT tracking: DeMeTrA
The DeMeTrA ML-based tracking algorithm by means of specialization in cyclone recognition is
currently based on VideoMAEv2.
The VideoMAEv2 model is a transformer-based self-supervised learning architecture specifically
designed for video data. It efficiently captures spatio-temporal features by masking input patches
and reconstructing the missing content. The pre-trained version of the model processes video
clips consisting of 16 frames, each with dimensions of 224×224 pixels. Each frame is then
subdivided, during the forward pass, into non-overlapping patches of size 14x14 pixels, forming
the basic units for input to the transformer network. The architecture includes:
•an encoder-decoder transformer structure.
•tube masking strategy, where spatial and temporal patches (tubelets) are masked to encourage the model to learn robust video representations.
Technical details and deeper insights can be found in ANNEX A, and in the related VideoMAEv2
repository on GitHub (https://github.com/dandarm/VideoMAEv2) specifically built for this project, and the original research paper on Arxiv.

(figura VideoMAEv2_flowchart.png)
Fig. 5.1 – Technical details of the VideoMAEv2 model: encoder-decoder structure, with separate masking for
each one. The latent representation vector (embedding) is shown between the encoder and the decoder.

5.2.1. VideoMAE training Dataset
The training dataset has been built from IR measurements by EUMETSAT Rapid Scan High Rate
SEVIRI Level 1.5 Image Data MSG (Rapid Scan Service: RSS) with high frequency availability (every
5 minutes) at high spatial resolution (3 km at s.s.p.), useful for monitoring rapidly evolving
meteorological phenomena. Data is downloadable from the Google Cloud platform “Big Query
Public data”, with permission by EUMETSAT to redistribute data, provided by Open Climate Fix,
that processed the data further using satip and SatPy library.
The original data have been slightly transformed: the numerical values produced by the sensor
have been rescaled with SatPy python library to produce calibrated Brightness Temperatures
values, and the pixel values have been linearly mapped to the range [0, 1023] (i.e. 10 bits per
channel).. The dataset was built using the Zarr format, and has not been spatially reprojected
(and so is in "geostationary" projection).
Therefore data needs to de-normalized by means the following formula:
𝐵𝑇 = v (𝑥𝑚𝑎𝑥 − 𝑥𝑚𝑖𝑛) + 𝑥𝑚𝑖𝑛

where v is the rescaled value ranging in the [0, 1] domain, while xmax and xmin are the arrays of
maximum and minimum values, respectively, and they are specific for each channel, as shown in
the following Table 5.1:

SEVIRI channel    Xmin      Xmax
IR_097     2,84         317,87
IR_108     199,10       313,28
WV_062     199,57       249,92
WV_073     198,95       286,96

Table 5.1 – Maximum and minimum values for the channels of interest.




These infrared and water vapor channels are further pre-processed, and are combined into the
Airmass RGB composite. According to EUMETSAT guidelines (RGB Recipes https://eumetrain.org/sites/default/files/2020-05/RGB_recipes.pdf and Best Practices  https://www-cdn.eumetsat.int/files/2020-04/pdf_using_rgb_best_practices.pdf),
Airmass RGB enhances visualization of air masses and frontal systems, facilitating the visual
identification of cyclone dynamics and atmospheric moisture, as well as cloud features, by
integrating:
•Red channel: infrared water vapour channel around 6.2 µm (WV_062) – upper-level water
vapor 7.3 µm (WV_073);
• Green channel: difference between channels sensitive to ozone absorption (IR_097 -
IR_108), highlighting stratospheric air intrusions;
• Blue channel: WV_062.
This combination is highly effective for medicane detection and analysis, and so it is used to build
up the entire dataset
Then, spatial coordinates were cropped to a geostationary projection covering the
Mediterranean area (Fig. 5.2), specifically:


• Latitude range: 30°– 48° N
• Longitude range: from -7° – 46°

Fig. 5.2 – Air mass RGB example image covering the Mediterranean basin, with superimposed borders. Image time: 17th September 2020, 03:40 UTC


By using this spatial selection, each image of the Mediterranean region consists of 1290×420 pixels

A python script was developed and refined for efficient downloading and processing of the data
as described, and the resulting collected images sum up for a total of nearly 860’000,
corresponding to 600 Gbyte of data, spanning more the 7 years and half of time interval, in a
wider range between 2010 and 2023 years.

This dataset is named ‘Source Dataset’ to distinguish it from the further processing that enables
it to become input for the model as training set and validation sets. The procedures and code
responsible for this next step, must be efficient as well; in order to be executed every time a
parameter update is needed to build it. This last dataset is named ‘Working Dataset’, there are
many of them, one for each trial design choice.

These procedures split the entire images into ‘tiles', small squares of 224×224 pixel, with partial
overlapping that results into 12 tiles, this number can change depending on the overlapping
parameters choice; after that, 16 tiles are stacked to form a single video clip, each frame being
equally spaced temporally by 5 minutes. This resulting video is an input sample to the model. The
side length and the total number of frames are fixed parameters needed by the pretrained
model. A single tile covers an area of approximately 757 km latitude by 805 km longitude The
spatial resolution of each pixel is about 3.38 km in latitude and 3.59 km in longitude. Each video
clip spans a time window of 80 minutes, and there can be an overlap in time among video tiles,
too.

Further labeling process is required to build the working dataset. For this purpose, the TracksCL7
database is used (Flaounas et al., 2023), from which cyclone center coordinates are taken to
determine the presence of cyclones in each video tiles given its pixel offset respect to the cropped
angles.

A proper coordinate transformation from geospatial latitude and longitude to pixel domain is
performed for this task, and the positive labels are therefore assigned where and when the
center cyclone track is included into the tile, for at least 6 frames out of 16. All other videos are
marked with a negative label, for this binary classification task. A further subdivision in more
classes could be made for cyclone detection task, but the binary classification is the simplest trial
to start with.

This labeling process is needed for the supervised training, while the self-supervised training
does not need labels, so the information regarding cyclones position is not considered.

This ’tiling’ and labeling algorithms applied to the source dataset produce the working dataset.
In the following fig 5.3 it is shown the process of tiling and labeling from a source image.

Fig. 5.2 – Airmass RGB source frame: the squares are sliced to form video clips in the same position for 16 frames.
A green square is shown where the video tile is selected as positive sample, containing a cyclone center track
(cyan dots).

A two-stage training is chosen to exploit the big dataset available: the first stage is self-
supervised, and includes a wide variety of Airmass RGB images, with the purpose of building a
specialized model on such kind of images. We used the ‘giant’ pre-trained model for this
unsupervised “specialization” pre-training, which is a training in addition to the pre-training from
scratch, in order to increase the model feature extraction capability for the second stage
trainings, necessary for detection and tracking tasks.
The dataset for the first stage self-supervised training is very big since it doesn’t need labels or
balancing, with a total number of samples:
80’000 for the training set
20’000 for the test set

After that, the second stagesupervised classification training is performed,
using the model trained in the first stage, with a balanced dataset having size of nearly 3000
videos, divided evenly between two classes:


•class 1: includes cyclones recorded in the TRACKS_CL7 database, including both
medicanes and other Mediterranean cyclones.
•class 0: no cyclones occurred

This was only the first trial. After that, further dataset processing was needed, due to noise and
labeling mismatch with respect to the information contained in the airmassRGB images.
Such a mismatch was found in many occurrences, where a cyclone center was provided without
visible clouds, or without rotating cyclone formations, only translatory moving clouds.
For this reason, a more limited dataset consisting of TrackCL10 cyclones was taken into account
for a second trial training. Other refinements were necessary to increase the training and
generalization performance, with accurate visual inspection for the entire dataset.
This analysis led to another version of the dataset including only clearer medicanes in a third trial
of trainings, excluding almost all tracks from CL10 not identified as medicanes. Moreover the
time window of the medicane occurrences was reduced by eliminating the initial and final stages
with respect to the CL10 track, maintaining the frames when a visible clear cloud rotation can be
observed. This was necessary to achieve a satisfactory performance for the detection task.

The final dataset consists of the following 18 cyclones:
ID 		1283	1328	1358	1421	1461		1466	1500	1521 	1542
Name 	Unnamed Rolf 	Unnamed Unnamed Qendresa	Unnamed Unnamed Unnamed Trixie
ID 		1575 	1674	1702	1715	1716		-		-		-		-
Name 	Numa 	Ianos 	Unnamed Unnamed Unnamed 	Apollo	Blas	Daniel	Juliette

ids are from TRACKS_CL7 file, and the case studies not included in Flaounas et al. (2023) Apollo,
Blas, Daniel, Juliette, are based on the minimum MSLP ERA5 reanalysis.
One training set and two validation sets were built, the difference between the two validations
sets being the balancing of positive and negative classes, as shown in the following table:

			Num. cyclones 	Total time interval 	Num. Video clips
Train set   12 				23 days 8h45’ 			1238 					Balanced
Val  set    3 				7 days 8h5' 			354 					Balanced
Test set    3 				8 days 13h25' 			2400 					201 positives (cyclones)
																			2199 negatives (no cyclones)


In the next tables it is shown the time window for some example medicane, before and after the
narrower time window selection:
(tabella completa medicanes_new_windows.csv)

The dataset is also partitioned. This means that cyclones’ events belonging in the training dataset
are keep distinct from cyclones considered in the testing dataset to avoid data leakage and
ensure robustness of evaluation.



Tracking:
Another second stage training was performed for the regression task of tracking the cyclone
center. For this training another working dataset is built from the source dataset, simpler than
the classification dataset, since it consists only of video tiles containing cyclone with their rotation
center clearly visibile, and the center track labeled based on TracksCL7. The model used is the
same trained in the first stage.

The dataset has the following number of samples:
Training set samples: 835
Test set samples: 280

...following activity:  devo completare cosa ho fatto

In the following figure some dataset samples are shown, along with their labeled center.
Fig. 5.4 tracking samples – Example frames from tracking video dataset. The red dot represents the center track label.



