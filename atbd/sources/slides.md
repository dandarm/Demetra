1- Main objective

Deep-learning Medicane Tracking Algorithm (DeMeTrA)
Exploits IR images by the Spinning Enhanced Visible InfraRed Imager SEVIRI
on board the Meteosat Second Generation (MSG).

DeMeTrA main objectives
• To detect the presence of Medicanes
- by means of Binary Classification
• To identify and track the position of the medicane’s RC in NRT
- by means of Regression Analysis
Detection is preparatory to tracking



2- Vision Transformers for DeMeTrA: VideoMAE

Vision Transformers (ViTs) have quickly become state of the art in image recognition and video analysis (1)
Video Masked Autoencoder (VideoMAE (2)) is one of the most advanced ViT architectures for video understanding:
• Open source pre-trained model on huge video data
• AutoEncoder architecture for self-supervised learning (do not need labels)
(1) Mauricio et al., 2023
(2) Tong et al., 2022, Wang et al. 2023

DeMeTrA AI-based algorithm is a specialization of
the pre-trained VideoMAEv2 model
for cyclone recognition and tracking
through fine-tuning on our dataset

3- DeMeTrA input data (Airmass RGB)
Input data consists of Airmass RGB composite video clips, built from the
SEVIRI IR measurements from the MSG Rapid Scan Service (RSS)
Multiple IR and water vapor channels are combined into an Airmass RGB composite by integrating:
• infrared water vapour channel ~6.2 µm (WV_062) & upper-level water vapor 7.3 µm (WV_073) → red channel;
• channels sensitive to ozone absorption (IR_097 & IR_108) highlighting stratospheric air intrusions
→ green channel;
• WV_062 → blue channel.
Airmass RGB enhances visualization of air masses, atmospheric moisture and frontal systems facilitating the
visual identification of cyclone dynamics, making it highly effective for medicane detection and analysis.
R = WV_062 – WV_073
G = IR_097 – IR_108
B = WV_062


4- DeMeTrA dataset building
• Big models need big data
• Available dataset from Ifremer was not complete (lack of temporal continuity) and
not enough for a proper training
• We downloaded the EUMETSAT Rapid Scan High Rate SEVIRI Level 1.5 Image Data MSG
data made available by Google Cloud Storage «Big Query Public Data».
Google hosts public and third-party datasets on behalf of their providers, giving users
reliable, large-scale access without the burden of data storage.
• This dataset does not contain the original numerical values,
but it is transformed by Open Climate Fix* (* openclimatefix.org):
pixel values have been calibrated by SatPy to produce
normalized Brightness Temperatures values

5- DeMeTrA dataset building
• Need to download data
• Transform back to original Brightness Temperature values
• Build the composite with proper normalization ranges as in the Meteosat recipe
We have the Source Dataset  (fig airmassRGB senza land borders)

~ 860K AirmassRGB image frames (1290 X 420 pixels) over Mediterranean sea area
~ 600 GByte total size
⪞ 7.5 years total time, between 2010 and 2023


6- DeMeTrA dataset building - tiling
Building the Working Dataset
- Slicing Mediterranean region into 12 partially overlapping tiles - 224 x 224 pixel each
- Stacking 16 frame tiles to build a single video tile (~ 1h 20' time span)
- One video tile each hour: overlapping in time
(pixel size and number of frames are Required by the pre-trained model)
fig tiles-mediterraneo  - fig video-tiles



7- DeMeTrA dataset building - labeling
• Tracks are used to label each tile frame (static) using TRACKS_CL7* (Flaounas et al., 2023)
• Transform track position from geo coordinates (lon, lat) to pixel domain (x,y)
• Higly computationally optimized process:
the Working Dataset has to be rebuilt many times, depending on training design choices
The tile where the track is contained
is marked as label 1 (cyclone)
green square
All other tiles: label 0 (no cyclone)
• Label 1 videos: at least 6 tile frames with cyclone
• Label 0 videos: otherwise
* Mediterranean storms produced through the combination of 7 different cyclone-tracking methods applied to reanalysis (Flaounas et al., 2023)

fig labeled-tiles


8- DeMeTrA training
Three stage training:
• Self-supervised "specialization" training, for data feature learning,
to leverage unlabeled big data available
• Supervised classification training, in order to detect the presence of a cyclone
labels of Mediterranean cyclones occurrence using TRACKS_CL7*
• Supervised regression training, to learn the cyclone center coordinates
the lst two training: Using learned features


9- Working Dataset - details

• Self-supervised dataset: random videos
train set ~ 80K video tiles
test set ~ 20K video tiles
• Supervised binary classification dataset - 4 refining steps:
o Step 1: Tracks CL7
o Step 2: Tracks CL10
o Step 3: Only Medicanes *
o Step 4: Only Medicanes * and narrow time window with clearly observable rotation

* From the Full_List_Medicanes


10- Self-supervised "specialization" training
(masking - reconstruction)

Training and validation losses versus epochs
Long time training (~ days)
Results: no overfitting → good!
→ save this model as a checkpoint for the next stage training


11- Supervised classification training results
Using Tracks CL7: Max. Accuracy 80 %
Need for increased dataset quality: why? because of 
Mismatch with the tracks: IR images may not capture cyclonic dynamics for certain tracks (cloud absence, no cloud rotation...)


12- Dataset refinement: Cyclones new time boundaries (manual selection)
Using Only Medicanes *
• New Start and End times based on visual inspection
• Clearly visible cyclone clouds rotation
(medicane_new_windiws.csv)

13- Final dataset for detection
• 18 cyclones
splitting train/val/test (same shown in TR_v2.md)

14-
Best results: Max. Accuracy 91 %
Balanced dataset (validation set)

Best results: Max. Accuracy 89 %
UNbalanced dataset (test set)
(confusion matrices images)



15- Medicane Tracking training

Tracking dataset:
collecting only video tiles with cyclones 
from detection dataset

Train set: 12 cyclones, 834 videos
Test set: 3 cyclones, 160 videos
Validation set: 3 cyclones, 192 videos



16- tracking results 
(image plot histogram error in pixel and in km)


17- Cyclone first pass