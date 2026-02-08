In the following a few key elements of the VideoMAE transformer-based model are provided.

- How Transformers Work and Why They Outperform CNNs

Transformers rely on the self-attention mechanism, which computes dependencies between
different parts of an image or video sequence. Unlike CNNs, which are constrained to local
feature extraction, transformers can:
1. Model Long-Range Dependencies: Each patch in an image or frame in a video interacts
directly with all other patches.
2. Learn Contextual Relationships: Instead of static filters, transformers dynamically adjust
feature weights based on the entire input.
3. Scale Effectively: Larger datasets and models (e.g., ViT-Large, ViT-Huge) lead to improved
performance without excessive overfitting.
This architecture has been particularly effective for video understanding, where temporal
relationships between frames are crucial.

A review that highlights this topic is well reported by:
Comparing Vision Transformers and Convolutional Neural Networks for Image
Classification: A Literature Review, J. Maurício, I. Domingues, J. Bernardino -
https://www.mdpi.com/2225-1154/12/12/220

- VideoMAE: A Transformer-Based Model for Video Understanding

Video Masked Autoencoders (VideoMAE) are a self-supervised learning approach designed for
efficient video representation learning. They follow the success of Masked Autoencoders (MAE)
for images, but introduce a novel tube masking strategy to handle video data.
Key Features of VideoMAE
1. High Masking Ratio (90-95%): Unlike image-based MAEs (which mask ~75% of tokens),
VideoMAE can mask up to 95% of the input video tokens due to the redundancy in video
frames.
2. Tube Masking: Instead of randomly dropping individual pixels or patches, VideoMAE
masks entire spatiotemporal cubes (tubes) across frames, making the reconstruction task
more challenging.
3. Asymmetric Encoder-Decoder Architecture:
• The encoder processes only the visible (unmasked) tokens, making training efficient.
• The decoder reconstructs missing patches using a lightweight architecture, reducing
computational costs.

- Why VideoMAE is Effective for Short Video Classification


• Efficient Representation Learning: By training on a self-supervised pretext task
(reconstructing missing video frames), the model learns rich spatiotemporal features without
requiring labeled data.
• Scalability: VideoMAE can be scaled to large datasets (e.g., Kinetics-400, Something-
Something V2) and larger architectures (ViT-L, ViT-H, ViT-g) to achieve state-of-the-art
performance.
• Transferability: Pre-trained VideoMAE models perform well on various downstream tasks,
including video classification, action recognition, and spatiotemporal localization.

- VideoMAE Architecture Overview

The Video Masked Autoencoder (VideoMAE) is a self-supervised learning framework designed to
efficiently pre-train vision transformers for video understanding. Its tube masking strategy, joint
space-time attention, and asymmetric encoder-decoder architecture make it one of the most
effective models for learning spatiotemporal representations. Below is an in-depth look at its
architecture based on the papers you provided.
- 1. Key Components of VideoMAE
VideoMAE consists of three major components:
1. Cube Embedding – Converts input frames into a sequence of tokens.
2. Encoder (Masked ViT) – Processes only visible tokens to extract high-level features.
3. Decoder (Masked Reconstruction) – Reconstructs missing patches using a lightweight
architecture.
- 2. Cube Embedding (Patch Tokenization)
Before processing, VideoMAE divides each video into small spatiotemporal cubes (patches).
These cubes are converted into tokens and serve as input for the transformer.
Patch Size: Each cube is 2 × 16 × 16 pixels (2 frames, 16×16 resolution).
Feature Representation: Each patch is embedded into a vector and assigned a position
encoding.
Reduction of Spatial and Temporal Redundancy: Helps eliminate unnecessary information
from static parts of videos
- 3. Encoder: Masked Spatiotemporal Transformer
The encoder follows a Vision Transformer (ViT) backbone with a joint space-time attention
mechanism, which allows it to model complex video dynamics.
Joint Space-Time Attention: Unlike CNNs, which process spatial features independently,
VideoMAE attends over both spatial and temporal dimensions simultaneously.
Masked Token Strategy: Only 10% of input tokens are processed in the encoder, while the
rest are dropped, significantly improving efficiency
- 4. Tube Masking Strategy
A key innovation in VideoMAE is the Tube Masking Strategy, which significantly differs from
conventional patch masking in images.
Why Tube Masking?
In images, missing patches can be inferred from surrounding areas.
In videos, due to temporal redundancy, masking individual patches is too easy for
reconstruction models.
How It Works? 
Instead of randomly masking individual pixels or patches, entire spatiotemporal tubes
(continuous regions in time and space) are masked.
This makes the task harder and forces the model to learn robust video representations
- 5. Decoder: Lightweight Reconstruction
The decoder attempts to reconstruct the missing patches given only a fraction of the original
input.
•
•
Small and Efficient: Unlike the encoder, which operates on only 10% of the patches, the
decoder operates on 100% of the tokens but is significantly smaller in depth and width.
Asymmetric Architecture:
    Encoder: 12 blocks (ViT-Base)
    Decoder: 4 blocks (lighter, but sufficient for reconstruction)
    This asymmetry drastically reduces computation time

- 6. Dual Masking in VideoMAE V2
The second iteration of VideoMAE (VideoMAE V2) introduces a dual masking approach, further
improving efficiency:
    Encoder Masking: 90% of tokens are dropped before entering the encoder.
    Decoder Masking: Instead of reconstructing the entire frame, only a subset of missing cubes is reconstructed.
    Benefit: This further reduces memory usage and speeds up training, allowing VideoMAE to scale to billion-parameter video models

- 7. Computational Efficiency
One of the most significant advantages of VideoMAE is its ability to train efficiently on large-scale
video datasets.
- Asymmetric Architecture: The small decoder reduces computational overhead while still
providing high-quality reconstructions.
- Efficient Self-Supervised Pre-Training: No labeled data is required—models learn
representations by reconstructing missing patches.
- Scalability: VideoMAE can scale from small models (ViT-B) to billion-parameter models
(ViT-g)

