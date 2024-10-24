# Video
## General idea and thought process
Video is done using a spatiotermporal CNN from "The Conversation: 
Deep Audio-Visual Speech Enhancement".
This paper details that 3D convolutional layer is used which is then
pumped into a ResNet. We do not think that the ResNet part is used in the
implementation of our paper. According to the paper "Combining Residual
Networks with LSTMs for Lipreading" the convolution layer was implemented
as such:
## Resources
### Paper: The Conversation: Deep Audio-Visual Speech Enhancement
Visual features are extracted from the input image frame se-
quence with a spatio-temporal residual network similar to the
one proposed by [33], pre-trained on a word-level lip reading
task. The network consists of a 3D convolution layer, followed
by a 18-layer ResNet [36]. For every video frame the network
outputs a compact 512 dimensional feature vector f v
0 (where
the subscript 0 refers to the layer number in the audio-visual net-
work). Since we train and evaluate on datasets with pre-cropped
faces, we do not perform any extra pre-processing, besides con-
version to grayscale and an appropriate scaling.
### Paper: Combining Residual Networks with LSTMs for Lipreading

"The first set of layers applies spatiotemporal convolution to the
preprocessed frame stream. Spatiotemporal convolutional lay-
ers are capable of capturing the short-term dynamics of the
mouth region and are proven to be advantageous, even when
recurrent networks are deployed for back-end, [1]. They con-
sist of a convolutional layer with 64 3-dimensional (3D) kernels
of 5×7×7 size (time/width/height), followed by Batch Normal-
ization (BN, [29]) and Rectified Linear Units (ReLU). The ex-
tracted feature maps are passed through a spatiotemporal max-
pooling layer, which drops the spatial size of the 3D feature
maps. The number of parameters of the spatiotemporal front-
end is ∼16K."

## Implementation
### 3D/2D CNN
1. convolutional layer 64 3 dimensional kernals of 5x7x7 size (time/width/height)
2. Batch normalization
3. Rectigied Linear Units ReLU
4. spatiotemporal max- pooling layer

### PE (Position Encoding)
PE{a,v,q} ∈ Rt{a,v,q}×c

implemented as sinusoidal vectors
### ME (Modality Encoding)
ME{a,v,q} ∈ Rc

learnable vectors
### Combination
V = V + P Ev + M Ev 
