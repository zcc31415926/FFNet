# FFNet: Monocular Pedestrian Orientation Estimation Based on Deep 2D-3D Feedforward
Chenchen Zhao, Yeqiang Qian, and Ming Yang

<br />

The PyTorch implementation of our paper: *[Monocular Pedestrian Orientation Estimation Based on Deep 2D-3D Feedforward](https://arxiv.org/abs/1909.10970)*

We propose a test-time monocular 2D pedestrian orientation estimation model. The model receives the image features and the 2D & 3D (train-time) dimension information as inputs, and outputs the estimated orientation of each pedestrian object

<br />

*3.5 years later...*
- Rewrite the code
- Add high-freq embedding strategies to the input 2D & 3D dimensions. The strategies are similar to the timestep embedding strategy in diffusion models

<br />

Code inspired by [Deep3DBox](https://github.com/smallcorgi/3D-Deepbox)
