# Video Semantic Segmentation
The complex temporal relationships between frames in a video sequence are crucial for effective semantic segmentation. However, some methods have relied on processing individual frames, resulting in a lack of consideration for temporal image information, which limits segmentation performance. While state-of-the-art approaches use deep networks to address this issue, they can be challenging to train and time-consuming for large datasets such as Cityscapes used in our experiments. To overcome these limitations, we use U-nets, an encoder-decoder style model architecture containing convolution units and convLSTM bridge. We compare this approach to a ResUnet with pre-trained ResNet-18 and ResNet-34 encoders that utilizes recurrent units and preserves the original U-net's structure. Results show that the different variations of ResUnet-34 outperforms the other U-net based architecture after training. All results are further evaluated through quantitative metrics namely mIoU and mAcc, and qualitatively using GIFs creation. The  base ResUnet model achieved the lowest mIoU (24\%) during model training, while ResUnet-34 resulted in overall better mIoUs, achieving upto 32\% mIoU.

**U-net architecture downsampling from 512x512 input dimension to 32x32 pixels followed by ConvLSTM cells as bridge, followed by upsampling to original dimensions.**
![Database](image.png)


