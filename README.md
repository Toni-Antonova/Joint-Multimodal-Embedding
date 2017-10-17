# Joint-Multimodal-Embedding
Keras implementation of a Siamese Neural Network for Joint Multimodal Text-Image Embedding

*Tested with a Tensorflow backend*

This embedding network was used in a larger deep learning project aimed at using GANs to generate video from text descriptions. Check out my [blogpost](https://antonia.space/text-to-video-generation/) on the results.

#### Tools Used:
Python 3  
Keras  
[Keras' Xception V1 model pre-trained on ImageNet](https://keras.io/applications/#xception)  

Text data vectorized prior to running this network using a Variational Autoencoder and Facebook's fastText pre-trained Word2Vec. More information [found here](https://github.com/Toni-Antonova/VAE-Text-Generation).
