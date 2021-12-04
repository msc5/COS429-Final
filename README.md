# COS429-Final

By Alkin Kaz, Sam Liang, Matthew Coleman

### Helpful Links

- [Resnet 1st Implementation](https://towardsdatascience.com/residual-network-implementing-resnet-a7da63c7b278)
- [Resnet 2nd Implementation](https://github.com/kuangliu/pytorch-cifar/blob/master/models/resnet.py)
- [Implementation of Matching Nets and Prototypical Networks](https://github.com/oscarknagg/few-shot/tree/672de83a853cc2d5e9fe304dc100b4a735c10c15)
- [Explanation of Matching Nets, Prototypical Networks, and MAML](https://towardsdatascience.com/advances-in-few-shot-learning-a-guided-tour-36bc10a68b77)
- [Timeline for Meta-Learning](https://neptune.ai/blog/understanding-few-shot-learning-in-computer-vision)
- Look into concatenating/comparing feature maps at different levels in the CNN
Final Project for COS 429 Computer Vision
- As most few-shot learning models utilise four convolutional blocks for embedding module [39, 36], we follow the
same architecture setting for fair comparison, see Figure 2.
More concretely, each convolutional block contains a 64-
filter 3 × 3 convolution, a batch normalisation and a ReLU
nonlinearity layer respectively. The first two blocks also
contain a 2 × 2 max-pooling layer while the latter two do
not. We do so because we need the output feature maps
for further convolutional layers in the relation module. The
relation module consists of two convolutional blocks and
two fully-connected layers. Each of convolutional block
is a 3 × 3 convolution with 64 filters followed by batch
normalisation, ReLU non-linearity and 2 × 2 max-pooling.
The output size of last max pooling layer is H = 64 and
H = 64 ∗ 3 ∗ 3 = 576 for Omniglot and miniImageNet
respectively. The two fully-connected layers are 8 and 1
dimensional, respectively. All fully-connected layers are
ReLU except the output layer is Sigmoid in order to generate relation scores in a reasonable range for all versions
of our network architecture.
The zero-shot learning architecture is shown in Figure 3.
In this architecture, the DNN subnet is an existing network
(e.g., Inception or ResNet) pretrained on ImageNet.
- With multiple feature maps at different levels being compared, is it even necessary to use a pre-trained base classifier? i.e. Pure meta-learning?
- Pre-training base classifier can lead to bias in object scope
- [Matching Networks DeepMind crew](https://github.com/karpathy/paper-notes/blob/master/matching_networks.md)
