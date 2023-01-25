# 9 Slots Are All You Need For Unpaired Image-to-Image

# Translation

## Berke Can Rizai — Eren Barıs ̧ Bostancı

## January 25, 2023

## 1 Introduction

Our aim in this project is to implement the SLOT attention mechanism from the SLOT attention paper[1],
into the AttentionGAN[2] and look for possible improvements.
Our contributionin this project is that, we explored the potential benefits of replacing the ”attention mask-
ing” part of the AttentionGAN[2] architecture with the SLOT attention[1] method developed by Google Re-
search. Our goal was to determine whether this substitution would improve the results achieved by the Atten-
tionGAN model.
To begin, we have done literature review and examined the AttentionGAN[2] and Slot Attention[1] papers.
For the implementation, we used PyTorch as our platform and converted the TensorFlow implementation of
SLOT attention[3] to PyTorch. However, this approach didn’t succeed, and we used another implementation
from the ”untitled-ai”[4]. Then we tested the modified AttentionGAN model using the CycleGAN dataset.
This dataset included a variety of different image translation tasks, including ”apple2orange”, ”horse2zebra”,
”maps”, ”cezanne2photo”, and ”monet2photo”. We started with summer to winter, which was used in the
original AttentionGAN paper, and continued with horse to zebra and apple to orange. Also compared the results
with the original results. All implementations can be found in https://github.com/ribo-apps/COMP411-Project.

## 2 Motivation and Goal

The motivation for this project is to investigate the potential benefits of replacing the ”attention masking” part
of the AttentionGAN architecture with the SLOT attention method developed by Google Research.
Our goal is to determine whether this substitution would improve the results achieved by the Attention-
GAN model on the CycleGAN data set and look for possible improvements. We will also compare the results
with the original results to see if there is any improvement.

## 3 AttentionGAN

The AttentionGAN model, like in the CycleGAN, contains two GANs, which each of them has two main
components, a generator and a discriminator. The generator generates new images, while the discriminator tries
to recognize if the image is real or generated. During training, the generator learns to produce images that
are similar to the real images, while the discriminator learns to correctly classify real and generated images.
The main difference of the AttentionGAN and CycleGAN is, AttentionGAN has mask generator to separate
foreground objects with the background. This let the AttentionGAN to perform better performance while trans-
formation process. In the original implementation[2], AttentionGAN model uses spatial-attention masks. The
attention mechanism is implemented into the generator part of the GANs as can be seen at figure 1. We wanted
to see if can we improve the model by changing the attention part with the slot attention.


```
Figure 1: AttentionGAN Structure.
```
## 4 SLOT Attention

This attention module was proposed in the paper titled as ”Object-Centric Learning with Slot Attention” from
researchers in ETH Zurich and Google Brain. Idea is to have vectors called ”SLOTs” and have each object that
are present in the image to bind to them. By making each slot attend to different objects, we can use these slots
for different purposes. This is obtained by feeding the slots with high level representations from some network
such as CNN, rather than learning to attend to particular class of object, these slots can bind to different type
of objects that is enabled by the encoded representation. Also, these slots compete for binding the objects by
passing them through slot-wise softmax.

```
Figure 2: SLOT Attention slot.
```
This module does not have its own special loss, it has the loss of recurrent unit and MLP and these take the
upstream gradient and pass to back after these, so the loss is task dependent. The authors of the original paper
explored a variety of possible use cases from both supervised and unsupervised domains.
In the attention module, queries are slots, keys and values are inputs that is the encoded feature. If we look
at the algorithm, first, slots are randomly initialized from noise. Then, for number of iterations (such as 3 in our
model), they are updating themselves recurrently. First, they are passed through layer normalization, they are
multiplied with inputs then taken softmax. After that step, we take product with the inputs again and these slots
are fed into Gated recurrent unit where state is last iterations output slots and new input is the new slots. After
that, they are passed through some MLP. Hidden layers in MLP are defined by user, we had (64, 64, 64, 256)
hidden dimensions with three iterations in this SLOT module. As you may realize, last hidden layer produces
256 dimensions which is same as our slot dimension size.


```
Figure 3: Slot Attention Architecture.
```
```
Figure 4: SLOT Attention Algorithm
```
## 5 Implementation Details & Experiments

### 5.1 Literature Reading and Reviewing

We scraped the internet for possible implementations we could use or get inspired from. We have listed these
resources at the bottom. Even though one particular implementation was quite easy to understand, we decided
not to use it and will explain why down below.

### 5.2 Implementing SLOT From Scratch

We first started by directly converting from TensorFlow to PyTorch since it seemed quite an easy and trivial
implementation, however we could not get the outputs to be the same between the official implementation from
Google and our implementation in PyTorch. We put this idea on the shelf.

### 5.3 Unofficial AttentionGAN Implementation with Slot Attention

Firstly, we wanted to start with a simple example to see what we should expect as an output from models.
Since the main paper’s implementation is complex, we found an open-source unofficial implementation for the
AttentionGAN[5]. We changed the spatial-attention with slot attention. To compare the results, we trained
both the original spatial attention version and the changed slot attention version. Both versions trained for 200
epochs. In slot attention version we used 5 slots and in the spatial attention it has 5 attention masks. We first
used summer2winter data. We also trained for horse2zebra and apple2orange. We trained both version with
128x128 resolution.


Slot Attention vs Spatial Attention

```
Figure 5: Summer2Winter Slot Attention vs Spatial Attention.
```

Slot Attention vs Spatial Attention

```
Figure 6: Horse2Zebra Slot Attention vs Spatial Attention.
```

Slot Attention vs Spatial Attention

```
Figure 7: Apple2Orange Slot Attention vs Spatial Attention.
```

For summer2winter data set, we can say that slot attention produce more vivid color tones and better
preserved the lines of objects such as trees and mountains as we can see in the figure 5. However, there are more
noise in the images due to slot attention’s mask.
For horse2zebra data set, slot attention done better job to preserve the features such as face structure,
colored markings, eyes, manes while changing horses to zebras as we can see in the figure 6. However, like in
the summer2winter results, some samples have more noise even the overall image is better.
For apple2orange data set, we expected better results since this data set is more similar to the slot atten-
tion’s original data set CLEVR. However, we got worst results in this data set. For both version, results are
noisy, and transformation process is not good enough. The AttentionGAN model mostly tries to change the
skin and color of the objects during transformation process instead of shapes, which might be the reason for
worse results. As we can see in the figure 7, even both version has bad results, the spatial attention version can
be considered better due to less noise.

### 5.4 Yet Another Unofficial Slot Attention Implementation

We have also found another implementation from Wang’s implementation[6] which is a package that can
be installed via pip in python that makes it easy to use. However, when we have found this, we had already
integrated the other package in our model, because of this, we did not use it.

### 5.5 Main Implementation with Slot Attention from untitled-ai and AttentionGAN

Our most promising attempt was done with the implementation of SLOT attention from the ”untitled-ai”
where we incorporated the given model in [4] into AttentionGAN[7], specifically into the attention mechanism
of the model. In this case, the novelty in our approach was to directly feed the image (256x256x3) into SLOT
attentions instead of the encoded space, even though the authors did not intend to use these as such. The rest
of the process was to take the SLOTS (vectors) from the attention mechanism and apply positional embedding
to them, pass them through decoder layer and transpose convolution, get attention masks that are as the same
size of the original input, and we multiply these masks with the 27 dimensional encoded features from the CNN
which are also same resolution (256x256) of the original image. We have 9 slots, each and thus 9 attention
masks, each slot is multiplied by 3 layers of the output of CNN. In this case, each SLOT was 256 dimensional,
we had three recurrent iterations, and the decoder resolution was 16x16.

```
Figure 8: Samples of the last model
```

```
Figure 9: Samples of the last model
```
```
Figure 10: Samples of the last model
```
This model would take about 12 days to complete 200 epochs on Tesla V100 GPU, we had to limit our time
by reducing training size to 1/8th which is about 150 images for the Yosemite Winter2Summer dataset. In our
local machines, we had issues with RAM, and also the not even a single epoch was completed in 8 hours. See:
figure 11
At the end of training, our cycle loss for epochs looks as follows. We can see the gradual decrease in loss,
although we have some random variations time to time.
When we plot all the losses combined i.e. (sum of GAN losses, Cycle losses for both generator, discrim-
inator), we can compare the original AttentionGAN to our version.


```
Figure 11: Train time.
```
```
Figure 12: Cycle Losses
```
Paper’s loss graph:

```
Figure 13: Original AttentionGAN loss.
```

```
Our loss:
```
```
Figure 14: Our AttentionGAN loss.
```
These graphs are similar but one difference is the Lid where our loss did not quite match the one in
original paper.
One particular issue in our last iteration of model was that, it ocassionally added some weird artifacts to
output photos that were looking like some random attentions (ie attending to rocks in picture).

```
Figure 15: Artifacts
```

```
Figure 16: Attention examples
```
```
Figure 17: Attention examples
```
```
Figure 18: Attention examples
```
Here are some examples of artifacts we have discussed earlier. We can see the attentions of the model on
generator output.


```
Figure 19: Bad samples
```
```
Figure 20: Bad samples
```
## 6 Performance Gap Between our SLOT Attention and Original

One thing to note is that in the original implementation, SLOT attention was used, trained and tested in the
CLEVR dataset. This dataset mostly consist of objects of various colors, shapes on a plain background. That is
an issue and also a reason for performance gap because our data (and also the real world data) is not that simple.
In our data, we don’t have that clearly defined objects and cleanly separable background from the foreground.
9 slots could be to blame as well.

```
Figure 21: Example from CLEVR dataset
```
## 7 Possible Improvements

One major improvement could be increasing the iteration size of the SLOT attention module, trying different
hyperparameters in this module such as hidden dimensions, size of each fully connected hidden dimension,


vector dimensions of slots. We feel that too many SLOT dimensions made the model unresponsive to objects
because it doesn’t try to do dimension reduction in a sense.
Another improvement could be to have another small model that takes the SLOTs and determines which
ones are binding to objects in the foreground and which ones are background explicitly. We could also test these
in another dataset where separation is more obvious.
For the AttentionGAN, we could try deeper architecture before the SLOT module since the SLOT atten-
tion takes encoded features to bind the objects, a deeper CNN could yield better results.
We theorize that we could eliminate these grid like patterns that are coming from the SLOT attention
mechanism either by increasing the epoch of training or changing our approach such as decreasing SLOT
numbers and dimensions of each SLOTs.

## References

[1] F. Locatello, D. Weissenborn, T. Unterthiner, A. Mahendran, G. Heigold, J. Uszkoreit, A. Dosovitskiy, and
T. Kipf, “Object-centric learning with slot attention,” 2020.

[2] H. Tang, H. Liu, D. Xu, P. H. Torr, and N. Sebe, “Attentiongan: Unpaired image-to-image translation using
attention-guided generative adversarial networks,”IEEE Transactions on Neural Networks and Learning
Systems (TNNLS), 2019.

[3] F. Locatello, D. Weissenborn, T. Unterthiner, A. Mahendran, G. Heigold, J. Uszkoreit, A. Dosovitskiy, and
T. Kipf, “Slot attention,” https://github.com/google-research/google-research/tree/master/slotattention,
2020.

[4] P. L. K. Bryden Fogelman, “Slot attention unoffical implementation,” https://github.com/untitled-ai/slot
attention, 2021.

[5] delta6189, “Attentiongan unoffical implementation,” https://github.com/delta6189/AttentionGAN, 2020.

[6] R. B. P. Phil Wang, “Slot attention unoffical implementation,” https://github.com/lucidrains/slot-attention,
2020.

[7] H. Tang, H. Liu, D. Xu, P. H. Torr, and N. Sebe, “Attentiongan,” https://github.com/Ha0Tang/
AttentionGAN, 2019.


