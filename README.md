# Self-Attention Music Generative Adversarial Network (SAMGAN)

## Overview
Recent success in transformer based language models such as BERT and GPT-2 has inspired me to develop the SAMGAN. The generator and discriminator consist of decoder self-attention blocks with no masking.

## Data
All audio data was downloaded for free from the [Free Music Archive](https://github.com/mdeff/fma/). The mel-scaled spectogram is then computed and fed as input into the GAN model. The model trains only on music files classified in the electronic genre.

## Train the Model
A list named `filenames` is required to be defined that contains the full or relative paths to the audio files to be trained on. Checkout the TF app command line flags available in [fma_gen.py](fma_gen.py). To train the GAN and create Tensorboard logs for example:

```bash
python fma_gen.py --train True --logging True --epochs 200 --seed 198
```

## Results
Initial experiments have proven unsuccessful. Sample quality seems to improve with the hinge loss objective and a [two time-scale update rule (TTUR)](https://arxiv.org/abs/1706.08500) but are still incoherent. Other GAN architectures and better informed dataset features should be explored.
