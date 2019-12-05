# Self-Attention Music Generative Adversarial Network (SAMGAN)

## Overview
Recent success in transformer based language models such as BERT and GPT-2 has inspired me to develop the SAMGAN. The generator and discriminator consist of decoder self-attention blocks with no masking.

## Data
All audio data was downloaded for free from the [Free Music Archive](https://github.com/mdeff/fma/). The mel-scaled spectogram is then computed and fed as input into the GAN model. The model trains only on music files classified in the electronic genre.