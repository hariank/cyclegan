# PyTorch CycleGAN

Minimal implementation of [Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks](https://arxiv.org/pdf/1703.10593.pdf). 

Results on Google Maps dataset (Real/Generated):

<p float="left">
  <img src="results/B_real_1.png" width="185" />
  <img src="results/A_fake_1.png" width="185" />
</p>
<p float="left">
  <img src="results/B_real_2.png" width="185" />
  <img src="results/A_fake_2.png" width="185" />
</p>
<p float="left">
  <img src="results/A_real_1.png" width="185" />
  <img src="results/B_fake_1.png" width="185" />
</p>
<p float="left">
  <img src="results/A_real_2.png" width="185" />
  <img src="results/B_fake_2.png" width="185" />
</p>

These transformations are learned *unpaired* - the satellite and Google maps images are presented independently to the model.

## Data Format

    dataset_dir
    |__trainA
    |__trainB
    |__testA
    |__testB

## Running
Required:

 - Python 3+
 - PyTorch 1.0.1+
 - tensorboardX 

`python train.py <run_name> --data dataset_dir --identity-loss --gen-buffer`

Generated images are displayed in Tensorboard (real/generated/real/generated):
<p float="left">
  <img src="results/tb.png" width="400" />
</p>
