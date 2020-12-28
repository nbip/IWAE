# IWAE

Reproducing results from the [original IWAE paper](https://arxiv.org/pdf/1509.00519.pdf) in TensorFlow 2. 

## Usage
The results for the model with 1 stochastic layer and 1, 5 or 50 importance samples can be obtained by running `main.py` with the default settings, adjusting the number of samples.
``` 
python main.py --n_latent   <# of latent space dimensions, 50 by default>  
               --n_samples  <# of importance samples, 5 by default>  
               --n_hidden   <# of hidden units in the nueral network layers, 200 by default>  
               --batch_size <20 by default>
               --epochs     <# of epochs, if set to -1 the number of epochs will be based on the learning rate scheme from the paper>
```
The model is investigated further in a series of tasks found in `./tasks`

## Results
Test-set log likelihoods are estimated using 5000 importance samples

#### Dynamic binarization
| Method | Test-set LLH (this repo) | Test-set LLH ([original paper](https://arxiv.org/pdf/1509.00519.pdf)) |
| --- | --- | --- |
| 1 | -87.75 | -86.76 |
| 5 | -85.50 | -85.54 |
| 50 | -84.65 | -84.78 |

#### Static binarization
Should be compared to the results in appendix D.  
| Method | Test-set LLH (this repo) | Test-set LLH ([original paper](https://arxiv.org/pdf/1509.00519.pdf)) |
| --- | --- | --- |
| 1 | -89.93 | -88.71 |
| 5 | -87.65 | -87.63 |
| 50 | -86.71 | -87.10 |  

#### DReG dynamic binarization
The Doubly Reparameterized Gradient Estimator for Monte Carlo Objectives, [DReG](https://arxiv.org/pdf/1810.04152.pdf) provides even lower variance gradients for the inference network, than just using the reparameterization trick. 

| Method | Test-set LLH (this repo) | Test-set LLH ([original paper](https://arxiv.org/pdf/1509.00519.pdf)) |
| --- | --- | --- |
| 1 |  |  |
| 5 |  |  |
| 50 |  |  |

## Tasks
`main.py`: original experiment with 1 stochastic layer.  
`task01.py`: 1 stochastic layer, but with validation set and early stopping.  We use the validation set to monitor performance via tensorboard, and to store the optimal set of weights.  
`task02.py`: 1 stochastic layer and static binarization.  
`task03.py`: 1 stochastic layer, static binarization and validation set for tensorboard and early stopping.  
`task04.py`: Monitor how reconstructions and sampels from the prior changes during training.  
`task05.py`: Use the Double Reparameterized Gradient Estimator, [DReG](https://arxiv.org/pdf/1810.04152.pdf), to the original experiment.  
`task06.py`: Two stochastic layers.  
`task07.py`: Use a 2D latent space to investigate both true and variational posteriors. We can use *self-normalized importance sampling* to estimate posterior means and *sampling importance resampling* to draw samples from the true posterior.  

### Two stochastic layers
See [Ladder VAE](https://arxiv.org/pdf/1602.02282.pdf) and accompanying [github](https://github.com/casperkaae/LVAE) and [xqding](https://github.com/xqding/Importance_Weighted_Autoencoders/blob/master/model/vae_models.py).  

[xqding](https://github.com/xqding/Importance_Weighted_Autoencoders) and []ShwanMario](https://github.com/ShwanMario/IWAE) train IWAEs with two stochastic layers, bot without reaching Yburdas results. But with better results than me. They use a different training setup then me, in order to check my implementation I will try to reproduce their results instead. Specifically I will start with xqding, batch-size 1000 and 5000 epochs.

## Comparisons
A number of other repositories have reproduced these results, see for example  
- The original code acompanying the paper [yburda](https://github.com/yburda/iwae)  
- [abdulfatir](https://github.com/abdulfatir/IWAE-tensorflow)  
- [shwanmario](https://github.com/ShwanMario/IWAE)  
- [xqding](https://github.com/xqding/Importance_Weighted_Autoencoders)
- [yoonholee](https://github.com/yoonholee/pytorch-vae)

## Settings:  

I get significantly different results when using the ELBO as formulated in eq 8 (with 1-monte-carlo sample) compared to the formulation in eq 14. 15dec2020 I am running a head-to-head comparison on cronus in task07 and task08. Here the correct $\epsilon$ is also used.

It also looks like there is an issue with the IWAE loss compared to the analytical VAE loss in the 2 layer model. Check if this is also an issue in the 1 layer model. Specifically check  
IWAE vs analytical VAE
IWAE eq 8 vs eq 14

In the paper, the initalizer from [Glorot & Bengeio (2010)](http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf?source=post_page---------------------------) is used to initialize hidden layers. This is also the keras default initializer, which has also been used here.  
The paper uses Adam optimizer [Kingma & Ba](https://arxiv.org/abs/1412.6980), with $\beta_1 = 0.9$, $\beta_2=0.999$ and $\epsilon = 10^{-4}$. This is a bit different from the default Adam settings in keras, which has $\epsilon=10^{-7}$. This turned out not to make any difference.  

## Issues:

Is using eq 8 and eq 14 for training equivalent? It seems so I trained task07 using eq 8 and task 11 using eq 14, to get the same marginal llh results. However, you cannot use eq 8 to evaluate the marginal LLH. They are only equivalent in terms of gradients. Well, when $k=1$, they are also equivalent in terms of marginal LLH estimate.  

Note that pytorch dataloaders makes it easier to work with dynamically binarized mnist, see [xqding](https://github.com/xqding/Importance_Weighted_Autoencoders/blob/master/model/vae_models.py)

## Resources:
https://github.com/addtt/ladder-vae-pytorch  
https://github.com/xqding/Importance_Weighted_Autoencoders  
https://github.com/yburda/iwae  
https://github.com/ShwanMario/IWAE  
https://github.com/AntixK/PyTorch-VAE  
https://paperswithcode.com/paper/importance-weighted-autoencoders  
https://github.com/casperkaae/LVAE/blob/master/run_models.py  
https://github.com/yoonholee/pytorch-vae  
https://github.com/abdulfatir/IWAE-tensorflow  
https://github.com/xqding/AIWAE  
https://arxiv.org/pdf/1602.02282.pdf  
https://arxiv.org/pdf/1509.00519.pdf  
https://arxiv.org/pdf/1902.02102.pdf  
https://github.com/larsmaaloee/BIVA
https://github.com/vlievin/biva-pytorch    
https://github.com/casperkaae/parmesan  
https://github.com/casperkaae/LVAE/blob/master/run_models.py  
https://arxiv.org/pdf/1802.04537.pdf

## TODO:
Extend to two stochastic layers  
Implement [DReG](https://arxiv.org/abs/1810.04152)  
Show true and variational posteriors
Show active units
Implement quantized distribution  
Implement MIWAE  
Show
- SNIS
- SIR
2 layer:
- posterior SNIS
- PCA on the SNIS