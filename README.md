# IWAE

Reproducing results from the [IWAE paper](https://arxiv.org/pdf/1509.00519.pdf) in TensorFlow 2. 

## Usage
The results for the model with 1 stochastic layer and 1, 5 or 50 importance samples can be obtained by running `main.py` with the default settings, adjusting the number of samples.
``` 
python main.py --n_samples  <# of importance samples, 5 by default>  
               --objective  <choose vae_elbo or iwae_elbo, iwae_elbo by default>
```
The model is investigated further in a series of tasks found in `./tasks`

## Results
<img src="results/iwae_50.gif" width="600" height="300" />

Test-set log likelihoods are estimated using 5000 importance samples

#### 1 stochastic layer
| Method | Test-set LLH (this repo) | Test-set LLH ([original paper](https://arxiv.org/pdf/1509.00519.pdf)) |
| --- | --- | --- |
| 1 | -86.35 | -86.76 |
| 5 | -85.28 | -85.54 |
| 50 | -84.63 | -84.78 |

#### 2 stochastic layers
| Method | Test-set LLH (this repo) | Test-set LLH ([original paper](https://arxiv.org/pdf/1509.00519.pdf)) |
| --- | --- | --- |
| 1 | -84.93 | -85.33 |
| 5 | -83.43 | -83.89 |
| 50 | -82.87 | -82.90 |

#### DReG dynamic binarization
The Doubly Reparameterized Gradient Estimator for Monte Carlo Objectives, [DReG](https://arxiv.org/pdf/1810.04152.pdf) provides even lower variance gradients for the inference network, than just using the reparameterization trick. 

| Method | Test-set LLH (this repo) | Test-set LLH ([original paper](https://arxiv.org/pdf/1509.00519.pdf)) |
| --- | --- | --- |
| 1 |  |  |
| 5 |  |  |
| 50 |  |  |

## Tasks
`main.py`: original experiment with 1 or 2 stochastic layers.  
`task01.py`: Use a 2D latent space to investigate both true and variational posteriors. We can use *self-normalized importance sampling* to estimate posterior means and *sampling importance resampling* to draw samples from the true posterior.  
`task02.py`: Use the Double Reparameterized Gradient Estimator, [DReG](https://arxiv.org/pdf/1810.04152.pdf), to the original experiment.
`task03.py`: 2 stochastic layers with 2d and 1d latent spaces

### Two stochastic layers
See 
- [xqding](https://github.com/xqding/Importance_Weighted_Autoencoders/blob/master/model/vae_models.py)  
- [ShwanMario](https://github.com/ShwanMario/IWAE)
- [addtt](https://github.com/addtt/ladder-vae-pytorch)
- [Ladder VAE](https://arxiv.org/pdf/1602.02282.pdf) and accompanying [github](https://github.com/casperkaae/LVAE)

[xqding](https://github.com/xqding/Importance_Weighted_Autoencoders) and [ShwanMario](https://github.com/ShwanMario/IWAE) train IWAEs with two stochastic layers, both without reaching Yburdas results, using a different training setup than me.  

## Comparisons
A number of other repositories have reproduced results from the IWAE paper, see for example  
- The original code acompanying the paper [yburda](https://github.com/yburda/iwae)  
- [abdulfatir](https://github.com/abdulfatir/IWAE-tensorflow)  
- [shwanmario](https://github.com/ShwanMario/IWAE)  
- [xqding](https://github.com/xqding/Importance_Weighted_Autoencoders)
- [yoonholee](https://github.com/yoonholee/pytorch-vae)

## Settings:  

In the paper the initalizer from [Glorot & Bengeio (2010)](http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf?source=post_page---------------------------) is used to initialize hidden layers. This is also the keras default initializer, which has also been used here.  
The paper uses Adam optimizer [Kingma & Ba](https://arxiv.org/abs/1412.6980), with $\beta_1 = 0.9$, $\beta_2=0.999$ and $\epsilon = 10^{-4}$. The default Adam settings in keras, which has $\epsilon=10^{-7}$.

## Issues:
1 stochastic layer VAE/IWAE: I get test-set LLH 1 nat worse than reported in the paper. If I use warmup I can get the same performance.  
2 stochastic layer VAE/IWAE: I get worse performance than reported in the paper. What if I use warmup here as well? 

Is using eq 8 and eq 14 for training equivalent? It seems so, I trained task07 using eq 8 and task 11 using eq 14, to get the same marginal llh results. However, you cannot use eq 14 to evaluate the marginal LLH. They are only equivalent in terms of gradients. Well, when $k=1$, they are also equivalent in terms of marginal LLH estimate.  

Note that pytorch dataloaders makes it easier to work with dynamically binarized mnist, see [xqding](https://github.com/xqding/Importance_Weighted_Autoencoders/blob/master/model/vae_models.py)

## TODO:
Implement [DReG](https://arxiv.org/abs/1810.04152)  
Show true and variational posteriors
Show active units
Implement quantized distribution  
Implement MIWAE  
Show
- SNIS
- SIR
Two layer model with 2 and 1 units, to investigate the marginal distribution in the first stochastic layer.  
Conditional sampling in two-layer models, a la [addtt](https://github.com/addtt/ladder-vae-pytorch).  

## Done:
Extend to two stochastic layers  
2 layer:
- posterior SNIS
- PCA on the SNIS


# Additional results:
#### 1 stochastic layer VAE
| Method | Test-set LLH (this repo) | Test-set LLH ([original paper](https://arxiv.org/pdf/1509.00519.pdf)) |
| --- | --- | --- |
| VAE 1 | -86.35 | -86.76 |
| VAE 5 | -86.10 | -86.47 |
| VAE 50 | -86.05 | -86.35 |

#### 2 stochastic layers VAE
| Method | Test-set LLH (this repo) | Test-set LLH ([original paper](https://arxiv.org/pdf/1509.00519.pdf)) |
| --- | --- | --- |
| VAE 1 | -84.93 | -85.33 |
| VAE 5 | -84.04 | -85.01 |
| VAE 50 | -83.76 | -84.78 |

## Using warm-up, as suggested in [LVAE](https://arxiv.org/pdf/1602.02282.pdf)
#### IWAE equation (8) with warm-up
| Method | Test-set LLH (this repo) | Test-set LLH ([original paper](https://arxiv.org/pdf/1509.00519.pdf)) |
| --- | --- | --- |
| 1 |  | -86.76 |
| 5 |  | -85.54 |
| 50 |  | -84.78 |


#### Standard VAE with warmup
| Method | Test-set LLH (this repo) | Test-set LLH ([original paper](https://arxiv.org/pdf/1509.00519.pdf)) |
| --- | --- | --- |
| 1 |  | -86.76 |
| 5 |  | -86.46 |
| 50 |  | -86.35 |

# Implementation learnings
I could not get the 1-sample VAE/IWAE results to match paper results, they were consistently 1 nat worse. Furthermore, when increasing the number of monte-carlo samples for the VAE, the elbo did not necessarily go down. Initializing the bias of the p(x|z) layer alleviated this issue.  
The model with two stochastic layers, did not yield similar results to the paper, again, the bias initialization mitigated this issue.

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
https://github.com/neha191091/IWAE/blob/master/iwae/experiments.py  
https://github.com/jmtomczak/vae_vampprior  
https://github.com/harvardnlp/sa-vae  

