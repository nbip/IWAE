# IWAE

Reproducing results from the [original IWAE paper](https://arxiv.org/pdf/1509.00519.pdf) in tensorflow2. 

### Usage

``` 
python main.py --n_latent   <# of latent space dimensions, 50 by default>  
               --n_samples  <# of importance samples, 5 by default>  
               --n_hidden   <# of hidden units in the nueral network layers, 200 by default>  
               --batch_size <20 by default>
               --epochs     <# of epochs, if set to -1 the number of epochs will be based on the learning rate scheme from the paper>
```
The original paper uses a train-test split and does not monitor a validation loss during training. Just out of curiosity we will set aside a validation set from the training set, to monitor performance during training. This can also be used to do early stopping by loading the weights from the iteration with the highest validation loss. On the other hand there will be less training data available.  
Use tensorboard as
``` 
tensorboard --logdir=/tmp/iwae 
```

### Results
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

### Tasks
`main.py`: original experiment with 1 stochastic layer.  
`task01.py`: 1 stochastic layer, but with validation set and early stopping.  We use the validation set to monitor performance via tensorboard, and to store the optimal set of weights.  
`task02.py`: 1 stochastic layer and static binarization.  
`task03.py`: 1 stochastic layer, static binarization and validation set for tensorboard and early stopping.  
`task04.py`: Monitor how reconstructions and sampels from the prior changes during training.  
`task05.py`: Use the Double Reparameterized Gradient Estimator, [DReG](https://arxiv.org/pdf/1810.04152.pdf), to the original experiment.  
`task06.py`: Two stochastic layers.  
`task07.py`: Use a 2D latent space to investigate both true and variational posteriors. We can use *self-normalized importance sampling* to estimate posterior means and *sampling importance resampling* to draw samples from the true posterior.  

### Comparisons
A number of other repositories have reproduced these results, see for example  
- The original code acompanying the paper [yburda](https://github.com/yburda/iwae)  
- [abdulfatir](https://github.com/abdulfatir/IWAE-tensorflow)  
- [shwanmario](https://github.com/ShwanMario/IWAE)  
- [xqding](https://github.com/xqding/Importance_Weighted_Autoencoders)
- [yoonholee](https://github.com/yoonholee/pytorch-vae)

### TODO:
Extend to two stochastic layers  
Implement [DReG](https://arxiv.org/abs/1810.04152)  
Show true and variational posteriors
Show active units
Implement quantized distribution  
Implement MIWAE  
Show
- SNIS
- SIR