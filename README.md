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
``` tensorboard --logdir=/tmp/iwae ```

### Results
Test-set log likelihoods are estimated using 5000 importance samples
#### Dynamic binarization
| Method | Test-set LLH (this repo) | Test-set LLH ([original paper](https://arxiv.org/pdf/1509.005\
19.pdf)) |
| --- | --- | --- |
| 1 | | |
| 5 | | |
| 50 | | |

#### Static binarization
Should be compared to the results in appendix D.  
| Method | Test-set LLH (this repo) | Test-set LLH ([original paper](https://arxiv.org/pdf/1509.00519.pdf)) |
| --- | --- | --- |
| 1 | | |
| 5 | -87.65 | -87.63 |
| 50 | | |  


### Comparisons
A number of other repositories have reproduced these results, see for example  
- The original code acompanying the paper [yburda](https://github.com/yburda/iwae)  
- [abdulfatir](https://github.com/abdulfatir/IWAE-tensorflow)  
- [shwanmario](https://github.com/ShwanMario/IWAE)  
- [xqding](https://github.com/xqding/Importance_Weighted_Autoencoders)
- [yoonholee](https://github.com/yoonholee/pytorch-vae)

### TODO:
Extend to dynamically binarized mnist  
Implement [DReG](https://arxiv.org/abs/1810.04152)  
Implement quantized distribution  
Implement MIWAE  
Show
- SNIS
- SIR
- 2D posterior