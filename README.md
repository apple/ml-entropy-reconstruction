# The Role of Entropy and Reconstruction in Multi-View Self-Supervised Learning

This software project accompanies the ICML 2023 research paper, [The Role of Entropy and Reconstruction in Multi-View Self-Supervised Learning](https://openreview.net/forum?id=YJ3ytyemn1). It contains the code to compute the entropy and reconstruction used to train the `ER` models.

```bibtex
@InProceedings{
  rodriguez2023er,
  title={The Role of Entropy and Reconstruction for Multi-View Self-Supervised Learning},
  author={Borja Rodriguez-Galvez and Arno Blaas and Pau Rodriguez and Adam Golinski and Xavier Suau and Jason Ramapuram and Dan Busbridge and Luca Zappella},
  year={2023},
  booktitle={ICML},
}
```

## Abstract

The mechanisms behind the success of multi-view self-supervised learning (MVSSL) are not yet fully understood Contrastive MVSSL methods have been studied through the lens of InfoNCE, a lower bound of the Mutual Information (MI). However, the relation between other MVSSL methods and MI remains unclear. We consider a different lower bound on the MI consisting of an entropy and a reconstruction term (ER), and analyze the main MVSSL families through its lens. Through this ER bound, we show that clustering-based methods such as DeepCluster and SwAV maximize the MI. We also re-interpret the mechanisms of distillation-based approaches such as BYOL and DINO, showing that they explicitly maximize the reconstruction term and implicitly encourage a stable entropy, and we confirm this empirically. We show that replacing the objectives of common MVSSL methods with this ER bound achieves competitive performance, while making them stable when training with smaller batch sizes or smaller exponential moving average (EMA) coefficients. 


## Documentation
### Install dependencies
`pip -r requirements.txt`

### Getting Started
To verify that the code is working as expected, run:

`python entropy_reconstruction.py`

It should return the following:

```
Continuous entropy
Entropy sphere plug-in estimator: -249.30848693847656
Entropy sphere Joe's estimator: -249.30833435058594
Discrete entropy
Max entropy 6.907755278982137
Entropy on uniform sample: 6.907612323760986
Entropy on uniform sample (high temp): 6.695544242858887
Entropy on one_hot vector: -1.1920928244535389e-07
Continuous reconstruction
Reconstruction error: -248.65525817871094
Discrete reconstruction
Reconstruction error: 6.100684642791748
```

### Computing entropy
```python
def entropy(
    embeddings: torch.Tensor,
    kappa: float = 10,
    support: str = "sphere",
    reduction: str = "expectation",
) -> torch.Tensor:
    """Computes the entropy from a tensor of embeddings

    :param embeddings: tensor containing a batch of embeddings
    :type embeddings: torch.Tensor
    :param kappa: von Misses-Fisher Kappa (https://en.wikipedia.org/wiki/Von_Mises-Fisher_distribution), defaults to 10
    :type kappa: float, optional
    :param support: support of the random variables. Sphere or discrete, defaults to "sphere"
    :type support: str, optional
    :param reduction: "average" for Joe's estimator and "expectation" for the plug-in estimator (see Section 4.1), defaults to "expectation"
    :type reduction: str, optional
    :return: entropy value
    :rtype: torch.Tensor
    """
```

### Computing the reconstruction loss
```python
def reconstruction(
    projection1: torch.Tensor,
    projection2: torch.Tensor,
    kappa: float = 10,
    support: str = "sphere",
) -> torch.Tensor:
    """Reconstruction error from ER

    :param projection1: projection of augmentation1
    :type projection1: torch.Tensor
    :param projection2: projection of augmentation2
    :type projection2: torch.Tensor
    :param kappa: von Misses-Fisher kappa, defaults to 10
    :type kappa: float, optional
    :param support: support of the random variables, defaults to "sphere"
    :type support: str, optional
    :return: reconstruction error
    :rtype: torch.Tensor
    """
```
