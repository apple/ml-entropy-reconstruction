#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2020 Apple Inc. All Rights Reserved.
#
import math

import torch
import torch.nn.functional as F
from scipy.special import iv

eps = 1e-7


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
    k = embeddings.shape[0]
    d = embeddings.shape[1]

    if support == "sphere":
        # If the support is in the sphere, the received random variable is Z
        # and it belongs to S^{d-1}
        csim = kappa * torch.matmul(embeddings, embeddings.T)
        const = (
            -math.log(kappa) * (d * 0.5 - 1)
            + 0.5 * d * math.log(2 * math.pi)
            + math.log(iv(0.5 * d - 1, kappa) + 1e-7)
            + math.log(k)
        )
        if reduction == "average":
            entropy = -torch.logsumexp(csim, dim=-1) + const  # -> log(p).sum
            entropy = entropy.mean()
        elif reduction == "expectation":
            logp = -torch.logsumexp(csim, dim=-1) + const
            entropy = F.softmax(-logp, dim=-1) * logp
            entropy = entropy.sum()
        else:
            raise NotImplementedError(f"Reduction type {reduction} not implemented")
    elif support == "discrete":
        # If the support is discrete, the received random variable is W and
        # it belongs to [d]
        embeddings_mean = embeddings.mean(0)
        if reduction == "expectation":
            entropy = -(embeddings_mean * torch.log(embeddings_mean + eps)).sum()
        elif reduction == "average":
            entropy = -torch.log(embeddings_mean + eps).mean()
        else:
            raise NotImplementedError(f"Reduction type {reduction} not implemented")
    else:
        raise NotImplementedError(f"Support type {support} not implemented")

    return entropy


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
    :param kappa: von Misses-Fisher kappa (https://en.wikipedia.org/wiki/Von_Mises-Fisher_distribution), defaults to 10
    :type kappa: float, optional
    :param support: support of the random variables, defaults to "sphere"
    :type support: str, optional
    :return: reconstruction error
    :rtype: torch.Tensor
    """
    d = projection1.shape[1]

    if support == "sphere":
        # If the support is in the sphere, the reconstruction is done with a
        # von Mises--Fisher distribution

        const = (
            -(0.5 * d - 1) * math.log(kappa)
            + 0.5 * d * math.log(2 * math.pi)
            + math.log(iv(0.5 * d - 1, kappa) + 1e-7)
        )
        csim = kappa * torch.sum(projection1 * projection2, dim=-1)
        rec = -csim + const

    elif support == "discrete":
        # If the support is discrete, the received random variable is W and
        # it belongs to [d]
        rec1 = -torch.sum(projection1 * torch.log(projection2 + eps), dim=-1)
        rec2 = -torch.sum(projection2 * torch.log(projection1 + eps), dim=-1)
        rec = 0.5 * (rec1 + rec2)

    rec_mean = rec.mean()
    return rec_mean


if __name__ == "__main__":
    # Some checks and usage examples
    embeddings = torch.randn(1000, 1000)
    embeddings = F.normalize(embeddings, 2, 1)
    print("Continuous entropy")
    print(
        "Entropy sphere plug-in estimator:",
        float(entropy(embeddings, support="sphere", reduction="expectation")),
    )
    print(
        "Entropy sphere Joe's estimator:",
        float(entropy(embeddings, support="sphere", reduction="average")),
    )
    print("Discrete entropy")
    print("Max entropy", math.log(1000))
    embeddings = F.softmax(torch.rand(1000, 1000), -1)
    entropy_uniform = entropy(
        embeddings, kappa=1, support="discrete", reduction="expectation"
    )
    print(
        "Entropy on uniform sample:",
        float(entropy_uniform),
    )
    embeddings = F.softmax(1000 * torch.rand(1000, 1000), -1)
    entropy_high_temp = entropy(
        embeddings, kappa=1, support="discrete", reduction="expectation"
    )
    print(
        "Entropy on uniform sample (high temp):",
        float(entropy_high_temp),
    )
    assert entropy_high_temp < entropy_uniform
    embeddings = torch.zeros_like(embeddings)
    embeddings[:, 0] = 1000
    embeddings = F.softmax(embeddings, -1)
    entropy_one_hot = entropy(
        embeddings, kappa=1, support="discrete", reduction="expectation"
    )
    assert entropy_one_hot < entropy_uniform
    print("Entropy on one_hot vector:", float(entropy_one_hot))
    print("Continuous reconstruction")
    embeddings = torch.rand(1000, 1000)
    projection1 = embeddings + 1 * torch.randn(1000, 1000)
    projection1 = F.normalize(projection1, 2, 1)
    projection2 = embeddings + 1 * torch.randn(1000, 1000)
    projection2 = F.normalize(projection2, 2, 1)
    assert reconstruction(projection1, projection1) < reconstruction(
        projection1, projection2
    )
    print("Reconstruction error:", float(reconstruction(projection1, projection2)))
    print("Discrete reconstruction")
    embeddings = torch.rand(1000, 1000)
    projection1 = 10 * embeddings + torch.randn(1000, 1000)
    projection1 = F.softmax(projection1, -1)
    projection2 = 10 * embeddings + torch.randn(1000, 1000)
    projection2 = F.softmax(projection2, -1)
    print(
        "Reconstruction error:",
        float(reconstruction(projection1, projection2, support="discrete")),
    )
    assert reconstruction(projection1, projection1) < reconstruction(
        projection1, projection2
    )
