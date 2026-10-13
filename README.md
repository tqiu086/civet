## CIVET

In this question, I started from the civet repo, I read the Xu et al. 2025 paper, have the idea of a support set for the latent. I read Liu et al. 2023 studies universal adversarial perturbations on a wireless VAE, where one perturbation is shared across inputs. That’s why the setup here has the same noise v on both x1 and x2.

I used the repo’s relu FC-VAE and its ibp idea: bound the encoder, turn the latent into a support set, then push that set through the decoder. I had to write a few helper scripts of my own, since the repo’s training code doesn’t run as is (a syntax error in utils.py, a missing model.ibp, and a couple of robust/unrobust bugs). I left the original files alone and put everything in a separate hw2/ folder.

- I trained a standard FC_VAE and a CIVET-trained one on MNIST (eps = 0.1). The civet one is the one for results. 
- I implement symbolic bounds through the encoder in the shared v. A difference tracker follows x1−x2 through every layer. After layer 1 it is exactly W(x1−x2), because the shared v cancels.
- I implement the support set, the latent is $\mu(v) + \sigma \cdot e$. I picked a radius c so that all 20 noice coordinates (10 per input) stay inside with probability at least 95% for any v. 
- I implement the same tracker through the decoder, then a bound on the MSE. I checked it against a PGD attack over v and the latent noise, which gives an empirical lower bound. The attack never went above any bound.
- I have a joint bound on (err1 + err2)/2 under the shared v, using the tracker's relation r1 - r2 = d_out - (x1-x2). Did the noisy-target version gainst x_i + v (wasnt sure which reading of error the problem wants). Sweep eps from 0.02 to 0.2.

- I picked deepPoly-stype tracker because it keeps the relation between the two inputs, where plain intervals throw it away. 
- I use symmetric support set not the civet asymmetric box. Because I want the tracker have a noise term that doesn't depend on $\mu$. Though the civet box is a bit tighter, it doesn't carry the tracker, and mixing the two would possibly break 95%.
- I used random pairs, I just have no idea with what pairs to start with. I tried same-class, nearest-neighbour and different-class paires, 20 each. Results look the same. 

**Results**:
The bounds are valid and not vacuous on the civet model: at eps = .1, the joint bound is about 0.115 aagainst an attack lower bound of about 0.0052. But the tracker doesn't help much: at most about 1% in the best case, and zero in most settings. I guess the latent noise (about .56 per side) is bigger than the information the tracker has, I will test it next time. The attack lose very little when the noise is shared, meaning most of the gap to the bound is loose in the bound. If I have more time, I would try tighter relaxations, like alpha-crown. 


