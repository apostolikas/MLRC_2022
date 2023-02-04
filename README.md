## Concept Subspace Network

Code for the concept subspace network (CSN) as introduced by Tucker et al. in "Prototype Based Classification from Hierarchy to Fairness" in ICML 2022.
Reproducibility Study

This github repository contains the reimplementation of the paper "Prototype Based Classification from Hierarchy to Fairness". Our aim was to verify the claims of the paper and convert the already available Tensorflow code to PyTorch, since it gave us more flexibility to understand in detail the architecture introduced in the paper.

The architecture is relatively straightforward, as it consists of a VAE and a modified PCN. The PyTorch-based implementation was tested on several fair and hierarchical tasks. Overall, the results obtained were very similar and in some cases identical with the ones stated in the paper. Therefore, our efforts replicating the paper, while also converting the code using a more research-oriented framework, were generally successful.

To get started, install the environment using the "our_env.yml" file. For the experiments a Python 3.10 interpreter is used.

In order to run the scripts, you can use the inference.ipynb file in the "submittion" branch for either train the model and evaluate (set inference= False) or use the 
checkpoints and only evaluate (inference = True)




