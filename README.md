# Overview

This is an ablation study of the Bayesian classification head we proposed in our preliminary work [Hierarchical Relationships: A New Perspective to Enhance Scene Graph Generation](https://arxiv.org/abs/2303.06842) accepted at NeurIPS 2023 New Frontiers in Graph Learning Workshop ([GLFrontiers](https://glfrontiers.github.io/)) and NeurIPS 2023 [Queer in AI](https://www.queerinai.com/neurips-2023).

We started from the codebase from [Unbiased Scene Graph Generation from Biased Training](https://arxiv.org/abs/2002.11949), which provides re-implementations of several SOTA SGG frameworks and the evaluation metrics. 
We provide the testing results of predicate classifications(PLS) task on Visual Genome before and after we integrate
our Bayesian head to three existing works: NeuralMotifs, VTransE and VCTree.

