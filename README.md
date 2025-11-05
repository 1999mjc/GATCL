# GATCL

GATCL: Graph Attention Network Meets Contrastive Learning for Spatial Domain Identification

## Abstract

Spatial domain identification is an essential task to reveal spatial heterogeneity within tissues, providing insights into disease mechanisms, tissue development and the cellular microenvironment. In recent years, emerging as the new frontier in spatial domain identification, spatial multi-omics offers deeper insights into the complex interplay and functional dynamics of heterogeneous cell communities within their native tissue context. The existing methods rely on static graph structures that treat all neighboring cells uniformly, failing to capture the nuanced cellular interactions within the microenvironment and thus blurring functional boundaries. Furthermore, cross-modal reconstruction performance is often degraded by overfitting to modality-specific noise, which may impair the precise delineation of spatial domains. So we present GATCL, a novel deep learning framework that integrates a graph attention network with contrastive learning for robust spatial domain identification. First, GATCL leverages a graph attention mechanism to dynamically weigh the information from neighboring spots, adaptively modeling the complex cellular architecture. Second, it implements a cross-modal contrastive learning strategy that forces representations from the same spatial location to be similar while pushing those from different locations apart, thereby achieving robust alignment between modalities. Comprehensive benchmarking across six distinct datasets(spanning transcriptome, proteome and chromatin) reveals that GATCL demonstrates superior performance than seven representative methods on six key evaluation metrics.


## Installation Requirements

```txt
python=3.8
numpy=1.24.3
pandas=2.0.3
scanpy=1.9.8
scipy=1.10.1
scikit-learn=0.24.0
torch=2.4.1
anndata=0.9.2
matplotlib=3.7.5
```
## Example
```txt
example.py
```
