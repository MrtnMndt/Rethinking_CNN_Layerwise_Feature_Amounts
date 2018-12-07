#Rethinking Layer-wise Feature Amounts in Convolutional Neural Network Architectures

In this repository we provide open-source [PyTorch](https://pytorch.org) code for our NeurIPS 2018 CRACT workshop paper, where we characterize the classification accuracy of a family of VGG-like models by shifting a constant amount of total features to different convolutional layers and show how large amounts of features in early layers challenge common design assumptions.

If you use or extend the code please cite our work:

> Martin Mundt, Sagnik Majumder, Tobias Weis, Visvanathan Ramesh, "Rethinking Layer-wise Feature Amounts in Convolutional Neural Network Architectures", International Conference on Neural Information Processing Systems (NeurIPS) 2018, Critiquing and Correcting Trends in Machine Learning (CRACT) Workshop

You can find the complete workshop here: [CRACT](https://ml-critique-correct.github.io) 

and our paper here: [PAPER](https://www.dropbox.com/s/vjt0on2dxizzv8v/CRACT_2018_paper_19.pdf?dl=0)


# Interactive visualization of results

We provide the results presented in the paper and additional results in interactive form here. You can rotate the plots, click and hover over datapoints to see precise architecture accuracy and parameter counts.

## CIFAR-10 VGG-D architecture variants
</p>
<p align="center">
<iframe width="700" height="533" frameborder="0" scrolling="no" src="//plot.ly/~martinmundt/10.embed"></iframe>
</p>  

### Corresponding training accuracies
</p>
<p align="center">
<iframe width="700" height="533" frameborder="0" scrolling="no" src="//plot.ly/~martinmundt/12.embed"></iframe>
</p>  

## Fashion-MNIST VGG-D architecture variants
</p>
<p align="center">
<iframe width="700" height="533" frameborder="0" scrolling="no" src="//plot.ly/~martinmundt/16.embed"></iframe>
</p>  

### Corresponding training accuracies
</p>
<p align="center">
<iframe width="700" height="533" frameborder="0" scrolling="no" src="//plot.ly/~martinmundt/18.embed"></iframe>
</p>  

## MNIST VGG-D architecture variants
</p>
<p align="center">
<iframe width="700" height="533" frameborder="0" scrolling="no" src="//plot.ly/~martinmundt/22.embed"></iframe>
</p>  

### Corresponding training accuracies
</p>
<p align="center">
<iframe width="700" height="533" frameborder="0" scrolling="no" src="//plot.ly/~martinmundt/24.embed"></iframe>
</p>  

## Short discussion
In addition to the results reported in the paper we observe that architectures that are characterized through small xi (large amount of features in early layers) overfit less than the typically used counterparts. 

## Additional results for VGG-A architecture variants
We observe similar trends for architectures with less layers (based on the VGG-A variant) that were not reported in the paper due to space constraints. 

### CIFAR-10
</p>
<p align="center">
<iframe width="700" height="533" frameborder="0" scrolling="no" src="//plot.ly/~martinmundt/8.embed"></iframe>
</p>  

### Fashion-MNIST
</p>
<p align="center">
<iframe width="700" height="533" frameborder="0" scrolling="no" src="//plot.ly/~martinmundt/14.embed"></iframe>
</p>  

### MNIST
</p>
<p align="center">
<iframe width="700" height="533" frameborder="0" scrolling="no" src="//plot.ly/~martinmundt/20.embed"></iframe>
</p>  