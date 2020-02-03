# Standard-method-apply-on-MNIST-data

 Use the MNIST digit recognition dataset having 10 classes for the purpose of this code from this link
 [MNIST](http://yann.lecun.com/exdb/mnist/)
 
 # DIMENSIONALITY REDUCTION
 Perfrom These task using [code.py](https://github.com/sunil-17112/Standard-method-apply-on-MNIST-data/blob/master/code.py) and compare with [Report](https://github.com/sunil-17112/Standard-method-apply-on-MNIST-data/blob/master/Report.pdf).
 
 a. Compute the global mean and covariance of the data.
 
 b. Implement PCA and FDA from scratch.
 
 c. Visualize data using a scatter plot after applying PCA & FDA. (You can
 transform the data into 2 dimensions and then plot it.)
 
 d. Implement the LDA discriminant function from scratch.
 
 e. Apply PCA with 95% eigen energy on MNIST and then LDA for classification
 and report the accuracy on test data.
 
 f. Visualize and analyze the eigenvectors obtained using PCA (only for
 eigenvectors obtained in part(e). I.e., Display eigenvectors by converting
 them into image form).
 
 g. Perform step(e) with different eigen energy mentioned below and show the
 comparisons and analysis on accuracy.


  ● 70% eigen energy
  ● 90% eigen energy
  ● 99% eigen energy
  
 h. Apply FDA on MNIST and then LDA for classification and report the accuracy
 on test data.
 
 i. Perform PCA then FDA. Classify the transformed datasets using LDA.
 Analyze the results on Accuracy.
 
 # Noise Reduction
 
 Perform the following steps on the given dataset(MNIST) using [noise.py](https://github.com/sunil-17112/Standard-method-apply-on-MNIST-data/blob/master/noise.py) :
 
 Before this comment_in Line 116 and Comment_out Line 118 - 123 in [code.py](https://github.com/sunil-17112/Standard-method-apply-on-MNIST-data/blob/master/code.py)
 
 a. Add Gaussian noise to the dataset. (NOTE: You can take mean=0 and
 variance can be varied upon your choice such that the noise reduction can
 be seen clearly from the image.)
 
 b. Perform PCA on the noisy dataset for Noise Reduction.
 
 c. Visualize the dataset before & after noise reduction. (Report the images as
 shown below. Linear PCA in the below image refers to normal PCA only.).
 
 d. Report the number of components for which PCA works the best in Noise
 Reduction.
 
 After performing all task, Delete the changes from [code.py](https://github.com/sunil-17112/Standard-method-apply-on-MNIST-data/blob/master/code.py) that was done in before.
 
