# Data Target Augmentation
This code is implemented for augmenting data target using Principal Component Analysis and Random Linear Target Correlation proposed by Tsoumakas et al.

---

# PCAAugment
PCA Transformation for target data. This is implemented using 
Scikit-learn library, using its default values except for 
the number of principal components


Parameters
----------
n_components : int, default=2
    Specify the number of principal component for the new
    augmented target data

Examples
--------
```
>>> from data.augment import PCAAugment
>>> import numpy as np
>>> pca = PCAAugment(2)
>>> target = np.arange(6).reshape(3,-1)
>>> print(target)
    [[0 1 2]
     [3 4 5]
     [6 7 8]]
>>> pca.fit(target)
>>> transformed = pca.transform(target)
>>> print(transformed)
    [[-5.19615242e+00 -1.33226763e-15]
     [ 0.00000000e+00  0.00000000e+00]
     [ 5.19615242e+00  1.33226763e-15]]
>>> print(pca.inverse_transform(transformed))
    [[4.4408921e-16 1.0000000e+00 2.0000000e+00]
     [3.0000000e+00 4.0000000e+00 5.0000000e+00]
     [6.0000000e+00 7.0000000e+00 8.0000000e+00]]
```

---

# RandomLinearCorrelation
Random Linear Correlation transformation will transform
the data with a random correlation matrix


Parameters
----------
r : int, default = 500
    Specify the number of new dimension of the target data.
    Expected to be higher than the original data

random_seed : int, default = None
    Specify the random seed generator for reproducibility

    
Examples
--------
```
>>> from data.augment import RandomLinearCorellation
>>> import numpy as np
>>> rlc = RandomLinearCorellation(10)
>>> target = np.arange(6).reshape(3,2)
>>> print(target)
    [[0 1]
     [2 3]
     [4 5]]
>>> rlc.fit(target, 2)
>>> transformed = rlc.transform(target)
>>> print(transformed)
    [[0.73689971 0.87557506 0.50592154 0.57558735 0.5981449  0.96570957
      0.1765568  0.56997664 0.40511597 0.43316827]
     [3.74793267 3.71664164 1.56196224 2.00330381 2.61513391 3.780683
      1.91782421 1.82394165 1.77457261 1.96605611]
     [6.75896563 6.55770822 2.61800295 3.43102026 4.63212293 6.59565642
      3.65909163 3.07790666 3.14402925 3.49894395]]
>>> print(rlc.inverse_transform(transformed))
    [[-3.11735226e-16  1.00000000e+00]
     [ 2.00000000e+00  3.00000000e+00]
     [ 4.00000000e+00  5.00000000e+00]]
```