from data.augment import RandomLinearCorellation, PCAAugment
import numpy as np

pca = PCAAugment(2)
target = np.arange(9).reshape(3,-1)
print(target)
pca.fit(target)
transformed = pca.transform(target)
print(transformed)
print(pca.inverse_transform(transformed))

rlc = RandomLinearCorellation(10)
target = np.arange(6).reshape(3,2)
print(target)
rlc.fit(target, 2)
transformed = rlc.transform(target)
print(transformed)
print(rlc.inverse_transform(transformed))