# Quantum SVM and QBoost

See [here](https://doi.org/10.1016/j.cpc.2019.107006) and [here](https://proceedings.mlr.press/v25/neven12.html).


# Quantum Machine Learning for Building Detection in LiDAR Point Clouds

This repository contains the code for experiments comparing classical and quantum machine learning algorithms for building detection in LiDAR point cloud data.

## Overview

We explore the application of quantum machine learning techniques to the task of building detection in urban environments using LiDAR point cloud data. The experiments compare the performance of:

1. Classical Support Vector Machines (SVM)
2. Quantum Support Vector Machines (QSVM)
3. AdaBoost ensemble learning
4. QBoost ensemble learning
5. SVMs adn QSVMs with various quantum kernels

## Data

- Study area: 1 km² residential region in Kitsilano, Vancouver, B.C.
- Source: Vancouver Open Data Portal
- Mean point density: 46 points/m²
- Pre-classified into 8 classes (ground, vegetation, buildings, etc.)

### Feature Extraction

Four features were extracted for classification:
1. Normalized height - the height of each point above ground
2. Height variation - the absolute difference between minimum and maximum values of normalized height within a disk of radius $r = 0.5m$.
3. Normal variation - the negative of the average dot product of each normal with other normals within a disk of radius $r = 1m$. This value gives a measure of planarity near each point.
4. Return intensity - the amplitude of the response reflected back to the laser scanner.  This can provide information of about the properties of the reflected surface.

## Experiments and Results

### 1. SVM vs. QSVM

- Training set: 100 random points
- Kernel: Radial Basis Function (RBF)
- Results:
  - Classical SVM: 92% (validation), 87.36% (full dataset)
  - Quantum SVM: 92% (validation), 87.66% (full dataset)

### 2. AdaBoost vs. QBoost

- Training set: 1000 random points
- 50 weak classifiers (QSVMs trained on random 20-point subsets of the training set)
- Results:
  - AdaBoost: 90% (validation), 88.33% (full dataset)
  - QBoost: 90% (validation), 90.14% (full dataset)

### 3. Quantum Kernel Methods

Tested 5 quantum kernels:
1. Pauli
2. IQP (Instantaneous Quantum Polynomial)
3. Data re-uploading
4. QAOA-inspired
5. Polynomial

Best performing quantum kernel:
- Data re-uploading with 'full' entanglement
- Classical SVM: 89.7% (validation), ????% (full dataset)
- Quantum SVM: 92.3% (validation), ????% (full dataset)

## Conclusion

Quantum machine learning techniques show promising results for building detection in LiDAR point cloud data, with performance comparable to or exceeding classical methods in some cases.

## References

[1] D. Willsch, M. Willsch, H. De Raedt, and K. Michielsen. Support vector machines on the d-wave quantum annealer. _Computer Physics Communications_, 248:107006, 2020.

[2] Gabriele Cavallaro, Dennis Willsch, Madita Willsch, Kristel Michielsen, and Morris Riedel. Approaching remote sensing image classification with ensembles of support vector machines on the d-wave quantum annealer. In _IGARSS 2020 - 2020 IEEE International Geoscience and Remote Sensing Symposium_, pages 1973–1976, 2020.

[3] Yoav Freund and Robert E Schapire. A decision-theoretic generalization of on-line learning and an application to boosting. _Journal of Computer and System Sciences_, 55(1):119–139, 1997.

[4] Hartmut Neven, Vasil S. Denchev, Geordie Rose, and William G. Macready. Qboost: Large scale classifier training withadiabatic quantum optimization. In Steven C. H. Hoi and Wray Buntine, editors, _Proceedings of the Asian Conference on Machine Learning_, volume 25 of _Proceedings of Machine Learning Research_, pages 333–348, Singapore Management University, Singapore, 04–06 Nov 2012. PMLR.

[5] Hartmut Neven, Vasil S. Denchev, Geordie Rose, and William G. Macready. Training a binary classifier with the quantum adiabatic algorithm, 2008.

[6] Vojtech Havlıcek, Antonio D. Corcoles, Kristan Temme, Aram W. Harrow, Abhinav Kandala, Jerry M. Chow, and Jay M. Gambetta. Supervised learning with quantum-enhanced feature spaces. _Nature_, 567(7747):209–212, Mar 2019.

[7] Shu Su, Kazuya Nakano, and Kazune Wakabayashi. Building detection from aerial lidar point cloud using deep learning. _The International Archives of the Photogrammetry, Remote Sensing and Spatial Information Sciences_, XLIII-B2-2022:291–296, 2022.

[8] Suresh K. Lodha, Edward J. Kreps, David P. Helmbold, and Darren Fitzpatrick. Aerial lidar data classification using support vector machines (svm). In _Third International Symposium on 3D Data Processing, Visualization, and Transmission (3DPVT’06)_, pages 567–574, 2006.

[9] Michael J. Bremner, Richard Jozsa, and Dan J. Shepherd. Classical simulation of commuting quantum computations implies collapse of the polynomial hierarchy. _Proceedings of the Royal Society A: Mathematical, Physical and Engineering Sciences_, 467:459 – 472, 2010.

[10] qml.iqpembedding. <https://docs.pennylane.ai/en/stable/code/api/pennylane.IQPEmbedding.html>. Accessed Aug 15, 2024.

[11] K. Mitarai, M. Negoro, M. Kitagawa, and K. Fujii. Quantum circuit learning. _Phys. Rev. A_, 98:032309, Sep 2018.

[12] Edward Farhi, Jeffrey Goldstone, and Sam Gutmann. A quantum approximate optimization algorithm, 2014.

[13] Adrian Perez-Salinas, Alba Cervera-Lierta, Elies Gil-Fuster, and Jose I. Latorre. Data re-uploading for a universal quantum classifier. _Quantum_, 4:226, February 2020.

[14] Qiskit contributors. Qiskit: An open-source framework for quantum computing, 2023.
