# Quantum Machine Learning for Building Detection in LiDAR Point Clouds

This project explores the use of quantum machine learning techniques, specifically Quantum Support Vector Machines (QSVMs) and Quantum Boosting (QBoost), for the task of building detection from 3D LiDAR point cloud data. The goal is to compare the performance of these quantum models against their classical counterparts, such as Support Vector Machines (SVMs) and AdaBoost, on this spatial classification problem.

## Data

- **Study area**: Three 0.25 km² regions in Vancouver, B.C.
  - **Area 1**: Dense townhomes with scattered large vegetation in Kitsilano
  - **Area 2**: Large commercial buidlings, roads, and minimal vegetation in downtown Vancouver
  - **Area 3**: Large, sparsely distributed homes surrounded by forest in Point Grey
- **Source**: Vancouver Open Data Portal [[Vancouver LiDAR 2022](https://opendata.vancouver.ca/explore/dataset/lidar-2022/information/)]
- **Mean point density**: 49 points/m²
- **Pre-classification**: 8 classes (unclassified, ground, low vegetation, high vegetation, water, buildings, other, noise)
- **Preprocessing**: Noise and unclassified points discarded

### Feature Extraction

Four features were extracted for classification:
1. **Normalized height**: The height of each point above ground, using a cloth simulation filter to generate a DEM.
2. **Height variation**: The [median absolute deviation](https://en.wikipedia.org/wiki/Median_absolute_deviation) of the normalized hieght values within a disk of radius $r = 0.5m$.
3. **Normal variation**: The negative of the average dot product of each normal with other normals within a disk of radius $r = 0.5m$, where normal vectors are computed using standard PCA methods. This value gives a measure of planarity near each point. 
4. **Log-intensity**: The logarithm of the amplitude of the response reflected back to the laser scanner.  This can provide information of about the properties of the reflected surface.

## Experiments

The experiments compare the performance of:

1. Classical Support Vector Machines (SVM)
2. Support Vector Machines trained on Quantum Annealers (QSVM)
3. SVMs and QSVMs with various quantum kernels
4. AdaBoost ensemble learning using QSVM as a weak learner
5. QBoost ensemble learning using QSVM as a weak learner
6. Ensemble weighted by the softmax of the Matthew's correlation coefficent for each weak learner (softmax QSVM)

### Hyperparameter Optimization

- **Data**: 1,000-sample training set and a 100,000-sample validation set sampled uniformly at random for each area
- **Search algorithm**: Grid search was used on an extensive search space covering all hyperparameters for each model
- **Weak learners**: QSVMs trained on small subsets of the training set were used as weak learners of ensemble algorithms
- **Model training**: Each model was trained using 3-fold cross-validation on the training set
- **Model evaluation**: [Matthew's correlation coefficient](https://en.wikipedia.org/wiki/Phi_coefficient) used as an evaluation metric to accurately evaluate models in the presence of class imbalance
- **Results (MCC)**:
  - QSVM and QBoost generally outperfromed equivalent classical models (SVM and AdaBoost)
  - QBoost algorithm achieved best overall performance
  - Classical Gaussian RBF kernel outperformed all quantum kernels, but data re-uploading (DRU) kernel was a close second

### Evaluation of Optimal Models

Models selected through hyperparameter optimization were trained on a 5,000-sample training set and evaluated on a 100,000-sample validation set.  All QUBO problems were solved using quantum annealing or a hybrid quantum-classical solver.
- **Results**:

| Model                | Area 1 | Area 2 | Area 3 |
| :------------------- | :----: | :----: | :----: |
| SVM                  | 0.624  | **0.744**  | 0.489  |
| QSVM                 | **0.662**  | 0.741  | 0.607  |
| SVM with DRU Kernel  | 0.620  | 0.724  | 0.435  |
| QSVM with DRU Kernel | 0.653  | 0.701  | **0.616**  |
| AdaBoost             | 0.615  | 0.686  | 0.519  |
| QBoost               | 0.624  | 0.695  | 0.580  |
| Softmax QSVM         | 0.610  | 0.611  | 0.491  |

## Key Findings

The main findings of this project include:

1. The QSVM model outperformed the classical SVM in areas with high class imbalance, suggesting that quantum models may be more effective at handling imbalanced data.
2. The QBoost ensemble model achieved the best overall performance on the 1000-sample training set, demonstrating the potential of quantum boosting techniques for spatial classification tasks.
3. The quantum DRU kernel performed comparably to the classical Gaussian RBF kernel, indicating that quantum kernels can be a viable alternative in some applications.
4. Careful hyperparameter tuning was crucial for optimizing the performance of both the quantum and classical models.

## Conclusion

Quantum machine learning techniques show promising results for spatial data analysis, with performance comparable to or exceeding equivalent classical methods in some cases. The data re-uploading quantum kernel with full entanglement pattern demonstrated superior performance among the tested quantum kernels, slightly outperforming the classical RBF kernel.

## Future Work

Potential future directions for this project include:

1. Development of more sophisticated quantum kernels optimized for spatial data.
2. Application of the QSVM architecture to multi-class classification of LiDAR point clouds.
3. Investigation of additional LiDAR features (e.g. return number, point density, eigenvalue-derived features) to enhance classification performance.
4. Exploration of methods for reducing QUBO complexity while retaining solution quality.
5. Analysis of quantum model scaling behaviour with larger datasets, more diverse geographic areas, and improved quantum hardware.

## References

- D. Willsch, M. Willsch, H. De Raedt, and K. Michielsen. Support vector machines on the d-wave quantum annealer. _Computer Physics Communications_, 248:107006, 2020.
- Gabriele Cavallaro, Dennis Willsch, Madita Willsch, Kristel Michielsen, and Morris Riedel. Approaching remote sensing image classification with ensembles of support vector machines on the d-wave quantum annealer. In _IGARSS 2020 - 2020 IEEE International Geoscience and Remote Sensing Symposium_, pages 1973–1976, 2020.
- Yoav Freund and Robert E Schapire. A decision-theoretic generalization of on-line learning and an application to boosting. _Journal of Computer and System Sciences_, 55(1):119–139, 1997.
- Hartmut Neven, Vasil S. Denchev, Geordie Rose, and William G. Macready. Qboost: Large scale classifier training withadiabatic quantum optimization. In Steven C. H. Hoi and Wray Buntine, editors, _Proceedings of the Asian Conference on Machine Learning_, volume 25 of _Proceedings of Machine Learning Research_, pages 333–348, Singapore Management University, Singapore, 04–06 Nov 2012. PMLR.
- Hartmut Neven, Vasil S. Denchev, Geordie Rose, and William G. Macready. Training a binary classifier with the quantum adiabatic algorithm, 2008.
- Vojtech Havlıcek, Antonio D. Corcoles, Kristan Temme, Aram W. Harrow, Abhinav Kandala, Jerry M. Chow, and Jay M. Gambetta. Supervised learning with quantum-enhanced feature spaces. _Nature_, 567(7747):209–212, Mar 2019.
- Shu Su, Kazuya Nakano, and Kazune Wakabayashi. Building detection from aerial lidar point cloud using deep learning. _The International Archives of the Photogrammetry, Remote Sensing and Spatial Information Sciences_, XLIII-B2-2022:291–296, 2022.
- Suresh K. Lodha, Edward J. Kreps, David P. Helmbold, and Darren Fitzpatrick. Aerial lidar data classification using support vector machines (svm). In _Third International Symposium on 3D Data Processing, Visualization, and Transmission (3DPVT’06)_, pages 567–574, 2006.
- Michael J. Bremner, Richard Jozsa, and Dan J. Shepherd. Classical simulation of commuting quantum computations implies collapse of the polynomial hierarchy. _Proceedings of the Royal Society A: Mathematical, Physical and Engineering Sciences_, 467:459 – 472, 2010.
- qml.iqpembedding. <https://docs.pennylane.ai/en/stable/code/api/pennylane.IQPEmbedding.html>. Accessed Aug 15, 2024.
- K. Mitarai, M. Negoro, M. Kitagawa, and K. Fujii. Quantum circuit learning. _Phys. Rev. A_, 98:032309, Sep 2018.
- Edward Farhi, Jeffrey Goldstone, and Sam Gutmann. A quantum approximate optimization algorithm, 2014.
- Adrian Perez-Salinas, Alba Cervera-Lierta, Elies Gil-Fuster, and Jose I. Latorre. Data re-uploading for a universal quantum classifier. _Quantum_, 4:226, February 2020.
- City of Vancouver. Lidar 2022, 2022. Vancouver Open Data Portal.
- Qiskit contributors. Qiskit: An open-source framework for quantum computing, 2023.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.