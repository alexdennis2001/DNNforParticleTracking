# Using Machine Learning for Particle Tracking at the Large Hadron Collider

# Introduction
The ENLACE 2023 Research Summer Camp at UCSD gave rise to this project, which was completed within a span of 7 weeks. The outcomes of this endeavor were documented on a poster (accessible in the repository), fulfilling the criteria for University Students' project submissions. The majority of the code was crafted using the Pytorch framework.

# Description 
In this study, an advanced Deep Neural Network (DNN) was meticulously developed to process and analyze a dataset sourced from the Large Hadron Collider (LHC) experiment, utilizing data collected by the Compact Muon Solenoid (CMS) Detector. The main goal of the DNN was the following: the Line Segment Tracking (LST) algorithm is designed in a way that it can reconstruct a subset of particle tracks from a large point cloud (since it is not feasible to reconstruct all of them). This process occurs iteratively, as tracks are gradually constructed, allowing for substantial parallelization of the algorithm. A critical step in this iterative procedure involves the creation of line segments (LSs), which represent fundamental building blocks. The DNN was trained to recognize line segments that accurately correspond to a real particle's trajectory.

Given the presence of a magnetic field within the CMS Detector, charged particles resulting from collisions display curved paths. These curves hold vital information concerning the particles' charge and momentum. The DNN was meticulously crafted to handle these curved trajectories, assessing the reliability of individual line segments for subsequent analysis or considering their exclusion if inaccuracies are detected. In our context, a 'track' is not directly processed by the DNN; rather, the network focuses on processing line segments, which are the fundamental constituents of a complete 'track.' A 'track' is defined as the complete trajectory of a particle, while a line segment represents the smallest identifiable unit within this trajectory.

In this investigation, we harnessed the capabilities of deep learning methodologies to enhance the precision and efficiency of particle track selection. This, in turn, contributed to the accuracy of charge and momentum determination within the CMS Detector.

Furthermore, we conducted an in-depth comparison of the performance of our DNN versus a Graph Neural Network (GNN) in track classification. We evaluated metrics such as such as histograms, loss curves, F1 scores and ROC curves to identify optimal threshold values. Notably, our findings revealed that the performance of the simplest DNN model was comparable to more complex variants like the GNN that consisted of 3 neural networks that we compared it too.

In conclusion, this study demonstrated the remarkable potential of the developed DNN in enhancing particle track selection within the CMS Detector. The utilization of various analytical tools, including ROC curves and threshold optimization techniques, further underscored the effectiveness of our approach and highlighted the competitiveness of even the simplest DNN architecture.

# Results
![Final Poster](FinalPoster.png)

# Credits
* Alejandro Daniel Dennis Hernandez
* Abraham Jhared Flores Azcona
* Frank Wuerthwein (PI)
* Jonathan Guiang (mentor)
