                                                  OPV2V: An Open Benchmark Dataset and Fusion Pipeline for
                                                      Perception with Vehicle-to-Vehicle Communication
                                                                  Runsheng Xu1∗ , Hao Xiang1∗ , Xin Xia1 , Xu Han1 , Jinlong Li2 , Jiaqi Ma1


                                            Abstract— Employing Vehicle-to-Vehicle communication to                   information could be raw data, intermediate features, single
                                         enhance perception performance in self-driving technology has                CAV’s detection output, and metadata e.g., timestamps and
                                         attracted considerable attention recently; however, the absence              poses. Despite the big potential in this field, it is still in
                                         of a suitable open dataset for benchmarking algorithms has
                                         made it difficult to develop and assess cooperative perception               its infancy. One of the major barriers is the lack of a large
                                         technologies. To this end, we present the first large-scale                  open-source dataset. Unlike the single vehicle’s perception

arXiv:2109.07644v5 \[cs.CV\] 20 Jun 2022

                                         open simulated dataset for Vehicle-to-Vehicle perception. It                 area where multiple large-scale public datasets exist [12],
                                         contains over 70 interesting scenes, 11,464 frames, and 232,913              [13], [14], most of the current V2V perception algorithms
                                         annotated 3D vehicle bounding boxes, collected from 8 towns                  conduct experiments based on their customized data [15],
                                         in CARLA and a digital town of Culver City, Los Angeles.
                                         We then construct a comprehensive benchmark with a total of                  [16], [17]. These datasets are either too small in scale and
                                         16 implemented models to evaluate several information fusion                 variance or they are not publicly available. Consequently,
                                         strategies (i.e. early, late, and intermediate fusion) with state-           there is no large-scale dataset suitable for benchmarking
                                         of-the-art LiDAR detection algorithms. Moreover, we propose                  distinct V2V perception algorithms, and such deficiency will
                                         a new Attentive Intermediate Fusion pipeline to aggregate                    preclude further progress in this research field.
                                         information from multiple connected vehicles. Our experiments
                                         show that the proposed pipeline can be easily integrated                         To address this gap, we present OPV2V, the first large-
                                         with existing 3D LiDAR detectors and achieve outstanding                     scale Open Dataset for Perception with V2V communication.
                                         performance even with large compression rates. To encourage                  By utilizing a cooperative driving co-simulation framework
                                         more researchers to investigate Vehicle-to-Vehicle perception,               named OpenCDA [18] and CARLA simulator [19], we col-
                                         we will release the dataset, benchmark methods, and all related              lect 73 divergent scenes with a various number of connected
                                         codes in https://mobility-lab.seas.ucla.edu/opv2v/.
                                                                                                                      vehicles to cover challenging driving situations like severe
                                                                 I. INTRODUCTION                                      occlusions. To narrow down the gap between the simulation
                                            Perceiving the dynamic environment accurately is critical                 and real-world traffic, we further build a digital town of
                                         for robust intelligent driving. With recent advancements                     Culver City, Los Angeles with the same road topology and
                                         in robotic sensing and machine learning, the reliability of                  spawn dynamic agents that mimic the realistic traffic flow on
                                         perception has been significantly improved [1], [2], [3], and                it. Data samples are shown in Fig. 1 and Fig. 4. We bench-
                                         3D object detection algorithms have achieved outstanding                     mark several state-of-the-art 3D object detection algorithms
                                         performance either with LiDAR point clouds [4], [5], [6],                    combined with different multi-vehicle fusion strategies. On
                                         [7] or multi-sensor data [8], [9].                                           top of that, we propose an Attentive Intermediate Fusion
                                            Despite the recent breakthroughs in the perception field,                 pipeline to better capture interactions between connected
                                         challenges remain. When the objects are heavily occluded                     agents within the network. Our experiments show that the
                                         or have small scales, the detection performance will dramat-                 proposed pipeline can efficiently reduce the bandwidth re-
                                         ically drop. Such problems can lead to catastrophic accidents                quirements while achieving state-of-the-art performance.
                                         and are difficult to solve by any algorithms since the sensor                                    II. R ELATED W ORK
                                         observations are too sparse. An example is revealed in
                                         Fig. 1a. Such circumstances are very common but dangerous                    Vehicle-to-Vehicle Perception: V2V perception methods
                                         in real-world scenarios, and these blind spot issues are                     can be divided into three categories: early fusion, late fusion,
                                         extremely tough to handle by a single self-driving car.                      and intermediate fusion. Early fusion methods [11] share raw
                                            To this end, researchers started recently investigating                   data with CAVs within the communication range, and the ego
                                         dynamic agent detection in a cooperative fashion, such as                    vehicle will predict the objects based on the aggregated data.
                                         USDOT CARMA [10] and Cooper [11]. By leveraging the                          These methods preserve the complete sensor measurements
                                         Vehicle-to-Vehicle (V2V) communication technology, differ-                   but require large bandwidth and are hard to operate in
                                         ent Connected Automated Vehicles (CAVs) can share their                      real time [15]. In contrast, late fusion methods transmit
                                         sensing information and thus provide multiple viewpoints                     the detection outputs and fuse received proposals into a
                                         for the same obstacle to compensate each other. The shared                   consistent prediction. Following this idea, Rauch et al. [20]
                                                                                                                      propose a Car2X-based perception module to jointly align
                                           *Equal contribution                                                        the shared bounding box proposals spatially and temporally
                                           1 University of California, Los Angeles, Mobility Lab. {rxx3386,
                                                                                                                      via an EKF. In [21], a machine learning-based method is
                                         haxiang, x35xia, hanxu417, jiaqima}@ucla.edu
                                           2 Cleveland   State   University,   Cleveland   Vision   and   AI   Lab,   utilized to fuse proposals generated by different connected
                                         j.li56@vikes.csuohio.edu                                                     agents. This stream of work requires less bandwidth, but

(a)

                                                                (b)

Fig. 1: Two examples from our dataset. Left: Screenshot of the
constructed scenarios in CARLA. Middle: The LiDAR point cloud collected
by the ego vehicle. Right: The aggregated point clouds from all
surrounding CAVs. The red circles represent the cars that are invisible
to the ego vehicle due to the occlusion but can be seen by other
connected vehicles. (a): The ego vehicle plans to turn left in a
T-intersection and the roadside vehicles block its sight to the incoming
traffic. (b): Ego-vehicle’s LiDAR has no measurements on several cars
because of the occlusion caused by the dense traffic.

      Sensors        Details
      4x Camera      RGB, 800 × 600 resolution, 110◦ FOV
                     64 channels, 1.3 M points per second,
      1x LiDAR       120 m capturaing range, −25◦ to 5◦
                     vertical FOV, ±2 cm error
      GPS & IMU      20 mm positional error, 2◦ heading error

              TABLE I: Sensor specifications.




       Fig. 2: Sensor setup for each CAV in OPV2V.                    Fig. 3: Examples of the front camera data and BEV map
                                                                      of two CAVs in OPV2V. The yellow, green, red, and white
                                                                      lanes in the BEV map represent the lanes without traffic light

the performance of the model is highly dependent on each control, under
green light control, under red light control, and agent’s performance
within the vehicular network. To meet crosswalks. requirements of both
bandwidth and detection accuracy, intermediate fusion \[22\], \[15\] has
been investigated, where intermediate features are shared among
connected vehicles there is no large-scale open-source dataset for V2V
percep- and fused to infer the surrounding objects. F-Cooper \[22\] tion
in the literature. Some work \[11\], \[22\] adapts KITTI \[14\] utilizes
max pooling to aggregate shared Voxel features, to emulate V2V settings
by regarding the ego vehicle at and V2VNet \[15\] jointly reason the
bounding boxes and different timestamps as multiple CAVs. Such synthetic
pro- trajectories based on shared messages. cedure is unrealistic and
not appropriate for V2V tasks Vehicle-to-Vehicle Dataset: To the best of
our knowledge, since the dynamic agents will appear at different
locations,  minimum of 2 and a maximum of 7 in each frame. As Fig. 2
shows, each CAV is equipped with 4 cameras that can cover 360◦ view
together, a 64-channel LiDAR, and GPS/IMU sensors. The sensor data is
streamed at 20 Hz and recorded at 10 Hz. A more detailed description of
the sensor configurations is depicted in Table I. Culver City Digital
Town. To incorporate scenarios that can (a) better imitate real-world
challenging driving environments and evaluate models’ domain adaptation
capability, we fur- ther gather several scenes imitating realistic
configurations. An automated vehicle equipped with a 32-channel LiDAR
and two cameras is sent out to Culver City during rush hour to collect
sensing data. Then, we populate the road topology of digital town via
RoadRunner \[26\], select buildings based on agreement with collected
data, and then spawn cars mimicking the real-world traffic flow with the
support of OpenCDA. We collect 4 scenes in Culver City with around (b)
600 frames in total (See Fig. 4). These scenes will be used for Fig. 4:
A caparison between the real Culver City and its validation of models
trained with simulated datasets purely digital town. (a) The RGB image
and LiDAR point cloud generated in CARLA. Future addition of data from
real captured by our vehicle in Culver City. (b) The corresponding
environments is planned and can be added to the model frame in the
digital town. The road topology, building layout, training set. and
traffic distribution are similar to reality. Data Size. Overall, 11,464
frames (i.e. time steps) of LiDAR point clouds (see Fig. 1) and RGB
images (see Fig. 3) are collected with a total file size of 249.4 GB.
Moreover, we leading to spatial and temporal inconsistency. \[15\]
utilizes a also generate Bird Eye View (BEV) maps for each CAV
high-fidelity LiDAR simulator \[23\] to generate a large-scale in each
frame to facilitate the fundamental BEV semantic V2V dataset. However,
neither the LiDAR simulator nor the segmentation task. dataset is
publicly available. Recently, several works \[17\], Downstream Tasks. By
default, OPV2V supports coop- \[24\] manage to evaluate their V2V
perception algorithms on erative 3D object detection, BEV semantic
segmentation, the CARLA simulator, but the collected data has a limited
tracking, and prediction either employing camera rigs or size and is
restricted to a small area with a fixed number LiDAR sensors. To enable
users to extend the initial data, of connected vehicles. More
importantly, their dataset is we also provide a driving log replay tool2
. along with not released and difficult to reproduce the identical data
the dataset. By utilizing this tool, users can define their based on
their generation approach. T&J dataset \[11\], \[22\] own tasks (e.g.,
depth estimation, sensor fusion) and set up utilizes two golf carts
equipped with 16-channel LiDAR for additional sensors (e.g., depth
camera) without changing any data collection. Nevertheless, the released
version only has original driving events. Note that in this paper, we
only report 100 frames without ground truth labels and only covers a the
benchmark results on 3D Lidar-based object detection. restricted number
of road types. A comparison to existing dataset is provided in Table II.
B. Data Analysis As Table III depicts, six distinct categories of road
types III. OPV2V DATASET are included in our dataset for simulating the
most common A. Data Collection driving scenarios in real life. To
minimize data redundancy, Simulator Selection. CARLA is selected as our
simulator we attempt to avoid overlong clips and assign the ego to
collect the dataset, but CARLA itself doesn’t have V2V vehicles short
travels with an average length of 16.4 seconds, communication and
cooperative driving functionalities by dissimilar locations, and
divergent maneuvers for each sce- default. Hence, we employ OpenCDA
\[18\], a co-simulation nario. We also allocate the gathered 73 scenes
with diverse tool integrated with CARLA and SUMO \[25\], to generate
traffic and CAV configurations to enlarge dataset variance. our dataset1
. It is featured with easy control of multiple Fig. 5 and Fig. 6 reveal
the statistics of the 3D bounding CAVs, embedded vehicle network
communication protocols, box annotations in our dataset. Generally, the
cars around the and more convenient and realistic traffic management.
ego vehicle are well-distributed with divergent orientations Sensor
Configuration. The majority of our data comes and bounding box sizes.
This distribution is in agreement from eight default towns provided by
CARLA. Our dataset with the data collection process where the object
positions has on average approximately 3 connected vehicles with a are
randomly selected around CAVs and vehicle models are also arbitrarily
chosen. As shown in Fig. 5, unlike 1 Codes for generating our dataset
have been recently released in
https://github.com/ucla-mobility/OpenCDA/tree/feature/data collection 2
The tool can be found here. TABLE II: Dataset comparison. († ) The
number is reported based on data used during their experiment. (†† )
Single LiDAR resolution’s data is counted. (‡ ) Ground truth data is not
released in the T&J dataset and it only has 100 frames and LiDAR data.
(-) means that the number is not reported in the paper and can’t be
found in open dataset. (∗ ) means the data has the format mean±std. GT
Dataset CAV cities Code Open Dataset Reproducibility& Dataset frames 3D
boxes Size range Extensibility V2V-Sim \[15\] 51,200 - - 10 ± 7∗ \>1
\[17\] 1,310† - - 3, 5 1 \[24\] 6,000†† - - 2 1 X T&J \[11\], \[22\]
100‡ 0‡ 183.7MB 2 1 X X OPV2V 11,464 232,913 249.4GB 2.89 ± 1.06∗ 9 X X
X

TABLE III: Summary of OPV2V dataset statistics. Traffic density means
the number of vehicles spawned around the ego vehicle within a 140m
radius and aggressiveness represents the probability of a vehicle
operating aggressive overtakes. Length(s) CAV number Traffic density
Traffic Speed(km/h) CAV speed(km/h) Aggressiveness Road Type
Percentage(%) mean/std mean/std mean/std mean/std mean/std mean/std
4-way Intersection 24.5 12.5/4.2 2.69/0.67 29.6/26.1 19.3/8.8 21.3/10.2
0.09/0.30 T Intersection 24.1 14.3/12.8 2.55/1.3 27.9/18.65 26.3/7.5
26.2/10.0 0.11/0.32 Straight Segment 20.7 20.2/12.7 3.54/1.21 38.0/36.3
45.7/14.8 54.3/20.1 0.82/0.40 Curvy Segment 23.3 17.8/6.8 2.86/0.95
19.1/9.2 45.8/15.1 51.6/19.2 0.50/0.51 Midblock 4.7 10.0/1.3 3.00/1.22
21.8/8.2 45.1/8.3 50.7/11.5 0.20/0.44 Entrance Ramp 2.7 9.3/0.9
2.67/0.57 20.3/2.8 54.8/1.7 66.7/4.8 0.67/0.57 Overall 100 16.4/9.1
2.89/1.06 26.5/17.2 33.1/15.8 37.5/21.0 0.34/0.47

Fig. 5: Polar density map in log scale for ground truth Fig. 6: Left:
Number of points in log scale within the ground bounding boxes. The
polar and radial axes indicate the truth bounding boxes with respect to
radial distance from ego angle and distance (in meters) of the bounding
boxes with vehicles. Right: Bounding box size distributions. respect to
the ego vehicle. The color indicates the number of bounding boxes (log
scale) in the bin. The darker color means a larger number of boxes in
the bin. between vehicles), a method that can pay attention to impor-
tant observations while ignoring disrupted ones is crucial for robust
detection. Therefore, we propose an Attentive Inter- the dataset for the
single self-driving car, our dataset still mediate Fusion pipeline to
capture the interactions between has a large portion of objects in view
with distance ≥ features of neighboring connected vehicles, helping the
net- 100m, given that the ground truth boxes are defined with work
attend to key observations. The proposed Attentive respect to the
aggregated lidar points from all CAVs. As Intermediate Fusion pipeline
consists of 6 modules: Metadata displayed in Fig. 6, although a single
vehicle’s LiDAR sharing, Feature Extraction, Compression, Feature
sharing, points for distant objects are especially sparse, other CAVs
Attentive Fusion, and Prediction. The overall architecture is are able
to provide compensations to remarkably boost the shown in Fig. 7. The
proposed pipeline is flexible and can be LiDAR points density. This
demonstrates the capability of easily integrated with existing Deep
Learning-based LiDAR V2V technology to drastically increase perception
range and detectors (see Table IV). provide compensation for occlusions.
Metadata Sharing and Feature Extraction: We first broad- cast each CAVs’
relative pose and extrinsics to build a spatial IV. ATTENTIVE I
NTERMEDIATE F USION P IPELINE graph where each node is a CAV within the
communication As sensor observations from different connected vehicles
range and each edge represents a communication channel potentially carry
various noise levels (e.g., due to distance between a pair of nodes.
After constructing the graph, an Fig. 7: The architecture of Attentive
Intermediate Fusion pipeline. Our model consists of 6 parts: 1) Metadata
Sharing: build connection graph and broadcast locations among
neighboring CAVs. 2) Feature Extraction: extract features based on each
detector’s backbone. 3) Compression (optional): use Encoder-Decoder to
compress/decompress features. 4) Feature sharing: share (compressed)
features with connected vehicles. 5) Attentive Fusion: leverage
self-attention to learn interactions among features in the same spatial
location. 6) Prediction Header: generate final object predictions.

ego vehicle will be selected within the group.3 And all the neighboring
CAVs will project their own point clouds to the ego vehicle’s LiDAR
frame and extract features based on the projected point clouds. The
feature extractor here can be the backbones of existing 3D object
detectors. Compression and Feature sharing: An essential factor in V2V
communication is the hardware restriction on trans- mission bandwidth.
The transmission of the original high- dimensional feature maps usually
requires large bandwidth and hence compression is necessary. One key
advantage of intermediate fusion over sharing raw point clouds is the
marginal accuracy loss after compression \[15\]. Here we deploy an
Encoder-Decoder architecture to compress the Fig. 8: The architecture of
PIXOR with Attentive Fusion. shared message. The Encoder is composed of
a series of 2D convolutions and max pooling, and the feature maps in the
bottleneck will broadcast to the ego vehicle. The Decoder V. E
XPERIMENTS that contains several deconvolution layers \[27\] on the ego-
vehicles’ side will recover the compressed information and A. Benchmark
models send it to the Attentive Fusion module. We implement four
state-of-the-art LiDAR-based 3D ob- Attentive Fusion: Self-attention
models \[28\] are adopted ject detectors on our dataset and integrate
these detectors to fuse those decompressed features. Each feature vector
with three different fusion strategies i.e., early fusion, late
(green/blue circles shown in Fig.7) within the same feature fusion, and
intermediate fusion. We also investigate the map corresponds to certain
spatial areas in the original model performance under a single-vehicle
setting, named no point clouds. Thus, simply flattening the feature maps
and fusion, which neglects V2V communication. Therefore, in calculating
the weighted sum of features will break spatial total 16 models will be
evaluated in the benchmark. All the correlations. Instead, we construct
a local graph for each models are implemented in a unified code
framework, and feature vector in the feature map, where edges are built
for our code and develop tutorial can be found in the project feature
vectors in the same spatial locations from disparate website. connected
vehicles. One such local graph is shown in Fig.7 Selected 3D Object
Detectors: We pick SECOND \[29\], and self-attention will operate on the
graph to reason the VoxelNet \[4\], PIXOR \[30\], and PointPillar \[5\]
as our 3D interactions for better capturing the representative features.
LiDAR detectors for benchmarking analysis. Prediction Header: The fused
features will be fed to the Early fusion baseline: All the LiDAR point
clouds will prediction header to generate bounding box proposals and be
projected into ego-vehicles’ coordinate frame, based on associated
confidence scores. the pose information shared among CAVs, and then the
ego 3 During training, a random CAV within the group is selected as ego
vehicle will aggregate all received point clouds and feed them vehicle
while in the inference, the ego vehicle is fixed for a fair comparison.
to the detector. TABLE IV: Object detection results on Default CARLA
Towns and digital Culver City. Default Culver Method AP@IoU AP@IoU 0.5
0.7 0.5 0.7 No Fusion 0.635 0.406 0.505 0.290 Late Fusion 0.769 0.578
0.622 0.360 PIXOR Early Fusion 0.810 0.678 0.734 0.558 Intermediate
Fusion 0.815 0.687 0.716 0.549 No Fusion 0.679 0.602 0.557 0.471 Late
Fusion 0.858 0.781 0.799 0.668 PointPillar Early Fusion 0.891 0.800
0.829 0.696 Intermediate Fusion 0.908 0.815 0.854 0.735 No Fusion 0.713
0.604 0.646 0.517 Late Fusion 0.846 0.775 0.808 0.682 SECOND Early
Fusion 0.877 0.813 0.821 0.738 Fig. 9: Average Precision at IoU=0.7 with
respect to CAV Intermediate Fusion 0.893 0.826 0.875 0.760 number. No
Fusion 0.688 0.526 0.605 0.431 Late Fusion 0.801 0.738 0.722 0.588
VoxelNet Early Fusion 0.852 0.758 0.815 0.677 Intermediate Fusion 0.906
0.864 0.854 0.775

Late fusion baseline: Each CAV will predict the bounding boxes with
confidence scores independently and broadcast these outputs to the ego
vehicle. Non-maximum suppres- sion (NMS) will be applied to these
proposals afterwards to generate the final object predictions.
Intermediate fusion: The Attentive Fusion pipeline is flex- ible and can
be easily generalized to other object detection networks. To evaluate
the proposed pipeline, we only need to add the Compression, Sharing, and
Attention (CSA) Fig. 10: Average Precision at IoU=0.7 with respect to
data module to the existing network architecture. Since 4 different size
in log scale based on VoxelNet detector. The number× detectors add CSA
modules in a similar way, here we only refers to the compression rate.
show the architecture of intermediate fusion with the PIXOR model as
Fig. 8 displays. Three CSA modules are added at the 2D backbone of PIXOR
to aggregate multi-scale features and it takes us 14 days to finish all
training on 4 RTX 3090 while all other parts of the network remain the
same. GPUs.

B. Metrics D. Benchmark Analysis We select a fixed vehicle as the ego
vehicle among all Table IV depicts the performance of the selected four
spawned CAVs for each scenario in the test and validation LiDAR
detectors combined with different fusion strategies. set. Detection
performance is evaluated near the ego vehicle All fusion methods achieve
≥10% AP gains at IoU 0.7 over in a range of x ∈ \[−140, 140\]m, y ∈
\[−40, 40\]m. Following no fusion counterparts for both default CARLA
towns and \[15\], we set the broadcast range among CAVs to be 70 me-
Culver City, showing the advantage of aggregating informa- ters. Sensing
messages outside of this communication range tion from all CAVs for V2V
perception. Generally, because will be ignored by the ego vehicle.
Average Precisions (AP) of the capability of preserving more sensing
measurements at Intersection-over-Union (IoU) threshold of both 0.5 and
and visual cues, early fusion methods outperform late fusion 0.7 are
adopted to assess different models. Since PIXOR methods. Except for
PIXOR at Culver City, intermediate ignores the z coordinates of the
bounding box, we compute fusion achieves the best performance on both
testing sets IoU only on x-y plane to make the comparison fair. For the
compared with all other methods. We argue that the AP gains evaluation
targets, we include vehicles that are hit by at least over early fusion
originate from the mechanism of the self- one LiDAR point from any
connected vehicle. attention module, which can effectively capture the
inherent correlation between each CAV’s perception information. It is C.
Experiment Details also worth noting that the prediction results for
Culver City The train/validation/test splits are 6764/1981/2719 frames.
are generally inferior to CARLA towns. Such a phenomenon The testing
frames contain all road types and are further is expected as the traffic
pattern in Culver City is more split into two parts–CARLA default maps
and Culver City similar to real life, which causes a domain gap with the
digital town. For each frame, we assure that the minimum training data.
Furthermore, we collect the Culver City data and maximum numbers of CAVs
are 2 and 7 respectively. in a busy hour under a very congested driving
environment, We use Adam Optizer \[31\] and early stop to train all
models, which leads to vastly severe occlusions and makes the detection
task very challenging. \[3\] C. R. Qi, H. Su, K. Mo, and L. J. Guibas,
“Pointnet: Deep learning on point sets for 3d classification and
segmentation,” in Proceedings E. Effect of CAV Quantity of the IEEE
conference on computer vision and pattern recognition, 2017,
pp. 652–660. We explore the detection performance as affected by the
\[4\] Y. Zhou and O. Tuzel, “Voxelnet: End-to-end learning for point
cloud number of CAVs in a complex intersection scenario where based 3d
object detection,” 06 2018, pp. 4490–4499. 150 vehicles are spawned in
the surrounding area. A portion \[5\] A. Lang, S. Vora, H. Caesar, L.
Zhou, J. Yang, and O. Beijbom, “Pointpillars: Fast encoders for object
detection from point clouds,” of them will be transformed into CAVs that
can share 06 2019, pp. 12 689–12 697. information. We gradually increase
the number of the CAVs \[6\] S. Shi, X. Wang, and H. Li, “Pointrcnn: 3d
object proposal generation up to 7 and apply VoxelNet with different
fusion methods for and detection from point cloud,” in The IEEE
Conference on Computer Vision and Pattern Recognition (CVPR), June 2019.
object detection. As shown in Fig. 9, the AP has a positive \[7\] S.
Shi, C. Guo, L. Jiang, Z. Wang, J. Shi, X. Wang, and H. Li, “Pv-
correlation with the number of CAVs. However, when the rcnn: Point-voxel
feature set abstraction for 3d object detection,” 2020 quantity reaches
4, the increasing rate becomes lower. This IEEE/CVF Conference on
Computer Vision and Pattern Recognition (CVPR), pp. 10 526–10 535, 2020.
can be due to the fact that the CAVs are distributed on \[8\] M. Liang,
B. Yang, Y. Chen, R. Hu, and R. Urtasun, “Multi-task multi- different
sides of the intersection and four of them can sensor fusion for 3d
object detection,” 06 2019, pp. 7337–7345. already provide enough
viewpoints to cover most of the blind \[9\] M. Liang, B. Yang, S. Wang,
and R. Urtasun, “Deep continuous fusion spots. Additional enhancements
with 5 or more vehicles for multi-sensor 3d object detection,” in
Computer Vision – ECCV 2018, V. Ferrari, M. Hebert, C. Sminchisescu, and
Y. Weiss, Eds. come from denser measurements on the same object. Cham:
Springer International Publishing, 2018, pp. 663–678. \[10\] T.
Lochrane, L. Dailey, and C. Tucker, “Carma℠: Driving F. Effect of
Compression Rates innovation,” Public Roads, vol. 83, no. 4, 2020.
\[Online\]. Available: https://its.dot.gov/cda/ Fig. 10 exhibits the
data size needed for a single trans- \[11\] Q. Chen, S. Tang, Q. Yang,
and S. Fu, “Cooper: Cooperative mission between a pair of vehicles and
corresponding AP perception for connected autonomous vehicles based on
3d point for all fusion methods on the testing set in CARLA towns.
clouds,” in 2019 IEEE 39th International Conference on Distributed We
pick VoxelNet for all fusion methods here and simulate Computing Systems
(ICDCS). Los Alamitos, CA, USA: IEEE Computer Society, jul 2019,
pp. 514–524. \[Online\]. Available: distinct compression rates by
modifying the number of layers
https://doi.ieeecomputersociety.org/10.1109/ICDCS.2019.00058 in
Encoder-Decoder. By applying a straightforward Encoder- \[12\] H.
Caesar, V. Bankiti, A. H. Lang, S. Vora, V. E. Liong, Q. Xu, A. Kr-
Decoder architecture to squeeze the data, the Attentive ishnan, Y. Pan,
G. Baldan, and O. Beijbom, “nuscenes: A multimodal dataset for
autonomous driving,” arXiv preprint arXiv:1903.11027, Intermediate
Fusion obtains an outstanding trade-off between 2019. the accuracy and
bandwidth. Even with a 4096x compression \[13\] P. Sun, H. Kretzschmar,
X. Dotiwalla, A. Chouard, V. Patnaik, P. Tsui, rate, the performance
still just drop marginally (around 3%) J. Guo, Y. Zhou, Y. Chai, B.
Caine, et al., “Scalability in perception for autonomous driving: Waymo
open dataset,” in Proceedings of the and surpass the early fusion and
late fusion. Based on the IEEE/CVF Conference on Computer Vision and
Pattern Recognition, V2V communication protocol \[32\], data
broadcasting can 2020, pp. 2446–2454. achieve 27 Mbps at the range of
300 m. This represents \[14\] A. Geiger, P. Lenz, C. Stiller, and R.
Urtasun, “Vision meets robotics: The kitti dataset,” International
Journal of Robotics Research (IJRR), that the time delay to deliver the
message with a 4096x 2013. compression rate is only about 5 ms. \[15\]
T.-H. Wang, S. Manivasagam, M. Liang, B. Yang, W. Zeng, J. Tu, and R.
Urtasun, “V2vnet: Vehicle-to-vehicle communication for joint VI.
CONCLUSIONS perception and prediction,” in ECCV, 2020. \[16\] Z. Y.
Rawashdeh and Z. Wang, “Collaborative automated driving: A In this
paper, we present the first open dataset and machine learning-based
method to enhance the accuracy of shared benchmark fusion strategies for
V2V perception. We further information,” in 2018 21st International
Conference on Intelligent come up with an Attentive Intermediate Fusion
pipeline, Transportation Systems (ITSC), 2018, pp. 3961–3966. \[17\] Z.
Zhang, S. Wang, Y. Hong, L. Zhou, and Q. Hao, “Distributed and the
experiments show that the proposed approach can dynamic map fusion via
federated learning for intelligent networked outperform all other fusion
methods and achieve state-of- vehicles,” ArXiv, vol. abs/2103.03786,
2021. the-art performance even under large compression rates. \[18\] R.
Xu, Y. Guo, X. Han, X. Xia, H. Xiang, and J. Ma, “Opencda: In the
future, we plan to extend the dataset with more An open cooperative
driving automation framework integrated with co-simulation,” in 2021
IEEE Intelligent Transportation Systems Con- tasks as well as sensors
suites and investigate more multi- ference (ITSC), 2021. modal sensor
fusion methods in the V2V and Vehicle-to- \[19\] A. Dosovitskiy, G. Ros,
F. Codevilla, A. Lopez, and V. Koltun, infrastructure (V2I) setting. We
hope our open-source efforts “CARLA: An open urban driving simulator,”
in Proceedings of the 1st Annual Conference on Robot Learning, 2017,
pp. 1–16. can make a step forward for the standardizing process of
\[20\] A. Rauch, F. Klanner, R. Rasshofer, and K. Dietmayer, “Car2x- the
V2V perception and encourage more researchers to based perception in a
high-level fusion architecture for cooperative investigate this new
direction. perception systems,” in 2012 IEEE Intelligent Vehicles
Symposium, 2012, pp. 270–275. R EFERENCES \[21\] Z. Y. Rawashdeh and Z.
Wang, “Collaborative automated driving: A machine learning-based method
to enhance the accuracy of shared \[1\] C. Liu, L.-C. Chen, F. Schroff,
H. Adam, W. Hua, A. L. Yuille, information,” in 2018 21st International
Conference on Intelligent and L. Fei-Fei, “Auto-deeplab: Hierarchical
neural architecture search Transportation Systems (ITSC), 2018,
pp. 3961–3966. for semantic image segmentation,” in Proceedings of the
IEEE/CVF \[22\] Q. Chen, X. Ma, S. Tang, J. Guo, Q. Yang, and S. Fu,
“F-cooper: Conference on Computer Vision and Pattern Recognition, 2019,
pp. Feature based cooperative perception for autonomous vehicle edge
82–92. computing system using 3d point clouds,” in Proceedings of the
4th \[2\] K. He, X. Zhang, S. Ren, and J. Sun, “Deep residual learning
for image ACM/IEEE Symposium on Edge Computing, ser. SEC ’19. New
recognition,” in Proceedings of the IEEE conference on computer York,
NY, USA: Association for Computing Machinery, 2019, p. vision and
pattern recognition, 2016, pp. 770–778. 88–100. \[Online\]. Available:
https://doi.org/10.1145/3318216.3363300 \[23\] S. Manivasagam, S. Wang,
K. Wong, W. Zeng, M. Sazanovich, S. Tan, B. Yang, W.-C. Ma, and R.
Urtasun, “Lidarsim: Realistic lidar simulation by leveraging the real
world,” 06 2020, pp. 11 164–11 173. \[24\] E. E. Marvasti, A. Raftari,
A. E. Marvasti, Y. P. Fallah, R. Guo, and H. Lu, “Cooperative lidar
object detection via feature sharing in deep networks,” in 2020 IEEE
92nd Vehicular Technology Conference (VTC2020-Fall). IEEE, 2020,
pp. 1–7. \[25\] C. Olaverri-Monreal, J. Errea-Moreno, A. Dı́az-Álvarez,
C. Biurrun- Quel, L. Serrano-Arriezu, and M. Kuba, “Connection of the
sumo microscopic traffic simulator and the unity 3d game engine to
evaluate v2x communication-based systems,” Sensors (Basel, Switzerland),
vol. 18, 2018. \[26\] “Roadrunner: Design 3d scenes for automated
driving simulation.” \[Online\]. Available:
https://www.1stvision.com/cameras/ models/Allied-Vision \[27\] H. Noh,
S. Hong, and B. Han, “Learning deconvolution network for semantic
segmentation,” in Proceedings of the IEEE international conference on
computer vision, 2015, pp. 1520–1528. \[28\] A. Vaswani, N. Shazeer, N.
Parmar, J. Uszkoreit, L. Jones, A. N. Gomez, Ł. Kaiser, and I.
Polosukhin, “Attention is all you need,” in Advances in neural
information processing systems, 2017, pp. 5998– 6008. \[29\] Y. Yan, Y.
Mao, and B. Li, “Second: Sparsely embedded convolutional detection,”
Sensors (Basel, Switzerland), vol. 18, 2018. \[30\] B. Yang, W. Luo, and
R. Urtasun, “Pixor: Real-time 3d object de- tection from point clouds,”
2018 IEEE/CVF Conference on Computer Vision and Pattern Recognition,
pp. 7652–7660, 2018. \[31\] D. P. Kingma and J. Ba, “Adam: A method for
stochastic optimiza- tion,” arXiv preprint arXiv:1412.6980, 2014. \[32\]
F. Arena and G. Pau, “An overview of vehicular communications,” Future
Internet, vol. 11, no. 2, p. 27, 2019. 
