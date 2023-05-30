# Hazards&Robots: A Dataset for Visual Anomaly Detection in Robotics
This is the main repository for *Hazards&Robots: A Dataset for Visual Anomaly Detection in Robotics* and relative papers.

The dataset can be find on Zenodo:
- (v1) TAROS version : https://zenodo.org/record/7035788
- (v2) RAL extension : https://zenodo.org/record/7074958
- ### (v3) Data in Brief final version: https://zenodo.org/record/7859211


## Papers

### <em>Sensing Anomalies as Potential Hazards: Datasets and Benchmarks</em>
Dario Mantegazza, Carlos Redondo, Fran Espada, Luca M. Gambardella, Alessandro Giusti and Jerome Guzzi

    We consider the problem of detecting, in the visual sensing data stream of an 
    autonomous mobile robot, semantic patterns that are unusual (i.e., anomalous) with
    respect to the robot's previous experience in similar environments.  These 
    anomalies might indicate unforeseen hazards and, in scenarios where failure is 
    costly, can be used to trigger an avoidance behavior.  We contribute three novel 
    image-based datasets acquired in robot exploration scenarios, comprising a total
    of more than 200k labeled frames, spanning various types of anomalies.  On these 
    datasets, we study the performance of an anomaly detection approach based on 
    autoencoders operating at different scales.

In the Proceedings of [23rd TAROS 2022 Conference](https://ukaeaevents.com/23rd-taros/)

DOI: https://doi.org/10.1007/978-3-031-15908-4_17

ArXiv: https://arxiv.org/abs/2110.14706

### <em>An Outlier Exposure Approach to Improve Visual Anomaly Detection Performance for Mobile Robots.</em>
Dario Mantegazza, Alessandro Giusti, Luca M. Gambardella and Jerome Guzzi

    We consider the problem of building visual anomaly detection systems for mobile 
    robots. Standard anomaly detection models are trained using large datasets composed 
    only of non-anomalous data. However, in robotics applications, it is often the case 
    that (potentially very few) examples of anomalies are available. We tackle the 
    problem of exploiting these data to improve the performance of a Real-NVP anomaly 
    detection model, by minimizing, jointly with the Real-NVP loss, an auxiliary outlier 
    exposure margin loss. We perform quantitative experiments on a novel dataset (which 
    we publish as supplementary material) designed for anomaly detection in an indoor 
    patrolling scenario. On a disjoint test set, our approach outperforms alternatives 
    and shows that exposing even a small number of anomalous frames yields significant 
    performance improvements.

Published in [Robotics and Automation Letters](https://www.ieee-ras.org/publications/ra-l) October 2022 Volume 7 Issue 4


DOI: https://doi.org/10.1109/LRA.2022.3192794

ArXiv: https://arxiv.org/abs/2209.09786

### <em>Hazards&Robots: A Dataset for Visual Anomaly Detection in Robotics</em>
Dario Mantegazza, Alind Xhyra, Luca M. Gambardella, Alessandro Giusti, Jérôme Guzzi

    We propose Hazards&Robots, a dataset for Visual Anomaly Detection in Robotics. 
    The dataset is composed of 324,408 RGB frames, and corresponding feature vectors; 
    it contains 145,470 normal frames and 178,938 anomalous ones categorized in 20 
    different anomaly classes. The dataset can be used to train and test current and 
    novel visual anomaly detection methods such as those based on deep learning vision models.
    The data is recorded with a DJI Robomaster S1 front facing camera. The ground robot, 
    controlled by a human operator, traverses university corridors. Considered anomalies 
    include presence of humans, unexpected objects on the floor, defects to the robot. 

DOI: https://doi.org/10.1016/j.dib.2023.109264

This is an Open-Access paper published in [Data in Brief](https://www.sciencedirect.com/journal/data-in-brief/vol/48/suppl/C) Volume 48, June 2023, Journal

### <em>NEXT</em>: Active Learning for Visual Anomaly Detection in Robotics? Stay Tuned ;)

## Codes
Under `./code` you can find the code used for the <em>TAROS</em> paper under `./code/OLD_CODE` and the code for <em>RAL</em> paper under `./code/Latest`; the code for the <em>Data in Brief</em> is available on the Zenodo repository.

We use python 3.8 and the requirements in `./code/Latest/requirements.txt`; follow the README.md under the `./code/Latest` to install and run the models.


# Description
The dataset is composed of three different scenarios:
- Tunnel
- Factory
- Corridors

The <em>TAROS</em>  version paper the Corridors scenario has 52'607 samples and 8 anomalies. 

In the <em>RAL</em> paper we extended this scenario up to 132'838 frames and 16 anomalies.

#### The latest <em>Data in Brief</em> release has 324'408 frames and 20 anomalies; for the first time we provide 512-dimension features vectors extracted with CLIP.

<figure>
<img src="images/samples.png" alt="DiB_paper_anomalies" style="background-color:white;"/>
<p align = "center">Examples of samples of the <em>Corridors</em> scenario from the <em>Data in Brief</em> paper </p>
</figure>

### Funding
This work was supported as a part of NCCR Robotics, a National Centre of Competence in Research, funded by the Swiss National Science Foundation (grant number 51NF40\_185543) and by the European Commission through the Horizon 2020 project 1-SWARM, grant ID 871743.
# Contact

- If you have questions please contact us via email dario (dot) mantegazza (at) idsia (dot) ch
- Questions or problems with the code? Just open an ISSUE, we will do our best to answer you as soon as possible :)
- For more information about us visit our site https://idsia-robotics.github.io/

# How to cite
If you use this dataset please cite it using the following bib

    @ARTICLE{mantegazza2022outlier,
        author={Mantegazza, Dario and Giusti, Alessandro and Gambardella, Luca Maria and Guzzi, Jérôme}, 
        journal={IEEE Robotics and Automation Letters},
        title={An Outlier Exposure Approach to Improve Visual Anomaly Detection Performance for Mobile Robots.},
        year={2022}, 
        volume={7},
        number={4}, 
        pages={11354-11361}, 
        doi={10.1109/LRA.2022.3192794}
      }

# Frames Examples
Across the three scenarios described before, we recorded various normal situations and numerous anomalies.
The anomalies are the following:
### Tunnel Anomalies
<details>
  <summary>Click for high resolution examples</summary>

<figure class="image">
<img src="images/tunnel/normal1.jpg" alt="tun_normal" width="512"/>
<p><b>Normal</b> - Empty underground man made tunnel</p>
</figure>
<p></p>
<figure>
<img src="images/tunnel/wet1.jpg" alt="wet" width="512"/>
<p><b>Wet</b> - Water condensation on the tunnel walls and ceiling</p>
</figure>
<p></p>

<figure>
<img src="images/tunnel/root1.jpg" alt="root" width="512"/>
<p><b>Root</b> - Roots coming down from the ceiling and walls</p>
</figure>
<p></p>

<figure>
<img src="images/tunnel/dust1.jpg" alt="dust" width="512"/>
<p><b>Dust</b> - Dust moved by the drone </p>
</figure>
<p></p>

</details>

### Factory Anomalies
<details>
  <summary>Click for high resolution examples</summary>

<figure>
<img src="images/factory/normal1.jpg" alt="fact_normal" width="512"/>
<p><b>Normal</b> - Empty factory facility</p>
</figure>
<p></p>


<figure>
<img src="images/factory/mist1.jpg" alt="mist" width="512"/>
<p><b>Mist</b> - Mist coming from a smoke machine</p>
</figure>
<p></p>

<figure>
<img src="images/factory/tape1.jpg" alt="tape" width="512"/>
<p><b>Tape</b> - Signaling tape stretched across the facility</p>
</figure>

</details>

### Corridors Anomalies
<details>
  <summary>Click for high resolution examples</summary>

<figure>
<img src="images/corridor/normal1.jpg" alt="corridor_normal" width="512"/>
<img src="images/corridor/normal2.jpg" alt="corridor_normal2" width="512"/>
<img src="images/corridor/normal3.jpg" alt="corridor_normal3" width="512"/>
<p><b>Normal</b> - Empty university corridors (on different floors)</p>
</figure>
<p></p>

<figure>
<img src="images/corridor/box.jpg" alt="box" width="512"/>
<p><b>Box</b> - Cardboard boxes placed in front/near of the robot</p>
</figure>
<p></p>

<figure>
<img src="images/corridor/cable.jpg" alt="cable" width="512"/>
<p><b>Cable</b> - Various cables layed on the floor around and in front of the robot</p>
</figure>
<p></p>

<figure>
<img src="images/corridor/debris.jpg" alt="debris" width="512"/>
<p><b>Debris</b> - Various debris </p>
</figure>
<p></p>

<figure>
<img src="images/corridor/defects.jpg" alt="defects" width="512"/>
<p><b>Defects</b> - Defects of the robot</p>
</figure>
<p></p>

<figure>
<img src="images/corridor/door.jpg" alt="door" width="512"/>
<p><b>Door</b> - Open doors where doors should be closed</p>
</figure>
<p></p>

<figure>
<img src="images/corridor/human.jpg" alt="human" width="512"/>
<p><b>Human</b> - Human presence</p>
</figure>
<p></p>

<figure>
<img src="images/corridor/clutter.jpg" alt="clutter" width="512"/>
<p><b>Clutter</b> - Chairs, tables and furniture moved around the corridor</p>
</figure>
<p></p>

<figure>
<img src="images/corridor/foam.jpg" alt="foam" width="512"/>
<p><b>Foam</b> - Foam placed on the floor</p>
</figure>
<p></p>

<figure>
<img src="images/corridor/sawdust.jpg" alt="sawdust" width="512"/>
<p><b>Sawdust</b> - Sawdust placed on the floor</p>
</figure>
<p></p>

<figure>
<img src="images/corridor/cellophane.jpg" alt="cellophane" width="512"/>
<p><b>Cellophane</b> - Cellophane foil stretched between walls</p>
</figure>
<p></p>

<figure>
<img src="images/corridor/floor.jpg" alt="floor" width="512"/>
<p><b>Floor</b> - Fake flooring different than original floor</p>
</figure>
<p></p>

<figure>
<img src="images/corridor/screws.jpg" alt="screws" width="512"/>
<p><b>Screws</b> - Small screws and bolts placed in front of the robot</p>
</figure>
<p></p>

<figure>
<img src="images/corridor/water.jpg" alt="water" width="512"/>
<p><b>Water</b> - Water puddle in front of robot</p>
</figure>
<p></p>

<figure>
<img src="images/corridor/cones.jpg" alt="cones" width="512"/>
<p><b>Cones</b> - Multiple orange cones placed in the corridor</p>
</figure>
<p></p>

<figure>
<img src="images/corridor/hanging_cable.jpg" alt="hanghingcables" width="512"/>
<p><b>Hanging cables</b> - Cables hanging from the ceiling</p>
</figure>

</details>
