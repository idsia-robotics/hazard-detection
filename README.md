# Anomaly Detection for Robots
This is the main repository for the dataset Dataset for Sensing Anomalies as Potential Hazards in Mobile Robots.
The dataset can be find on Zenodo at https://zenodo.org/record/5520933

## Papers
<!-- The relative video is available at TODO -->
### <em>Sensing Anomalies as Potential Hazards: Datasets and Benchmarks</em>
    We consider the problem of detecting, in the visual sensing data stream of an autonomous mobile robot, semantic patterns that are unusual (i.e., anomalous) with respect to the robot's previous experience in similar environments.  These anomalies might indicate unforeseen hazards and, in scenarios where failure is costly, can be used to trigger an avoidance behavior.  We contribute three novel image-based datasets acquired in robot exploration scenarios, comprising a total of more than 200k labeled frames, spanning various types of anomalies.  On these datasets, we study the performance of an anomaly detection approach based on autoencoders operating at different scales.

To appear at the [23rd TAROS Conference](https://ukaeaevents.com/23rd-taros/)

DOI: TODO

ArXiv: TODO

### <em>An Outlier Exposure Approach to Improve Visual Anomaly Detection Performance for Mobile Robots.</em>
    We consider the problem of building visual anomaly detection systems for mobile robots. Standard anomaly detection models are trained using large datasets composed only of non-anomalous data. However, in robotics applications, it is often the case that (potentially very few) examples of anomalies are available. We tackle the problem of exploiting these data to improve the performance of a Real-NVP anomaly detection model, by minimizing, jointly with the Real-NVP loss, an auxiliary outlier exposure margin loss. We perform quantitative experiments on a novel dataset (which we publish as supplementary material) designed for anomaly detection in an indoor patrolling scenario. On a disjoint test set, our approach outperforms alternatives and shows that exposing even a small number of anomalous frames yields significant performance improvements.

Pubblished in [Robotics and Automation Letters](https://www.ieee-ras.org/publications/ra-l) 
<!-- volume .... year ... -->

DOI: https://doi.org/10.1109/LRA.2022.3192794

ArXiv: TODO

## Codes
Under `./code` you can find the code used for the TAROS paper under `./code/OLD_CODE` and the code for RAL paper under `./code/Latest`.

We use python 3.8 and the requirements in `./code/Latest/requirementes.txt`; follow the README.md under the `./code/Latest` to install and run the models.


# Description
The dataset is composed of three different scenarios:
- Tunnel
- Factory
- Corridors

In the version of the dataset of the TAROS paper the Corridors scenario has 52'607 samples and 8 anomalies. 
In the RAL paper we extended this scenario up to 132'838 frames and 16 anomalies.

Across the three scenarios we recorded various normal situations and numerous anomalies.
The anomalies are the following:
<!-- AN1 - image - description -->

### Fundings
This work was supported as a part of NCCR Robotics, a National Centre of Competence in Research, funded by the Swiss National Science Foundation (grant number 51NF40\_185543) and by the European Commission through the Horizon 2020 project 1-SWARM, grant ID 871743.

# How to cite
If you use this dataset please cite it using the following bib

    @ARTICLE{mantegazza2022outlier,
        author={Mantegazza, Dario and Giusti, Alessandro and Gambardella, Luca Maria and Guzzi, Jérôme}, 
        journal={IEEE Robotics and Automation Letters},
        title={An Outlier Exposure Approach to Improve Visual Anomaly Detection Performance for Mobile Robots.},
        year={2022}, 
        volume={},
        number={}, 
        pages={1-8}, 
        doi={10.1109/LRA.2022.3192794}
      }
# Anomalies Examples
### Tunnel Anomalies
Normal - Empty underground man made tunnel

<img src="images/tunnel/normal1.jpg" alt="tun_normal" width="356"/>

Wet - Water condensation on the tunnel walls and cieling

<img src="images/tunnel/wet1.jpg" alt="wet" width="356"/>

Root - Roots coming down from the cieling and walls

<img src="images/tunnel/root1.jpg" alt="root" width="356"/>

Dust - Dust moved by the drone 

<img src="images/tunnel/dust1.jpg" alt="dust" width="356"/>

### Factory Anomalies
Normal - Empty factory facility

<img src="images/factory/normal1.jpg" alt="fact_normal" width="356"/>

Mist - Mist coming from a smoke machine

<img src="images/factory/mist1.jpg" alt="mist" width="356"/>

Tape - Signaling tape stretched across the facility

<img src="images/factory/tape1.jpg" alt="tape" width="356"/>

### Corridors Anomalies
Normal - Empty university corridors (on different floors)

<img src="images/corridor/normal1.jpg" alt="corridor_normal" width="356"/>
<img src="images/corridor/normal2.jpg" alt="corridor_normal2" width="356"/>
<img src="images/corridor/normal3.jpg" alt="corridor_normal3" width="356"/>

Box - Cardboard boxes placed infront/near of the robot

<img src="images/corridor/box.jpg" alt="box" width="356"/>

Cable - Various cables layed on the floor around and infront of the robot

<img src="images/corridor/cable.jpg" alt="cable" width="356"/>

Debris - Various debris 

<img src="images/corridor/debris.jpg" alt="debris" width="356"/>

Defects - Defects of the robot

<img src="images/corridor/defects.jpg" alt="defects" width="356"/>

Door - Open doors where doors should be closed

<img src="images/corridor/door.jpg" alt="door" width="356"/>

Human - Human presence

<img src="images/corridor/human.jpg" alt="human" width="356"/>

Clutter - Chairs, tables and furnitures moved around the corridor

<img src="images/corridor/clutter.jpg" alt="clutter" width="356"/>

Foam - Foam placed on the floor

<img src="images/corridor/foam.jpg" alt="foam" width="356"/>

Sawdust - Sawdust placed on the floor

<img src="images/corridor/sawdust.jpg" alt="sawdust" width="356"/>

Cellophane - Cellophane foil streched between walls

<img src="images/corridor/cellophane.jpg" alt="cellophane" width="356"/>

Floor - Fake flooring different than original floor

<img src="images/corridor/floor.jpg" alt="floor" width="356"/>

Screws - Small screwes and bolts placed infront of the robot

<img src="images/corridor/screws.jpg" alt="screws" width="356"/>

Water - Water puddle infront of robot

<img src="images/corridor/water.jpg" alt="water" width="356"/>

Cones - Multiple orange cones placed in the corridor

<img src="images/corridor/cones.jpg" alt="cones" width="356"/>

Hanging cables - Cables hanging from the cieling

<img src="images/corridor/hanging_cable.jpg" alt="hanghingcables" width="356"/>

