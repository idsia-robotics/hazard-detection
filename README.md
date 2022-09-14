# Hazards&Robots: A Dataset for Visual Anomaly Detection in Robotics
This is the main repository for the dataset Dataset for Sensing Anomalies as Potential Hazards in Mobile Robots.
The dataset can be find on Zenodo at https://zenodo.org/record/7074958 (RAL extension) and https://zenodo.org/record/7035788 (TAROS version)

## Papers
<!-- The relative video is available at TODO -->
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

To appear at the [23rd TAROS 2022 Conference](https://ukaeaevents.com/23rd-taros/)

DOI: https://doi.org/10.1007/978-3-031-15908-4_17

ArXiv: Coming Soon

### <em>An Outlier Exposure Approach to Improve Visual Anomaly Detection Performance for Mobile Robots.</em>
Dario Mantegazza, Alessandro Giusti, Luca Maria Gambardella and Jerome Guzzi

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

Published in [Robotics and Automation Letters](https://www.ieee-ras.org/publications/ra-l) year 2022
<!-- volume .... year ... -->

DOI: https://doi.org/10.1109/LRA.2022.3192794

ArXiv: Coming Soon

## Codes
Under `./code` you can find the code used for the <em>TAROS</em> paper under `./code/OLD_CODE` and the code for <em>RAL</em> paper under `./code/Latest`.

We use python 3.8 and the requirements in `./code/Latest/requirements.txt`; follow the README.md under the `./code/Latest` to install and run the models.


# Description
The dataset is composed of three different scenarios:
- Tunnel
- Factory
- Corridors

In the version of the dataset of the <em>TAROS</em> paper the Corridors scenario has 52'607 samples and 8 anomalies. 
In the <em>RAL</em> paper we extended this scenario up to 132'838 frames and 16 anomalies.
<figure>
<img src="images/dataset_examplev4.png" alt="RAL_paper_anomalies" style="background-color:white;"/>
<p align = "center">Examples of samples of the <em>Corridors</em> scenario from the <em>RAL</em> paper </p>
</figure>

### Funding
This work was supported as a part of NCCR Robotics, a National Centre of Competence in Research, funded by the Swiss National Science Foundation (grant number 51NF40\_185543) and by the European Commission through the Horizon 2020 project 1-SWARM, grant ID 871743.
# Contact

- If you have questions please contact us via email dario (dot) mantegazza (at) idsia (dot) ch
- If you have problems with running our code, please open an issue and we will do our best to answer you
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
