# CNN and Lucas Kanade Camera Tracking Pipeline

<p align="left">
  <img src="Resources/IPP.gif" alt="Pipeline Demo" width="800">
</p>

### Team Members
Brian Bartley, bmb187@pitt.edu <br>
Colton Frankenberry, cof29@pitt.edu <br>
Nick LaVine, nal124@pitt.edu <br>
Elizabeth Novikova, eln46@pitt.edu <br>
### Description
Developing a video camera object tracking algorithm to track an object and keep it in frame
### Milestones
1) Train CNN to detect and classify objects in camera feed
2) Draw bounding boxes around detected objects
3) For each bounding box, extract key feature points (e.g. corners and edges)
4) Use Lucas-Kanade optical flow to track key feature points from frame to frame
5) Update bounding box position based on tracked feature movement
6) Periodically (every so many frames) rerun CNN to correct drift from Lucas-Kanade
7) Loop program
8) (Time allowing) Develop camera hardware
