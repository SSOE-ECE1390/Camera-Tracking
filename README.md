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
1) Use a pre-trained CNN to detect and classify objects in camera feed
2) Adjust parameters within CNN to fit the tracked object
3) Draw bounding boxes around detected objects
4) For each bounding box, pass into selected LK algorithim
5) Use Lucas-Kanade optical flow to track key feature points from frame to frame
6) Update bounding box position based on tracked feature movement
7) Periodically (every so many frames) rerun CNN to correct drift from Lucas-Kanade
8) Loop program
9) (Time allowing) Develop camera hardware

### Running Our Program
To run our program, run the file main_loop.py. Inside main, one can also adjust DO_FRAME_EXTRACING = False, in order to run on new video feed (must be set to true on first run with new feed). One can also switch between Forward Additive LK Translational and Inverse Compositional LK Affine by switching the line: curr_bounding = LK.LucasKanadeTracker(prev_img, curr_img, prev_bounding) to the respective function call within LK_Work.
