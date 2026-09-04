
ISSUES AND PROBLEMS
Below are listed all approaches, each with their limitations.


1. BASIC COLOR SEGMENTATION:
Example works great but anything other does not work. Fails already in the segmentation: Most of the images are just black.
- different background colors (hard threshold do not recognize holes anymore)
- different hue and lightning
-> contrast normalization
-> adaptive thresholds
-> smaller kernel


2. ADAPTIVE COLOR SEGMENTATION:
Segmentation works much better, but we do not have a clear black/white patter for hole/background. 
This makes the categorisation very hard, and kernel are hard to apply. Hardly any holes get recognized, only numbers and other artifacts.
- targets with multiple colors (inner rings black, outer white)
- large numbers get read as holes
- circles on the target get read as holes
- bullet holes have different sizes, depending on image size
- requires differnt approach?


3. SIMPLE BLOB DETECTOR:
Using cv2.SimpleBlobDetector works slightly better than the above methods. 
But still struggles massively with the holes not consistingly being white/black.
- bullet holes have different colors


4. CONTRAST RESIDUAL DETECTOR:
A median blur over a window wider than a hole rebuilds the printed target as if the holes weren't there. 
So abs(image − median) flags anything differing from its local surroundings in either direction.
- numbers do no longer get read as bullet holes
- some holes still do not get counted (especially same-on-same color)
- sometimes image artifacts get magnified and counted as holes
- targets with small inner different-colored circle are hard


5. USING OBJECT DETECTION MODEL FROM ROBOFLOW
Directly using a trained object detection model via an API-key yields us significantly better results.
This is promising, since it generalizes quite well.
- very close images with large bullet holes are difficult
- close clusters of holes are difficult
- black-on-black is still not perfect
-> instance segmentation for close hole clusters?
-> zoom into clusters for second pass?


6. USING INSTANCE SEGMENTATION MODEL FROM ROBOFLOW
There are only a handful models trained on 100-200 images. Therefore quality is bad.
- no usable output
-> create our own instance segmentation data set?


7. TRAINING OUR OWN YOLO26 MODEL WITH EXISTING DATASET
We used yolo26 on the project bat dataset (see paper). They used yolov8 and we use yolo26.
This was a test, whether the new models are much better than the old (2023) models.
- old mAP50: 96.5% our mAP: 99.3% 
- performance was noticeably better with the new version
- generalization to our examples were still bad




OTHER OPTIONS:
- scikit-image blob detectors (LoG/DoG)
- feature matching against a known-clean target (if the targets are always the same)
- already trained object detection models
- already trained instance segmentation models
- finetune an already trained model
- training our own model
- center-point detection model




TRAINING OUR OWN MODEL:
1. Building a labeled dataset (start with 100-200 images, load into CVAT and draw bounding boxes)
2. Write model & training code (use torchvision.models.detection)
3. Train the model (either locally or on Google Colab / Kaggle Notebook)
4. Evaluation (test the model on held-out test images)


Five options of labeling tool × training location:
- CVAT + local training — label and train fully self-hosted. (Pro: full data control / Contra: more setup work)
- CVAT + Roboflow training — label locally, train via Roboflow. (Pro: fast managed training / Contra: images still leave our infra)
- Roboflow + local training — label via Roboflow, train yourself. (Pro: easy labeling UI / Contra: images uploaded for labeling)
- Roboflow + Roboflow training — fully hosted pipeline. (Pro: fastest to first model / Contra: full data + model on third party)
- Use existing Roboflow model — call the pre-trained model via API, no dataset needed. (Pro: zero setup, instant / Contra: unknown fit to your targets, full third-party dependence)



CENTER-POINT DETECTION MODEL:
In contrast to all existing models using YOLO, this would train a model to detect the center pixel of the bullet hole instead of overlapping boxes. 
It does so by creating a heatmap over the whole image, with a probability for each pixel to be a center.
Then it takes the peaks in this heatmap as centroids. Could combine this with learning the radius to get circles.
- promises much better results in overlapping bullet holes
- no existing work in this direction on bullet holes yet
- no existing polished frameworks existent yet (CenterNet2 is the closest to usable)