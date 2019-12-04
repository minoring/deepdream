# deepdream
Experiment that visualizes the patterns learned by a neural network.

## Usage
```
python main.py --num_steps=900 --learning_rate=0.001 --scaling=False
```
```
python main.py --num_steps=300 --learning_rate=0.001 --scaling=True --num_octaves=3
```
```
python main.py --num_steps=300 \
               --learning_rate=0.001 \
               --scaling=True \
               --URL=1024px-Van_Gogh_-_Starry_Night_-_Google_Art_Project.jpg
```
## Experiment
Training Without scaling          |  Same number of steps but add scaling
:-------------------------:|:-------------------------:
![](https://github.com/minoring/deepdream/blob/master/misc/Step900.jpg)  |  ![](https://github.com/minoring/deepdream/blob/master/misc/Step900_scaling.jpg)
![](https://github.com/minoring/deepdream/blob/master/misc/training.gif) | ![](https://github.com/minoring/deepdream/blob/master/misc/training_scaling.gif)

You can see that first without scaling
- The output is noisy.
- The image is low resolution.
- The patterns appear like they're all happening at the same granularity.

Addresses these problems is applying gradient ascent at different scales. 
This will allow patterns generated at smaller scales to be incorporated into patterns at higher scales and filled in with additional detail.

## References
### blogs
- [Inceptionism: Going Deeper into Neural Networks](https://ai.googleblog.com/2015/06/inceptionism-going-deeper-into-neural.html)
### code
- https://www.tensorflow.org/tutorials/generative/deepdream

### Model Architecture
Code: [Inception V3 model for Keras](https://github.com/keras-team/keras-applications/blob/master/keras_applications/inception_v3.py)

Paper: [Rethinking the Inception Architecture for Computer Vision](http://arxiv.org/abs/1512.00567) (CVPR 2016)
![](https://github.com/minoring/deepdream/blob/master/misc/model.png)
