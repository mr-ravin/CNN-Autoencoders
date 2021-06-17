# CNN based Autoencoders :
Implementation of CNN based autoencoders.

#### Author: [Ravin Kumar](https://mr-ravin.github.io)

##### Written in Tensorflow: 1.12

## Directory Structure :
```
|
|--input   (dir)
|
|--pros    (dir)
|
|--restmp  (dir)
|
|--test    (dir)
|
|--testres (dir)
|
|--script.py
|
|--run.py
```

### Description of various directories and files :
- input directory contains input images.
- test directory contains test images.
- pros directory will contain processed images, that then feed to the encoder side of autoencders.
- restmp directory contains intermediate results during training phase.
- testres directory will contain the output over the images at inference time.
- script.py this is a module based file in python, where cnn based autoencoders are implimented.
- run.py file is automated version of the process.
- Note: lets say at training time input image name is "abc.jpg", then please save it's desired output image by "abcr.jpg" in the input directory.

### Steps required :
- We can import script.py file and than proceed :
```
import script
script.preprocess() # it will convert all input dir images to fixed size of 256 x 256.
# processed images are saved to pros directory.
script.generate_more_data(90) # it will generate more data from processed images of pros directory.
# by rotating them. here images are rotated by 90 degree to produce more data.
script.run(1000) #  1000 defines the number of epoches for training.
```
- To automate the process of training and testing :
```
>>> python3 run.py 1000
```

Note: you do have to resize the images of test directory manually or by using the script code separately.

### Working Project Videos
A sample working demonstration of Autoencoders is available at [Youtube](https://www.youtube.com/watch?v=acfAf6eLbh8).

[![working demonstration](https://github.com/mr-ravin/CNN-Autoencoders/blob/master/flag_autoencoders.gif)](https://www.youtube.com/watch?v=acfAf6eLbh8)

```python
Copyright (c) 2018 Ravin Kumar
Website: https://mr-ravin.github.io

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation 
files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, 
modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the 
Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the 
Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE 
WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR 
COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, 
ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
```
