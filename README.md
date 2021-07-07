# keras-IconGAN 

**paper**: *Adversarial Colorization Of Icons Based On Structure And Color Conditions*
: [link](https://arxiv.org/abs/1910.05253)

conditional dual GAN model for icon generation
implemented with Keras

[official implementation (pytorch)](https://github.com/jxcodetw/Adversarial-Colorization-Of-Icons-Based-On-Structure-And-Color-Conditions)


![image](https://user-images.githubusercontent.com/13326768/124796544-402c6980-df8c-11eb-8352-ce83eedae27e.png)

<br>

## Examples

fake image | color image | shape image |
-----|-------|--------|
![image](https://user-images.githubusercontent.com/13326768/124796825-8d104000-df8c-11eb-8f79-ffbe4226c57f.png) | ![image](https://user-images.githubusercontent.com/13326768/124796877-9a2d2f00-df8c-11eb-8984-cceb1c74cacb.png) | ![image](https://user-images.githubusercontent.com/13326768/124796907-a31e0080-df8c-11eb-840e-2bc1e663c5d4.png)
![image](https://user-images.githubusercontent.com/13326768/124797389-27708380-df8d-11eb-8ad7-605cfb69b03a.png) | ![image](https://user-images.githubusercontent.com/13326768/124797415-30615500-df8d-11eb-9e19-3d9e8c499c31.png) | ![image](https://user-images.githubusercontent.com/13326768/124797444-37886300-df8d-11eb-90c6-b718cb2c14ef.png)
![image](https://user-images.githubusercontent.com/13326768/124797501-45d67f00-df8d-11eb-9bf8-69bbc0a79c51.png) | ![image](https://user-images.githubusercontent.com/13326768/124797524-4f5fe700-df8d-11eb-9ac1-6c56ad2e50b9.png) | ![image](https://user-images.githubusercontent.com/13326768/124797593-6b638880-df8d-11eb-87c0-4c53fc8433d4.png)


## Prerequisites
* Pillow
* tensorflow (2.5.0)
* tensorflow_addons (0.13.0)
* tensorboard (2.5.0)
* pickle
* opencv-python (for test.py)

## Directory 
```
├── preprocessed_data
│   ├── contour
│   ├── img
│   ├── labels
├── test_data
├── models
├── samples
├── logs
├── dataset.py
├── icongan.py
├── test.py
└── README.md
```

## Dataset
icon, contour from https://github.com/jxcodetw/Adversarial-Colorization-Of-Icons-Based-On-Structure-And-Color-Conditions
labels: used pickle 

## How to train
```bash
python icongan.py
```

## How to test
```bash
python test.py model_path=<model_path> color_input=<color_img_path> shape_input=<shape_img_path> output_path=<output_path>
```