## Project Details
Pipeline based on SAC_Object_Detection project 
Journal: Evolving Systems
DOI: 10.1007/s12530-025-09749-y
Title: Scale-invariant object detection by adaptive convolution with unified global-local context



# Supported Models
  - efficientdet-d0.pth
  - efficientdet-d1.pth
  - efficientdet-d2.pth
  - efficientdet-d3.pth
  - efficientdet-d4.pth
  - efficientdet-d5.pth
  - efficientdet-d6.pth
  - efficientdet-d7.pth

## Supported Models
  - EfficientDet-d1+SAC.pth
  - EfficientDet-d2+SAC.pth
  - EfficientDet_d1+SAC+Feature.pth
  - EfficientDet_d2+SAC+Feature.pth
  - EfficientDet_d1+SAC+Global+Feature.pth
  - EfficientDet_d2+SAC+Global+Feature.pth
  


## Installation

Supports 
- Python 3.9
- Cuda 9.0, 11.0 (Other cuda version support is experimental)
    
`cd installation`

`cat requirements_cuda9.0.txt | xargs -n 1 -L 1 pip install`


## Functional Documentation




## Pipeline

- Load Dataset

`gtf.set_train_dataset(root_dir, coco_dir, img_dir, set_dir, classes_list=[ ], batch_size=2, num_workers=3)`

- Load Model

`gtf.set_model(model_name="efficientdet-d3.pth", num_gpus=1, freeze_head=False);`

- Set Hyper Parameters

`gtf.set_hyperparams(optimizer="adamw", lr=0.00001, es_min_delta=0.0, es_patience=0)`

- Train

`gtf.train(num_epochs=5, val_interval=1, save_interval=1)`

