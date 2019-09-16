### Data

[NEU Surface Defect Database](http://faculty.neu.edu.cn/yunhyan/NEU_surface_defect_database.html) by [K. Song and Y. Yan, “A noise robust method based on completed local binary patterns for hot-rolled steel strip surface defects,” Applied Surface Science, vol. 285, pp. 858-864, Nov. 2013.](https://www.sciencedirect.com/science/article/pii/S0169433213016437)


This database contains six types of defects including crazing, inclusion, patches, pitted surface, rolled-in-scale, and scratches.


![data](dataset_sample.png)

### Usage
- Setup with virtual environment (Python 3):
```
  python3 -m venv see-me-env
  source see-me-env/bin/activate
```
- Install the requirements:
```
  pip3 install git+https://github.com/qubvel/segmentation_models.pytorch
  pip3 install -r requirements.txt
 ```

### The Directory Layout

    .
    ├── input          
    │   ├── images         
    |       └── crazing_1.jpg
    |       └── .
    |       └── .
    │   └── annotations.csv        
    ├── pretrainedmodels 
    |   └── best_model.pth
    |   └── train_val_loss.png
    ├── server
    │   ├── images          
    |       ├── uploads   
    |          └── crazing_1.jpg
    |       ├── model_outputs   
    |          └── crazing_1.jpg
    ├── SeeMeTrainNeu.py   
    ├── SeeMePredictOnceNeu.py   
    ├── LICENSE
    └── README.md
    
- [Download NEU database](https://drive.google.com/open?id=0B5OUtBsSxu1Bdjh4dk1SeGYtNFU) into input/images 
- Change directory to SeeME/ and run the train script to train the resnet model. The script will save the trained model as pretrainedmodels/best_model.pth
```
python3 SeeMeTrainNeu.py 
```
- Go back to SeeMe directory and run the script with an example upload that is contained in ~/SeeMe/server/images/uploads
```
python3 SeeMePredictOnceNeu.py --image crazing_1.jpg
```
- You can see the created output image in the directory ~/SeeMe/server/images/model_outputs

- The name of the output image is the same as the input image.

