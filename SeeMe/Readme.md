### Usage

Go to instance and open ssh in browser
```
cd SeeMe/seeMeEnv
source bin/activate
```

Go back to SeeMe directory and run the script with an example upload that is contained in ~/SeeMe/server/images/uploads
```
python3 SeeMePredictOnceNeu.py 4.jpg
```
You can see the created output image in the directory ~/SeeMe/server/images/model_outputs

The name of the output image is the same as the input image.
```
deactivate 
```