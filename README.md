# SeeMe
Hackathon project for #PTSH19



## Inspiration
In the manufacturing, materials are affected by multiple external reasons and some portions of the end-products are produced with defects and become inoperable. To ensure quality, products are inspected manually before distributing to the customers. Manual inspection is slow, costly, ineffective and occasionally dangerous, especially in the steel manufacturing process. 

## What it does
SeeMe is an automatic metallic surface defect inspection system can alarm and provide the ability to take precautions in time. It helps to increase efficiency and maintain high quality in production. It can be tested very easily by SeeMe mobile app or you can integrate SeeMe deep learning model to your environment to be used in real-time applications.

SeeMe deep learning model has two main components:

* Classifying surface defects 
* Localizing defected area


## How we built it
SeeMe is mainly based on open source technology and supports the integration of other platforms and services.

**Deep Learning Framework:** PyTorch

**Deep Learning Model:** U-Net 

**User Interface:** React Native

**Data**: NEU Surface Defect Database

**Server**: Google Cloud


## Challenges we ran into
* Different dataset structures and labeling types that we encountered during our quick testing stages on various data resources was challenging to adapt our pre-processing steps and fine-tuning our models.

* We first planned SeeMe only as an embedded system and looked for the component how to make it easy to test and more engaging within workers and technology. This motivated us to start with an app.

## Accomplishments that we're proud of
* We developed a product prototype in a very short amount of time as a team.
* We needed to learn mobile app development, server design, and bring up useful ideas for quick tests of data and deep learning models.

## What we learned
We learned how to use and adapt deep learning models in torch vision to our task in a very short amount of time and each part of end-to-end product development.

## What's next for SeeMe
**New Features**: We will add a property that allows project managers to expand the library of defect images and their labels by mobile phones to enable adaptation to new products or defect types 

**End-to-end Solutions:** We will develop a simple embedded system using Raspberry Pi.

**Security:** We will apply PySyft for encrypted, privacy-preserving deep learning.

SeeMe will look for new industries and defect identification challenges to help manufacturing and retail process more efficient and productive.
