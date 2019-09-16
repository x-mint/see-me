# SeeMe
Hackathon project for #PTSH19



## Inspiration
In the manufacturing, materials are affected by multiple external reasons and some portions of the end-products are produced with defects and become inoperable. Those outcomes increase operating cost and harm companies' reputation. 

## What it does
SeeMe is an automatic metallic surface defect inspection system can alarm and provide the ability to take precautions in time. It helps to increase efficiency and maintain high quality in production. It can be tested very easily by just downloading the app to your smartphone and placing the phone in a mobile phone mount.

SeeMe has two main components:

* Classifying surface defects 
* Localizing defected area


## How we built it
Our machine learning technology is based on U-Net, a promising model for the image segmentation tasks. 

**Deep Learning Framework:** PyTorch

**Deep Learning Model:** U-Net 

**User Interface:** React Native

**Data**: NEU Surface Defect Database

**Server**: Google Cloud

SeeMe is based on open source technology and supports the integration of other platforms and services.

## Challenges we ran into
Different dataset structures 

We first planned SeeMe only as an embedded system and looked for the component how to make it easy to test and more engaging within workers and technology. This motivated us to start with an app.

## Accomplishments that we're proud of
* We developed a product prototype in a very short amount of time as a team.
* We needed to learn mobile app development, server design, and bring up useful ideas for quick tests of data and deep learning models.

## What we learned
****


## What's next for SeeMe
**Adaptation**: We will add a property that allows project managers to expand the library of defect images by mobile

**End-to-end Solutions:** We will develop a simple embedded system using Raspberry Pi.

**Security:** We will apply PySyft for encrypted, privacy-preserving deep learning.

SeeMe will look for new industries and defect identification challenges to help manufacturing and retail process more efficient and productive.
