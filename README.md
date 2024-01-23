# ProjetFinal_JedhaDS27_Emotions - Emotion detector on photos and videos for emotion sensibility disorders  

Emotion detector on photos and videos for emotion sensibility disorders

Promo DataScience 27 - Jan.2024  
**Labeau Gregory, Remy Théo, Ghandri Firas, Morabit Youssef, Beraud Mélanie** 


## Directories:
README.md  
**Modèles**  
Modele_train.ipynb: exemple of model for pre- and post-processing purposes.  
|Resnet50: model base on Resnet50 from keras  
||Resnet50.ipynb: Notebook for Resnet50 implementation.  
||class_res1.h5/his_res1.csv : model and history from Resnet50 with Imagenet weights and fine-tuning of last layer only

**Datasets**  
|FER2013 : Publicly available dataset already decomposed in train and test sets.  
||train : 28709 images of famous and less famous faces. 
||test : 7178 images 
|Photos_classees : 189 images
|20240112_Votes.ods : Assignement of the scrapped images  
|listeimage_scrap_en_ordre.ods : Liste of the 189 images
|Datasets_comparison.ipynb : Some EDA relative to the datasets.

## Models:

Resnet50: transfer learning from keras Resnet169 _ +/- Imagenet weights  
model_Greg: model derived from Sanskar Hasija, https://www.kaggle.com/code/odins0n/emotion-detection
model_Youssef: home-made model
model_Theo: home-made model

## Check out the application:
Just wait a little bit for the code.
You need the api.py (application), streamlit, dockerfile.

