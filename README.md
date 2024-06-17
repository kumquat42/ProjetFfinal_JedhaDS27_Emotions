# ProjetFinal_JedhaDS27_Emotions - Emotion detector on photos and videos for emotion sensibility disorders  

Emotion detector on photos and videos for emotion sensibility disorders

Promo Jedha Bootcamp Data Science & Engineering #27 - Jan.2024
**Labeau Gregory, Remy Théo, Ghandri Firas, Morabit Youssef, Beraud Mélanie** 


### Check out the demo !!! 2024_Presentation_video_EmoTycoon.webm  

### Check out the presentation of the application during our DemoDay !!! 2024_EmoTycoon.pdf


## Directories:  
README.md  
2024_Presentation_video_EmoTycoon.webm: Demo  
2024_EmoTycoon.pdf: Presentation made in January 2024, DemoDay #49, at Jedha, Paris, France  
 
**Modèles**  
Resnet50 on 7 classes: transfer learning from keras Resnet169 _ +/- Imagenet weights  -> not ok.
model_Greg: model derived from Sanskar Hasija (https://www.kaggle.com/code/odins0n/emotion-detection): Notebook, graphs and predictions. -> not ok.  
**model_Youssef: home-made model: Notebook, graphs and predictions -> The one used for the application**  
model_Theo: home-made model: : Notebook, graphs and predictions -> not ok.  
Comparison models.ods: brainstormng on model comparison.  
Modele_train.ipynb: Gradcam tests, ongoing.

**Datasets**  
|FER2013 : Publicly available dataset already decomposed in train and test sets.  
||train : 28709 images of famous and less famous faces. 
||test : 7178 images 
|Photos_classees : 189 images
|20240112_Votes.ods : Assignement of the scrapped images  
|listeimage_scrap_en_ordre.ods : Liste of the 189 images
|Datasets_comparison.ipynb : Some EDA relative to the datasets.

**Deployment**  
Files necessary to run the application:  
Dockerfile  
api.py  
frontend.py  
requirements.txt  
youssef.h5  
youssef.json  
**!!! Check out the application: You need the api.py (application), streamlit, dockerfile !!!**

