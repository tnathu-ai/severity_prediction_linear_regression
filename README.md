# Assignment 2: Modeling
+ Identify the patterns of how these severe accidents happen and the key factors affecting vehicle accident severity levels in the US and provide the most accurate prediction model.
+ Might implement well-informed actions and better allocate financial and human resources.
+ Might be able to predict the severity of the accident based on the data.
+ Provide officials with suggestions to take adequate measures with higher precision to lessen accident impacts and improve road safety.

# Problem statement
+ Understand the cause and effect of the accidents
+ Building several machine learning models that can help forecast future accidents.
+ Predict severity level of car accidents in the USA from 2016-2020 using regression models.

# Target variable
Severity (Univariate variable)

# Problem Type
Regression (predict value of severity)


# Metric:
MAE
MSE
RMSE
R2

# Result:
* California has the highest number of accidents, then we have Texas and Florida.
* Most accidents occurred near a traffic signal, especially where a junction or a crossing was present.
* Distance of the accident is more or less proportional to the severity, and accidents with severity 4 have the longest distance.
* Most frequent cases of the weather condition are apparent.
* Days with the most accidents are working days, while we have a frequency of at least 2/3 less on the weekend.
* An accident is much less likely to be severe if it happens near a traffic signal while more likely if near a junction.

### Key Findings
* Country-wide accident severity can be accurately predicted with limited data attributes (location, time, weather, and POI).
* **Minute(frequency-encoding)** is the most useful feature. An accident is more likely to be a serious one when accidents happen less frequently at this time.
* Spatial patterns are also very important. For small areas like **street** and **zipcode**, severe accidents are more likely to happen at places having more accidents while for larger areas like **city** and **airport region**, at places having less accident.
* **Pressure** is top fourth important feature in the random-forest model and there is negative correlation between pressure and severity.
* If an accident happens on **Interstate Highway**, there is a 2% chance that it will be a serious one, which is about 2.3 times of average and higher than any other street type.
* An accident is much less likely to be severe if it happens near **traffic signal** while more likely if near **junction**.

# Objective
+ recognize key factors affecting the accident severity
+ Create ML model that can accurately predict accident severity

# Inspiration
US-Accidents can be used for numerous applications such as real-time car accident prediction, studying car accidents hotspot locations, casualty analysis and extracting cause and effect rules to predict car accidents, and studying the impact of precipitation or other environmental stimuli on accident occurrence. The most recent release of the dataset can also be useful to study the impact of COVID-19 on traffic behavior and accidents.


This paper describes US-Accidents, a unique, publicly available motor vehicle accident dataset, and its process of creation – thatincludes several important steps such as real-time traffic data col-lection, data integration, and multistage data augmentations usingmap-matching, weather, period-of-day, and points-of-interest data.To the best of our knowledge, US-Accidents is the first countrywidedataset of this scale, containing about2.25million traffic accidentrecords collected for the contiguous United States over three years.From this dataset, we were able to derive a variety of insights withrespect to the location, time, weather, and points-of-interest of anaccident. We believe that US-Accidents provides a context for fu-ture research on traffic accident analysis and prediction. In termsof our own future work, we plan to employ this dataset to performreal-time traffic accident prediction

# Application
+ Having a model prototype that can be deployed in production that further predict severe accidents in real-time in the US.
+ Studying accident hotspot locations; casualty analysis (extracting cause and effect rules to predict accidents);
+ The impact of precipitation or other environmental stimuli on accident occurrence.
+ Urban planning and improving transportation infrastructures.

# Description
This is a countrywide car accident dataset, which covers 49 states of the USA. The accident data are collected from February 2016 to Dec 2020, using multiple APIs that provide streaming traffic incident (or event) data. These APIs broadcast traffic data captured by a variety of entities, such as the US and state departments of transportation, law enforcement agencies, traffic cameras, and traffic sensors within the road-networks. Currently, there are about 1.5 million accident records in this dataset. Check here to learn more about this dataset.

# Usage Policy and Legal Disclaimer
This dataset is being distributed only for Research purposes, under Creative Commons Attribution-Noncommercial-ShareAlike license (CC BY-NC-SA 4.0). By clicking on download button(s) below, you are agreeing to use this data only for non-commercial, research, or academic applications. You may need to cite the above papers if you use this dataset.

# Limited: 
+ the final model is dependent on only a small range of data attributes that are easily achievable for all regions in the United States and before the accident really happened.




# Metric Forecasting
#### What?
+ Metric forecasting is self-explanatory — it refers to forecasting a given metric, like **the severity of the accident (long or short delay) impact on the traffic**, in the short-term future.
+ Specifically, forecasting involves techniques that use historical data as inputs to generate a predicted output. Even if the output itself is not entirely accurate, forecasting can be used to gauge the general trend of where a particular metric is going.
#### Why?
+ Forecasting is basically like looking into the future. By predicting (with some level of confidence) what will happen in the future, you can make more informed decisions more proactively. The result of this is that you’ll have more time to make decisions and ultimately reduce the likelihood of failure.


# Files to the Development Folder (Dash folder)

* `app.py`    a Dash application
* `.gitignore`    used by git, identifies files that won’t be pushed to production
* `Procfile`    used for deployment
* `requirements.txt`    describes your Python dependencies, can be created automatically

python version 3.8.8

# Option 1: WORKING ON YOUR LOCAL COMPUTER

1. Install Conda
   by [following these instructions](https://conda.io/projects/conda/en/latest/user-guide/install/index.html). Add Conda
   binaries to your system `PATH`, so you can use the `conda` command on your terminal.

2. Install jupyter lab and jupyter notebook on your terminal

+ `pip install jupyterlab`
+ `pip install jupyter notebook`

### Jupyter Lab

1. Download the 3879312 zipped project folder. Unzip it by double-clicking on it.

2. In the terminal, navigate to the directory containing the project and install these packages and libraries

```
conda install ecos  

conda install CVXcanon  

conda install fancyimpute

pip install pandas 

pip install scikit-learn 

pip install matplotlib 

pip install seaborn 

pip install missingno

pip install statsmodel

pip install dash

pip install dash-auth

pip install dash-renderer

pip install dash-bootstrap-components

pip install cufflinks

pip install --upgrade pandas

pip install plotly==5.3.1 

pip install plotly --upgrade

pip install dash_daq 

pip install dash-html-components 

pip install dash_bootstrap_components 

pip install dash-core-components

pip install numpy

pip install WordCloud
```

3. Enter the newly created directory using `cd directory-name` and start the Jupyter Lab.

```
jupyter lab

```

You can now access Jupyter's web interface by clicking the link that shows up on the terminal or by
visiting http://localhost:8888 on your browser.

4. Click on assignment1.ipynb in the browser tab. This will open up my main file in the Jupyter Lab.

5. Follow the steps in the Jupyter Lab. If you get to Task 4, you can look at `dash_as1.ipynb` for better understanding
of **Visualization Dashboard** before running the web app on the local machine.

### Note: If the Jupyter Notebook is not responding due to many requests

Error [(The page is not responding)](https://stackoverflow.com/questions/48615535/jupyter-notebook-takes-forever-to-open-and-then-pages-unresponsive-mathjax-i)

I had to restart the notebook; and it did not work. This is because I was printing out too much and the following
scripts resolved the issue by clear out all the output to run through the whole kernal:

1. `conda install -c conda-forge nbstripout` or `pip install nbstripout`

2. `nbstripout filename.ipynb`


### Deploying Dash Web App on the local machine

1. Enter the newly created directory using `cd directory-name` and start the Web App.

2. First create a **virtual environment** with conda or venv inside a temp folder, then activate it.

```
virtualenv venv

# Windows
py -m venv .env
.env\Scripts\activate

# Or Linux
source venv/bin/activate

```

3. Run the app

```

python app.py

```


# Option 2: RUNNING USING ONLINE RESOURCES (1-click)

The easiest way to start executing this notebook is to click the "Run" button at the top of this page, and select "Run
on Binder". This will run the notebook on [mybinder.org](https://mybinder.org), a free online service for running
Jupyter notebooks. You can also select "Run on Colab" or "Run on Kaggle".

You can access my full version of jupyter notebook here: 
+ **Task 1:** [Cleaning](https://plotly.com/~tnathu/17/assignment-1-data-exploration-this-is/)
+ **Task 1.8:** [Encoding](https://plotly.com/~tnathu/18/assignment-1-data-exploration-this-is/)
+ **Task 2:** [Exploration](https://plotly.com/~tnathu/19/assignment-1-data-exploration-this-is/)
+ **Task 4:** [Visualization](https://plotly.com/~tnathu/20/task-4-visualisation-dashboard-this-is/)



## Demo:

The following are screenshots for the app in this repo:

![img.png](Images/compensation_women.png)

![img.png](Images/participation_onl_dev.png)

![screenshot](Images/comp_gender.png)



## Future improvement:

#### Manging future data life cycle:
+ Labeling: For supervised learning, then I need to make sure that my labels are accurate. 

+ Feature space coverage: Ensure that my training dataset has examples that cover the same feature space as my future model's request. 

+ Minimal Dimensionality & Maximum predictive data: I also want to reduce the dimensionality of your feature vector to optimize my system performance while retaining or enhancing the predictive information on my data. 

+ Fairness:  I will need to consider and measure the fairness of my data and model, especially for rare conditions, for example, in this case, where gender inequality prerequisites may be critical to success. 

