# TalkingData
Kaggle competition 

see: https://www.kaggle.com/c/talkingdata-adtracking-fraud-detection

Fraud risk is everywhere, but for companies that advertise online, click fraud can happen at an overwhelming volume, resulting in misleading click data and wasted money. Ad channels can drive up costs by simply clicking on the ad at a large scale. With over 1 billion smart mobile devices in active use every month, China is the largest mobile market in the world and therefore suffers from huge volumes of fradulent traffic.

TalkingData, China’s largest independent big data service platform, covers over 70% of active mobile devices nationwide. They handle 3 billion clicks per day, of which 90% are potentially fraudulent. Their current approach to prevent click fraud for app developers is to measure the journey of a user’s click across their portfolio, and flag IP addresses who produce lots of clicks, but never end up installing apps. With this information, they've built an IP blacklist and device blacklist.

While successful, they want to always be one step ahead of fraudsters and have turned to the Kaggle community for help in further developing their solution. In their 2nd competition with Kaggle, you’re challenged to build an algorithm that predicts whether a user will download an app after clicking a mobile app ad. To support your modeling, they have provided a generous dataset covering approximately 200 million clicks over 4 days!

# Folder 
explore: jupyter notebooks to do exploratoary data analysis

generate: python script to generate features

predict: python script to do predictions

test: python script to test features

# Results

1.Efficient memory use
Since the entire training set is 7 GB in size and the computation resource is limited, special actions have to be taken to fit the data into memory. 1. Pandas uses int64 and float64 as default data type for integers and floats. We chose an appropriate data type based on the range of the feature. 2. Divide data into several chunks and process data separately. The entire data consist of clicks from 11-06 to 11-10, so it is naturally to divide based on the date. 3. Delete any dataframe immediately when it is no longer needed and use garbage collect. 

2. Feature engineering
The training data consists of four consecutive days, which are 11-06, 11-07, 11-08, 11-09, and the test data are from 11-10. Due to the limitation of the computational resources, only a subset of the training data were used. The data from 11-08 were used as the training set and those from 11-09 were used as local validation set, and they have about 62 million and 53 million entries respectively. 
In addition to the original features provided from the organizer, new features can be grouped into 4 categories. (1) Aggregate features: the total number of clicks for a type, e.g. ‘click_ip_app’ means total number of clicks from a particular combination of ip and app. (2) Time delta features: the time difference between the current click and the next click from the same type. (3) Cumulative count features: total number of clicks before this click for a certain type. (4) Unique features: unique number of one type grouped by another type, e.g. ‘uniq_app_by_ip’ means the number of unique app for each ip. 
Then, the Pearson correlation among all features were plotted and the features that have correlation higher than 0.85 were removed. Afterwards, further choice of the features were done by the following method: first, a smaller model, which used lightgbm with only 20 million training data and 10 million validation data, was constructed and then applied genetic algorithm to search for the best features. This method turns out to be better than the random search method and the recursive removing method. 

3. Model selection
Lightgbm was used. Bayesian grid-based search and the smaller evaluation model was used to tune the hyperparameters of the lightgbm model. Afterwards, last 25 million rows of the training data were used as cross-validation set, and the rest were used for training. The test was done on 11-10 data. The final local cv score is 0.9851, the public LB score is 0.9781 and the final private LB score is 0.9762, resulting top 37% in the competition.

4. Real world case
    In real world industrial application, simple models that can train and predict fast are often used. Logistic regression is a very popular simple linear model. A new model that is also used is factorization machine (FM). FM works beyond a linear model that it uses latent vectors to take correlation among features into account. Compared with logistic regression, it is faster and explicitly consider higher order term among features. Compared with support vector machine with kernel trick, it is much faster but still keeping the higher order interactions. In real world scenario, online training of the model is often required. A popular method is online gradient descent (OGD). Another new popular method is called follow the regularized leader (FTRL). It has the advantage of as accurate as OGD but still can yield a sparse model.
