import numpy as np
import pandas as pd
import gc
import lightgbm as lgbm
from sklearn.metrics import roc_auc_score
from deap import creator, base, tools, algorithms
import random
from dtypes import dtypes

params = {
        # Task based parameter
        'application' :'binary',
        'learning_rate' : 0.1,
        'num_iterations': 1000,
        'boosting' : 'goss',

        # Deal with overfitting
#        'bagging_fraction': 0.9, 
#        'bagging_freq': 1,
        'min_data_in_leaf': 5000,
        'feature_fraction': 0.8,
        'num_leaves': 31,
        'max_depth': -1,
        'max_bin': 255,

        # Others
        'metric': 'auc',
        'num_threads': 16,
        'scale_pos_weight': 200,
}


use_features = ['app','device','os','channel','total_click','click_per_channel','click_per_os','click_app_os','click_app_channel',\
               'click_ip_app_os_device_hour','click_ip_app_os_device_minute',\
               'hour','click_ip_app_os','click_app','click_channel',\
               'next_click_ip','next_click_ip_channel','next_click_ip_app','next_click_ip_device','next_click_ip_os',\
               'next_click_ip_app_os_device','next_click_ip_app_os','next_click_app_channel',\
               'p_click_ip','p_click_ip_app','p_click_ip_device','p_click_ip_os','p_click_ip_app_os_device','p_click_ip_app_os',\
               'tar_app','tar_os','tar_device','tar_channel']

ITERATION = 1000
NUM_TRAIN = 10000000
NUM_CV = 5000000

X_day8 = pd.read_csv("../../feature/train-day8-total.csv", dtype=dtypes, skiprows=range(1, 62945076-NUM_TRAIN))
X_day9 = pd.read_csv("../../feature/train-day9-total.csv", dtype=dtypes, skiprows=range(1, 53016938-NUM_CV))

Y_day8 = X_day8["is_attributed"]
X_day8 = X_day8.drop(["is_attributed"], axis=1)
Y_day9 = X_day9["is_attributed"]
X_day9 = X_day9.drop(["is_attributed"], axis=1)
for data in (X_day8, X_day9):
    data['app'] = data['app'].astype('category')
    data['os'] = data['os'].astype('category')
    data['device'] = data['device'].astype('category')
    data['channel'] = data['channel'].astype('category')
    data['hour'] = data['hour'].astype('category')

fout = open("evolution.dat",'w')

def getFitness(individual, X_train, X_test, y_train, y_test):

	# Parse our feature columns that we don't use
	# Apply one hot encoding to the features
	cols = [index for index in range(len(individual)) if individual[index] == 0]
	X_trainParsed = X_train.drop(X_train.columns[cols], axis=1)
	X_testParsed = X_test.drop(X_test.columns[cols], axis=1)

	# Apply gradient boosting on the data, and calculate accuracy
	gbm_train = lgbm.Dataset(X_train, y_train)
	gbm_cv = lgbm.Dataset(X_test, y_test)

	clf = lgbm.train(params, gbm_train, valid_sets=[gbm_cv], early_stopping_rounds=10)
	cv_pred = clf.predict(X_test, num_iteration=clf.best_iteration)
	score = roc_auc_score(y_test, cv_pred)

	# Return calculated accuracy as fitnes
	fout.write("Individual: {}  Fitness_score: {} ".format(individual,score))
	fout.flush()
	return (score,)

#========DEAP GLOBAL VARIABLES (viewable by SCOOP)========

# Create Individual
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

# Create Toolbox
toolbox = base.Toolbox()
toolbox.register("attr_bool", random.randint, 0, 1)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, len(X_day8.columns))
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# Continue filling toolbox...
toolbox.register("evaluate", getFitness, X_train=X_day8, X_test=X_day9, y_train=Y_day8, y_test=Y_day9)
toolbox.register("mate", tools.cxOnePoint)
toolbox.register("mutate", tools.mutFlipBit, indpb=0.1)
toolbox.register("select", tools.selTournament, tournsize=3)

#========

def getHof():

	# Initialize variables to use eaSimple
	numPop = 20
	numGen = 10
	pop = toolbox.population(n=numPop)
	hof = tools.HallOfFame(numPop)
	stats = tools.Statistics(lambda ind: ind.fitness.values)
	stats.register("avg", np.mean)
	stats.register("std", np.std)
	stats.register("min", np.min)
	stats.register("max", np.max)

	# Launch genetic algorithm
	pop, log = algorithms.eaSimple(pop, toolbox, cxpb=0.5, mutpb=0.2, ngen=numGen, stats=stats, halloffame=hof, verbose=True)

	# Return the hall of fame
	return hof


if __name__ == '__main__':

	#individual = [1 for i in range(len(X_day8.columns))]
	#accuracy = getFitness(individual, X_day8, X_day9, Y_day8, Y_day9)
	#print('Validation accuracy with all features: \t' + str(accuracy) + '\n')

	hof = getHof()

	for individual in hof:
		print(individual)

