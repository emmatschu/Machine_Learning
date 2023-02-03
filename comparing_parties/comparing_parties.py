import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math
import random
from sklearn import linear_model
from sklearn.metrics import accuracy_score
# Used for Trees
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
from IPython.display import Image
from pydot import graph_from_dot_data
from six import StringIO
from random import seed

# Used for SVM
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
 

# Extra: for processing my data
'''Splits data into bool based on a condition'''
def make_bool_col(data_pd, indx, column_name):

    #indx = data_pd[column_name] > val
    data_pd.loc[indx, column_name] = True
    data_pd.loc[~indx, column_name] = False
    
    # count occurances of true and false
    print(data_pd[column_name].value_counts())
    
    return data_pd
    
# Extra: for getting ginis
'''calculates gini index for true and false values of column'''
def col_gini(data_pd, column_name, target):

    print(f'The false value for "{column_name}" is: {gini_index(data_pd[data_pd[column_name] == False], target)}')
    print(f'The true value for "{column_name}" is: {gini_index(data_pd[data_pd[column_name] == True], target)}')
    print("")


""" Check accuracy of decision tree models made with different ccp_alpha values """
def kNN_tester(x_train, y_train, x_test, y_test, alpha_list):
    
    test_list = []
    train_list = []
    
    for alpha in alpha_list:
        
        my_dt = DecisionTreeClassifier(ccp_alpha = alpha)
        my_dt.fit(x_train, y_train)
        pred = my_dt.predict(x_train)
        train_list.append(accuracy_score(y_train, pred))
        
        pred = my_dt.predict(x_test)
        test_list.append( accuracy_score(y_test,pred))
    
    fig = plt.figure()
    ax = plt.axes()
    ax.plot(alpha_list, test_list)
    ax.plot(alpha_list, train_list)
    for i in range (len(test_list)):
        plt.text(alpha_list[i], test_list[i], f'{alpha_list[i]}, {test_list[i]}')
    plt.show()
    
    return(test_list)
 
""" Check accuracy of decision tree models made with different features excluded """
def exclude_feature_tester(x_train, y_train, x_test, y_test, column_list, alpha):
    
    test_list = []
    train_list = []
    
    for col in column_list:
        
        my_dt = DecisionTreeClassifier(ccp_alpha = alpha)
        my_dt.fit(np.delete(x_train, col, 1), y_train)
        pred = my_dt.predict(np.delete(x_train, col, 1))
        train_list.append(accuracy_score(y_train, pred))
        
        pred = my_dt.predict(np.delete(x_test, col, 1))
        test_list.append(accuracy_score(y_test,pred))
    
    fig = plt.figure()
    ax = plt.axes()
    ax.plot(column_list, test_list)
    ax.plot(column_list, train_list)
    for i in range (len(test_list)):
        plt.text(column_list[i], test_list[i], f'{column_list[i]}, {test_list[i]}')
    plt.show()
    
    return(test_list)
    
# SVM Model Functions
""" Check accuracy of SVM with different min_df """
def min_df_tester(trn_df, tst_df, inp, target, mindf_list, loss):
     
    test_list = []
    
    for mindf in mindf_list:
        
        cf, pred, tl = make_my_svm(trn_df, tst_df, inp, target, mindf, loss)
        
        test_list.append(accuracy_score(tst_df[target], pred))
        
    fig = plt.figure()
    ax = plt.axes()
    ax.plot(mindf_list, test_list)
    
    for i in range (len(mindf_list)):
        plt.text(mindf_list[i], test_list[i], f'{mindf_list[i]}, {test_list[i]}')
    plt.show()
    
    return(test_list)

""" Check accuracy of SVM with different loss techniques """
def loss_tester(trn_df, tst_df, inp, target, mindf, loss_list):
     
    test_list = []
    
    for loss in loss_list:
        
        cf, pred, tl = make_my_svm(trn_df, tst_df, inp, target, mindf, loss)
        
        test_list.append(accuracy_score(tst_df[target], pred))
        
    fig = plt.figure()
    ax = plt.axes()
    ax.plot(loss_list, test_list)
    
    for i in range (len(loss_list)):
        plt.text(i, test_list[i], f'{i}, {test_list[i]}')
    plt.show()
    
    return(test_list)
    
''' Build SVM model to address my question '''
def make_my_svm(trn_df, tst_df, inp, target, min_df, loss):
    # create my TfidfVectorizer, does some filtering
    vect = TfidfVectorizer(strip_accents="unicode", stop_words='english', min_df=min_df)
    # fits and transforms my CountVectorizer/data
    train_vect = vect.fit_transform(trn_df[inp])
    
    # makes classifier
    classf = SGDClassifier(loss=loss).fit(train_vect, trn_df[target])

    # adapt my test data
    new_counts = vect.transform(tst_df[target])

    # make prediction
    predicted = classf.predict(new_counts)

    return(classf, predicted, new_counts)


# Question 2 (Cross-validation)
''' Use cross-validation to select the appropriate model'''


def cross_val(model, input_c, target):
    scores = cross_val_score(model, input_c, target, cv=10, scoring='accuracy')
    return (scores)


# Question 3 (parameters)
''' Get specific parameters for my model'''


def params(trn_df, tst_df, train_indx_lst, test_indx_lst, target):
    # seems like the same deal but with full data
    my_mod, temp = my_tree(
        trn_df, tst_df, train_indx_lst, test_indx_lst, target)

    return(my_mod)

# Tree Functions
''' Build tree model to address my question '''
def make_my_tree(trn_df, tst_df, train_indx_lst, test_indx_lst, target):
    # necessary processing for this model
    for (indx, column_name) in enumerate(trn_df):
        trn_df = make_bool_col(trn_df, train_indx_lst[indx], column_name)
        tst_df = make_bool_col(tst_df, test_indx_lst[indx], column_name)

    # split into the input_c variables and the target classes
    input_c = trn_df.loc[:, trn_df.columns !=
                             target].to_numpy(dtype=np.float16)
    target_c = trn_df.loc[:, trn_df.columns == target].to_numpy(dtype=np.float16)

    # Get the variable names
    var_names = list(x for x in trn_df.columns if x != target)
    print(var_names)
    print(target)

    # build a decision tree
    dt = DecisionTreeClassifier(max_features=6, ccp_alpha=0.001)

    # train my decision tree
    dt.fit(input_c, target_c)

    # make prediction
    predicted = dt.predict(tst_df[var_names])

    return(dt, predicted, tst_df)




##################################################################################
# Supporting functions for building my tree
def gini_index(data_pd: pd.DataFrame, class_var: str) -> float:
    """
    Given the observations of a binary class and the name of the binary class column
    calculate the gini index
    """
    # count classes 0 and 1
    count_A = np.sum(data_pd[class_var] == 0)
    count_B = np.sum(data_pd[class_var])

    # get the total observations
    n = count_A + count_B

    # If n is 0 then we return the lowest possible gini impurity
    if n == 0:
        return 0.0

    # Getting the probability to see each of the classes
    p1 = count_A / n
    p2 = count_B / n

    # Calculating gini
    gini = 1 - (p1 ** 2 + p2 ** 2)

    # Returning the gini impurity
    return gini


def info_gain(data_pd: pd.DataFrame, class_var: str, feature: str) -> float:
    """
    Calculates how much info we gain from a split compared to info at the current node
    """
    # compute the base gini impurity (at the current node)
    gini_base = gini_index(data_pd, class_var)

    # split on the feature
    node_left, node_right = split_bool(data_pd, feature)

    # count datapoints in each split and the whole dataset
    n_left = node_left.shape[0]
    n_right = node_left.shape[0]
    n = n_left + n_right

    # get left and right gini index
    gini_left = gini_index(node_left, class_var)
    gini_right = gini_index(node_right, class_var)

    # calculate weight for each node
    # according to proportion of data it contains from parent node
    w_left = n_left / n
    w_right = n_right / n

    # calculated weighted gini index
    w_gini = w_left * gini_left + w_right * gini_right

    # calculate the gain of this split
    gini_gain = gini_base - w_gini

    # return the best feature
    return gini_gain


def split_bool(data_pd, column_name):
    """Returns two pandas dataframes:
    one where the specified variable is true,
    and the other where the specified variable is false"""
    node_left = data_pd[data_pd[column_name] == True]
    node_right = data_pd[data_pd[column_name] == False]

    return node_left, node_right


def best_split(data_pd: pd.DataFrame, class_var: str, exclude_features: list = []) -> float:
    """
    Returns the name of the best feature to split on at this node.
    If the current node contains the most info (all splits lose information), return None.
    EXCLUDE_FEATURES is the list of variables we want to omit from our list of choices
    """
    # compute the base gini index (at the current node)
    gini_base = gini_index(data_pd, class_var)

    # initialize max_gain and best_feature
    max_gain = 0
    best_feature = None

    # create list of features of data_pd not including class_var
    #features = list(set(data_pd.columns).difference(set(class_var)))
    #features = [f for f in np.array(data_pd.columns) if f not in sexclude_fratures]
    features = [f for f in np.array(
        data_pd.columns) if f not in np.array(class_var)]

    # This line will be useful later - can skip for now
    # remove features we're excluding
    # (already made decision on this feature)
    features = [f for f in features if f not in exclude_features]

    # test a split on each feature
    for ft in features:
        info = info_gain(data_pd, ft, class_var)

        # check whether this is the greatest gain we've seen so far
        # and thus the best split we've seen so far
        if info > max_gain:
            best_feature = ft
            max_gain = info

    # return the best feature
    return best_feature


def build_decision_tree(node_data: pd.DataFrame, class_var: str, depth: int = 0, exclude_features: list = []) -> None:
    """Build a decision tree for NODE_DATA with
    CLASS_VAR as the variable that stores the class assignments.
    EXCLUDE_FEATURES is the list of variables we want to omit from our list of choices"""
    # 0. stop at the base case
    max_depth = 2
    if depth >= max_depth:
        return

    # 1. determine which decision gives us the most information
    best_feature = best_split(node_data, class_var, exclude_features)
    print(
        f"{'>'*(depth+1)}Splitting {node_data.shape[0]} data points on {best_feature}")

    # 2a. if best_feature == None, don't split further
    if best_feature == None:
        print(f"{'>'*(depth+1)}No best next split.")
        return

    # 2b. else, make the split according to the best decision
    else:
        # node_data[node_data[best_feature]]
        data_left, data_right = split_bool(node_data, best_feature)
        print(
            f"{'>'*(depth+1)}Produces {data_left.shape[0]} True data points and {data_right.shape[0]} False data points")

        # and exclude this feature at future levels of the tree
        exclude_features.append(best_feature)

    # 3. continue recursively on each of the resulting two nodes
    build_decision_tree(data_left, class_var, depth + 1, exclude_features)
    build_decision_tree(data_right, class_var, depth + 1, exclude_features)
    return
