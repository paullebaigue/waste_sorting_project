from sklearn import svm, metrics  # model_selection, neural_network, neighbors,  cluster, preprocessing
from sklearn.model_selection import KFold
import numpy as np
from sklearn.metrics import hinge_loss


def print_SVM_results(SVM_model, X_test, Y_test, kernel='linear', C=1, gamma=None, print_all=True):
    """ display the result of the training tested on the test-set """

    # prediction on the test set
    Y_test_pred = SVM_model.predict(X_test)

    # computing the confusion matrix
    confusion_matrix = metrics.confusion_matrix(Y_test, Y_test_pred)

    # score from confusion matrix
    sum_diag = np.sum(np.diag(confusion_matrix))
    sum_all = np.sum(confusion_matrix, axis=None)
    score = sum_diag/sum_all

    # 2 cases if gamma is self-determined or not
    if print_all:
        if gamma is None:
            print("score SVM %s:    %.3f , \n C = %.4f" % (kernel, score, C))
        else:
            print("score SVM %s:    %.3f , \n C = %.4f, gamma = %.4f" % (kernel, score, C, gamma))

        # printing the confusion matrix
        print('classes', SVM_model.classes_)
        print('\n confusion matrix: \n', confusion_matrix, '\n')

    return score, confusion_matrix


def learn_SVM(X_train, Y_train, X_test, Y_test,
              kernel='linear', C=1, gamma=None, print_result=False, print_all=True):
    """ train an SVM model from extracted characteristics of a signal
    INPUT : listes train/test
    OUTPUT : training score, confusion matrix
    """

    # initialization of the SVM model
    # https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html
    if gamma is not None:
        SVM_model = svm.SVC(kernel=kernel, C=C, gamma=gamma, class_weight='balanced') # {0:0.99, 1:0.01})  #'balanced')
    else:
        SVM_model = svm.SVC(kernel=kernel, C=C, class_weight='balanced')

    # training the model
    SVM_model.fit(X_train, Y_train)
    pred_decision = SVM_model.decision_function(X_test)
    loss = hinge_loss(y_true=Y_test, pred_decision=pred_decision)

    assert print_result in [True, False]
    # if True, display the score of the trained model on the test-set and the confusion matrix
    if print_result:
        score, confusion_matrix = print_SVM_results(SVM_model, X_test, Y_test,
                                                    kernel=kernel, C=C, gamma=gamma, print_all=print_all)
    else:
        # testing the model
        score = SVM_model.score(X_test, Y_test)

        confusion_matrix = None

    return SVM_model, loss, score, confusion_matrix


# TODO: parloop
def cross_validation(X, y, n_fold=5,
                     kernel='linear', C=1, gamma=None,
                     print_result=True, print_all=False):
    """ A function that aims at rating a set of parameters or a model
        Using cross validation is paramount to avoid special case results and help to reduce misinterpretation
        It also detects over-fitting in some specific cases
    """

    # empty list which are going to be filled with the results of the K models
    loss_list = []
    score_list = []
    conf_matrix_list = []

    Folds = KFold(n_splits=n_fold)

    # creates K different model with K distinct test-set for the cross validation
    for train_index, test_index in Folds.split(X):

        # Splits train and test for the folder number n
        X_train, X_valid = X[train_index], X[test_index]
        y_train, y_valid = y[train_index], y[test_index]

        # train the n-th model
        SVM_model, loss, score, confusion_matrix = learn_SVM(X_train, y_train, X_valid, y_valid,
                                                             kernel=kernel, C=C, gamma=gamma,
                                                             print_result=print_result, print_all=print_all)

        # the results of the n-th model
        loss_list.append(loss)
        score_list.append(score)
        conf_matrix_list.append(confusion_matrix)

    # statistics on the K models
    mean_loss = np.mean(loss_list)
    mean_score = np.mean(score_list)
    std_score = np.std(score_list)

    # create the confusion matrix instance
    if print_result:
        total_confusion_matrix = np.sum(conf_matrix_list, axis=0)
    else:
        total_confusion_matrix = None

    return mean_loss, mean_score, std_score, total_confusion_matrix
