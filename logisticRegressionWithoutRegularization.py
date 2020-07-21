"""
    This program implements logistic regression without regularization.
    By: Rahul Golhar
"""
import numpy
import matplotlib.pyplot as plt
from ipython_genutils.py3compat import xrange
from scipy.special import expit
from scipy import optimize


# It takes the cost function and minimizes with the "downhill simplex algorithm."
# http://docs.scipy.org/doc/scipy-0.16.0/reference/generated/scipy.optimize.fmin.html
# Thanks to David Kaleko for this: https://github.com/kaleko/CourseraML/blob/master/ex2/ex2.ipynb
def optimizeTheta(theta, X, y, regTermLambda=0.):
    """
        This function optimizes the cost function value to
        give theta and minimum cost value.
    :param theta:           theta to be used
    :param X:               X matrix
    :param y:               y vector
    :param regTermLambda:   regularization term(= 0 by default)
    :return:                optimal theta values and minimum cost value
    """
    print("\n\t==================> Optimizing cost function <==================\n")
    result = optimize.fmin(costFunction, x0=theta, args=(X, y, regTermLambda), maxiter=400, full_output=True)
    print("\n\t==================> Done with optimizing cost function <==================")

    return result[0], result[1]


def h(theta, X):
    """
        This function returns the hypothesis value. (x*theta)
    :param theta:   the theta vector
    :param X:       the X matrix
    :return:        hypothesis value vector
    """
    return expit(numpy.dot(X,theta))


def costFunction(theta, X, y, regTermLambda = 0.0):
    """
        This function returns the values calculated by the
        cost function.
    :param theta:   theta vector to be used
    :param X:               X matrix
    :param y:               y Vector
    :param regTermLambda:   regularization factor
    :return:                Cost Function results
    """
    m = y.size
    y_log_hx = numpy.dot(-numpy.array(y).T, numpy.log(h(theta, X)))
    one_y_log_one_hx = numpy.dot((1 - numpy.array(y)).T, numpy.log(1 - h(theta, X)))

    # only for j>=1
    regTerm = (regTermLambda / 2) * numpy.sum(numpy.dot(theta[1:].T, theta[1:]))

    return float((1. / m) * (numpy.sum(y_log_hx - one_y_log_one_hx) + regTerm))


def plotInitialData(positiveExamples, negativeExamples):
    """
        This function plots the initial data and saves it.
    :param positiveExamples:    positive examples
    :param negativeExamples:    negative examples
    :return:    None
    """
    print("\n\tPlotting the initial data.")
    plt.figure(figsize=(10, 6))
    plt.grid(True)
    plt.plot(positiveExamples[:, 1], positiveExamples[:, 2], 'k+', label='Admitted')
    plt.plot(negativeExamples[:, 1], negativeExamples[:, 2], 'yo', label='Not Admitted')
    plt.title('Initial data')
    plt.xlabel('Score for exam 1')
    plt.ylabel('Score for exam 2')
    plt.legend()
    plt.savefig("initialDataPlot.jpg")
    print("\tSaved the initial data plotted to initialDataPlot.jpg.")


def plotDecisionBoundary(X, theta, positiveExamples, negativeExamples):
    """
        This function plots the decision boundary.
    :param X:                   X matrix
    :param theta:               calculated theta value
    :param positiveExamples:    list of examples with y=1
    :param negativeExamples:    list of examples with y=0
    :return:    None
    """
    # We draw line between 2 points where hx = 0
    # and theta_0 + theta_1 * x_1 + theta_2 * x_2 = 0
    # The equation y=mx+b is replaced by x_2 = (-1/theta_2)(theta_0 + theta_1*x_1)

    print("\n\tPlotting the decision boundary of data.")
    xExtremes = numpy.array([numpy.min(X[:, 1]), numpy.max(X[:, 1])])
    yExtremes = (-1. / theta[2]) * (theta[0] + theta[1] * xExtremes)

    plt.figure(figsize=(10, 6))

    plt.plot(positiveExamples[:, 1], positiveExamples[:, 2], 'k+', label='Admitted')
    plt.plot(negativeExamples[:, 1], negativeExamples[:, 2], 'yo', label='Not Admitted')
    plt.plot(xExtremes, yExtremes, 'b-', label='Decision Boundary')

    plt.title('Decision Boundary')
    plt.xlabel('Score for exam 1')
    plt.ylabel('Score for exam 2')
    plt.legend()
    plt.grid(True)
    plt.savefig("decisionBoundary.jpg")
    print("\tSaved the graph with decision boundary to decisionBoundary.jpg.")


def predictValue(X, theta):
    """
        This function returns the predicted value
        using theta values calculated.
    :param X:    X vector
    :param theta:   theta vector
    :return:        predicted value
    """
    return h(theta,X) >= 0.5


def main():
    """
        This is the main function.
    :return: None
    """
    print("******************* Starting execution **********************")

    # Read the data
    data = numpy.loadtxt('data/examScores.txt', delimiter=',', usecols=(0, 1, 2), unpack=True)

    print("\nSuccessfully read the data.")

    # ***************************************** Step 1: Initial data *****************************************
    print("Getting the data ready.")

    # X matrix
    X = numpy.transpose(numpy.array(data[:-1]))
    # y vector
    y = numpy.transpose(numpy.array(data[-1:]))
    # no of training examples
    m = y.size
    # Insert a column of 1's into the X matrix
    X = numpy.insert(X, 0, 1, axis=1)

    # Divide the sample into two: ones with positive classification, one with negative classification
    positiveExamples = numpy.array([X[i] for i in xrange(X.shape[0]) if y[i] == 1])
    negativeExamples = numpy.array([X[i] for i in xrange(X.shape[0]) if y[i] == 0])

    # plot initial data on screen
    plotInitialData(positiveExamples,negativeExamples)

    # ***************************************** Step 2: Optimize the cost function *****************************************

    # For theta = zeros the cost function returns the value around 0.693
    initial_theta = numpy.zeros((X.shape[1], 1))

    print("\n\tResult of cost function with theta = 0:(0.693) ",costFunction(initial_theta, X, y))

    theta, minimumCost = optimizeTheta(initial_theta, X, y)

    # plot the decision boundary
    plotDecisionBoundary(X, theta, positiveExamples, negativeExamples)

    # ***************************************** Step 3: Results *****************************************

    print("\n\t __________________________ Results __________________________")

    print("\n\tOptimal theta values: ", theta)
    print("\tMinimum cost value: ", minimumCost)

    testSet = numpy.array([1, 45., 85.])
    res = h(theta, testSet )

    print("\n\tFor a student having 45 and 85 in first and second exams respectively,")
    print("\t The hypothesis value will be: ",res)
    print("\t The student will be admitted: ",(res>=0.5))

    testSet = numpy.array([1, 55., 65.])
    res = h(theta, testSet )

    print("\n\tFor a student having 55 and 65 in first and second exams respectively,")
    print("\t The hypothesis value will be: ",res)
    print("\t The student will be admitted: ",(res>=0.5))

    # ***************************************** Step 4: Performance measure *****************************************

    # Compute accuracy on training set
    positiveCorrect = float(numpy.sum(predictValue(positiveExamples,theta)))
    negativeCorrect = float(numpy.sum(numpy.invert(predictValue(negativeExamples,theta))))
    totalExamples = len(positiveExamples) + len(negativeExamples)
    percentageCorrect = float(positiveCorrect + negativeCorrect) / totalExamples
    print("\n\tPercentage of correctly predicted training examples: %f" % percentageCorrect)

    print("\n******************* Exiting **********************")


if __name__ == '__main__':
    main()