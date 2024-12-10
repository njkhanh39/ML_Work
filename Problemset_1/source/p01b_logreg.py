import numpy as np
import util

from linear_model import LinearModel
from sklearn.metrics import accuracy_score

def main(train_path, eval_path, pred_path):
    """Problem 1(b): Logistic regression with Newton's Method.

    Args:
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
        pred_path: Path to save predictions.
    """
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)

    # *** START CODE HERE ***

    ##i steal this one from github cs229 :>

     # Train logistic regression
    model = LogisticRegression(eps=1e-5)
    model.fit(x_train, y_train)

    # Plot data and decision boundary
    util.plot(x_train, y_train, model.theta, 'output/p01b_{}.png'.format(pred_path[-5]))

    # Save predictions
    x_eval, y_eval = util.load_dataset(eval_path, add_intercept=True)
    y_pred = model.predict(x_eval)
    np.savetxt(pred_path, y_pred > 0.5, fmt='%d')

    y_pred2 = model.predictBin(x_eval)

    print(f"Accuracy of Newton method Logistic Regression of dataset: ", accuracy_score(y_pred2, y_eval))

    # *** END CODE HERE ***


class LogisticRegression(LinearModel):
    """Logistic regression with Newton's Method as the solver.

    Example usage:
        > clf = LogisticRegression()
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """

    def fit(self, x, y):
        """Run Newton's Method to minimize J(theta) for logistic regression.

        Args:
            x: Training example inputs. Shape (m, n).
            y: Training example labels. Shape (m,).
        """
        # *** START CODE HERE ***

        #innit theta
        m,n = x.shape

        self.theta = np.zeros(n)

        #train
        while(True):
            #theta is progressively updated
            old_theta = self.theta

            h_theta = self.predict(x)

            #calc gradient and hessian

            gradient = np.dot(x.T, (y-h_theta))
            hessian = np.dot(-x.T*h_theta*(1-h_theta), x)

            #update theta
            self.theta = self.theta - np.dot(np.linalg.inv(hessian),gradient)

            #check for stopping
            if np.linalg.norm(self.theta-old_theta, ord=1) < self.eps:
                break


        # *** END CODE HERE ***

    def predict(self, x):
        """Make a prediction given new inputs x.

        Args:
            x: Inputs of shape (m, n).

        Returns:
            Outputs of shape (m,).
        """
        # *** START CODE HERE ***
        linear_theta = np.dot(x,self.theta)
        return 1/(1 + np.exp(-linear_theta))
    
    def predictBin(self, x):
        return self.predict(x) > 0.5
        # *** END CODE HERE ***