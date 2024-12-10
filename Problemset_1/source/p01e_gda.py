import numpy as np
import util

from linear_model import LinearModel
from sklearn.metrics import accuracy_score

def main(train_path, eval_path, pred_path):
    """Problem 1(e): Gaussian discriminant analysis (GDA)

    Args:
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
        pred_path: Path to save predictions.
    """

    #we only add intercept when predicting, not training

    # Load dataset
    x_train, y_train = util.load_dataset(train_path, add_intercept=False)

    # *** START CODE HERE ***

    #train
    model = GDA()
    model.fit(x_train, y_train)

    #load
    util.plot(x_train, y_train, model.theta, 'output/p01e_{}.png'.format(pred_path[-5]))

    #save preds
    x_eval, y_eval = util.load_dataset(eval_path, add_intercept=True)
    y_pred = model.predict(x_eval)
    np.savetxt(pred_path, y_pred > 0.5, fmt='%d')

    y_pred2 = model.predictBin(x_eval)

    print(f"Accuracy of GDA: ", accuracy_score(y_pred2, y_eval))

    # *** END CODE HERE ***


class GDA(LinearModel):
    """Gaussian Discriminant Analysis.

    Example usage:
        > clf = GDA()
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """

    def fit(self, x, y):
        """Fit a GDA model to training set given by x and y.

        Args:
            x: Training example inputs. Shape (m, n).
            y: Training example labels. Shape (m,).

        Returns:
            theta: GDA model parameters.
        """
        # *** START CODE HERE ***
        m,n = x.shape

        #parameters
        self.theta = np.zeros(n+1)

        phi = 0
        muy0 = np.zeros(n) # n x 1
        muy1 = np.zeros(n) # n x 1
        sigma = np.zeros((n,n))

        num0 = 0 
        num1 = 0
        #innit phi and muy1
        for i in range(0,m):
            if y[i]==1:
                num1+=1
                muy1+=x[i] #vector sum
            else:
                num0+=1
                muy0+=x[i]
        
        phi = num1/m
        muy0 = muy0/num0
        muy1= muy1/num1

        #print("muy0 = ", muy0)

        #sigma
        for i in range (0,m):
            if(y[i]==0):
                # .dot for np arrays will be confused as dot product returning real number
                # since 1D arrays are shaped (n,) not (n,1). outer prod fixes this
                sigma += np.outer(x[i]-muy0, (x[i]-muy0).T)
            else:
                sigma += np.outer(x[i]-muy1, (x[i]-muy1).T)
        
        sigma= sigma/m

        #print("sigma = ", sigma)

        #parameters

        invsigma = np.linalg.inv(sigma)

        self.theta[0] = 0.5 * (muy0 + muy1).dot(invsigma).dot(muy0 - muy1) - np.log((1 - phi) / phi)
        self.theta[1:] = invsigma.dot(muy1 - muy0)

        # (theta)^Tx + theta0 can be replaced by (theta)^Tx with concatenated x[0]=1
        # and theta[0] = intercept = 1

        #we only add intercept when predicting, not training


        # *** END CODE HERE ***

    def predict(self, x):
        """Make a prediction given new inputs x.

        Args:
            x: Inputs of shape (m, n).

        Returns:
            Outputs of shape (m,).
        """
        # *** START CODE HERE ***
        linear_theta = x.dot(self.theta)
        return 1/(1 + np.exp(-linear_theta))
    
    def predictBin(self, x):
        return self.predict(x) > 0.5
        # *** END CODE HERE
