import numpy as np

from linear_model import LinearModel


def train_softmax(x_train, y_train):
    """ Train softmax regression parameters (one logistic model per class) """
    mods = []
    for ii in range(y_train.max() + 1):
        if np.mean(y_train == ii) > 0.005:
            print(f"Training class {ii} ...", flush=True)
            mods.append(train_logreg(x_train, y_train == ii))
        else:
            print(f"Skipping class {ii} (under 0.5% of pixels)", flush=True)
            mods.append(None)
    return mods


def eval_softmax(mods, x_eval):
    """ Evaluate softmax data based on softmax model (list of linear models) """
    probs = np.zeros((x_eval.shape[0], len(mods)))
    for ii, mod in enumerate(mods):
        if mod is not None:
            probs[:, ii] = mod.predict(x_eval)
    return np.argmax(forward_softmax(probs), axis=1)


def train_logreg(x_train, y_train):
    """Logistic regression with Newton's Method.

    Args:
        x_train: training x data
        y_train: training y data
    """
    n = x_train.shape[1]

    theta_0 = np.zeros((n, 1))
    linmod = LogisticRegression(theta_0=theta_0)
    linmod.fit(x_train, y_train)

    return linmod


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
        tol = 1e-3
        max_newton_iters = 25   # safety cap: the original loop had no limit
        dtheta = np.ones(self.theta.shape)
        it = 0
        while not all(np.absolute(dtheta / self.theta) < tol) and it < max_newton_iters:
            m = x.shape[0]
            h_theta = sigmoid(x @ self.theta)
            hessian = x.T @ (x * h_theta * (1 - h_theta.T).T) / m
            grad = ((y - h_theta.T) @ x).T / m
            try:
                dtheta = np.linalg.inv(hessian) @ grad
            except np.linalg.LinAlgError:
                # Hessian singular (e.g. saturated sigmoid): use pseudo-inverse
                dtheta = np.linalg.pinv(hessian) @ grad
            self.theta = self.theta + dtheta
            it += 1
            print(f"   Newton iteration {it}", flush=True)
        return

    def predict(self, x):
        """Make a prediction given new inputs x.

        Args:
            x: Inputs of shape (m, n).

        Returns:
            Outputs of shape (m,).
        """
        # Vectorised: same result as the original per-pixel loop, much faster
        return sigmoid(x @ self.theta).ravel()


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def forward_softmax(x):
    """
    Compute softmax function for a single example.
    The shape of the input is of size # num classes.

    Important Note: You must be careful to avoid overflow for this function. Functions
    like softmax have a tendency to overflow when very large numbers like e^10000 are computed.
    You will know that your function is overflow resistent when it can handle input like:
    np.array([[10000, 10010, 10]]) without issues.

    Args:
        x: A 1d numpy float array of shape number_of_classes

    Returns:
        A 1d numpy float array containing the softmax results of shape  number_of_classes
    """
    x = x - np.max(x, axis=0)
    exp = np.exp(x)
    s = exp / np.sum(exp, axis=0)
    return s