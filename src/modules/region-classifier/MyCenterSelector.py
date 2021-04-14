# Class used to perform Nystrom centers initialization

class MyCenterSelector():
    def __init__(self, center_indices):
        self.center_indices = center_indices

    def select(self, X, Y):
        X_to_return = X[self.center_indices, :]
        if len(X_to_return.size()) > 2:
            X_to_return = X_to_return.squeeze()

        if Y is None:
            return X_to_return
        else:
            return X_to_return, Y[self.center_indices, :]
