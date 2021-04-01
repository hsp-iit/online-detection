# Class used to perform Nystrom centers initialization

class MyCenterSelector():
    def __init__(self, center_indices):
        self.center_indices = center_indices
    def select(self, X, Y, M):
        # Check M == len(center_indices)
        if M != len(self.center_indices):
            raise ValueError("Predefined centers are not `M` (found %d expected %d)" %
                             (len(self.center_indices), M))

        X_to_return = X[self.center_indices, :]
        if len(X_to_return.size()) > 2:
            X_to_return = X_to_return.squeeze()

        if Y is None:
            return X_to_return
        else:
            return X_to_return, Y[self.center_indices, :]
