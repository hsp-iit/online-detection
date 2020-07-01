class MyCenterSelector():
    def __init__(self, center_indices):
        self.center_indices = center_indices
    def select(self, X, Y, M):
        """
        Ignore Y (only useful for LogisticFalkon)
        Ignore M (since we want to use center_indices), but check M == len(center_indices)
        to avoid mistakes.
        """
        if M != len(self.center_indices):
            raise ValueError("Predefined centers are not `M` (found %d expected %d)" %
                             (len(self.center_indices), M))
        # Ignore possibility to get sparse tensors, or F-contiguous tensors
        return X[self.center_indices, :].squeeze(), Y[self.center_indices, :]
