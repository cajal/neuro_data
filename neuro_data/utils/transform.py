
class DataTransform:
    def __repr__(self):
        return self.__class__.__name__

    def column_transform(self, label):
        return label

