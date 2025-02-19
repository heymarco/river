# imports
from __future__ import annotations

from abc import ABC

from river.datasets import base


class ChangePointDataset(base.FileDataset, ABC):
    """Base class for change point datasets that are stored in a local file.

    Datasets that are part of the Alan Turing Institute Change Point Detection project.

    Parameters
    ----------
    annotations
        The file's associated annotations.
    desc
        Extra dataset parameters to pass as keyword arguments.

    """

    def __init__(self, annotations, **desc):
        super().__init__(**desc)
        self._annotations = annotations

    @property
    def annotations(self):
        """Returns the annotations of the dataset"""
        return self._annotations

    def annotations_aggregated(self, annotator_aggregation):
        """The function `annotations_aggregated` takes an annotator aggregation method as input and returns
        the aggregated annotations based on that method.

        Parameters
        ----------
        annotator_aggregation
            The parameter "annotator_aggregation" is a string that specifies the method of aggregating
        annotations from different annotators. It can take one of the following values:

        Returns
        -------
            the aggregated annotations based on the specified annotator aggregation method.

        """
        if annotator_aggregation == "union":
            annotations = set()
            for annotator in self._annotations:
                annotations.update(self._annotations[annotator])
        elif annotator_aggregation == "intersection":
            annotations = set(self._annotations[0])
            for annotator in self._annotations:
                annotations.intersection_update(self._annotations[annotator])
        elif annotator_aggregation == "majority":
            annotations = {}
            for annotator in self._annotations:
                for change_point in self._annotations[annotator]:
                    if change_point in annotations:
                        annotations[change_point] += 1
                    else:
                        annotations[change_point] = 1
            annotations = {
                change_point
                for change_point, count in annotations.items()
                if count > len(self._annotations) / 2
            }
        else:
            raise ValueError("Unknown annotator aggregation method.")
        return annotations

    def __repr__(self):
        return f"{self.__class__.__name__}"
