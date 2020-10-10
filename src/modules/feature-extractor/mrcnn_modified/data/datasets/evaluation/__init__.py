from maskrcnn_benchmark.data import datasets as datasets_orig
from mrcnn_modified.data import datasets as datasets_custom

from maskrcnn_benchmark.data.datasets.evaluation.coco import coco_evaluation
from maskrcnn_benchmark.data.datasets.evaluation.voc import voc_evaluation
from mrcnn_modified.data.datasets.evaluation.icubworld import icw_evaluation
from mrcnn_modified.data.datasets.evaluation.ycbv import ycbv_evaluation

def evaluate(dataset, predictions, output_folder, **kwargs):
    """evaluate dataset using different methods based on dataset type.
    Args:
        dataset: Dataset object
        predictions(list[BoxList]): each item in the list represents the
            prediction results for one image.
        output_folder: output folder, to save evaluation files or results.
        **kwargs: other args.
    Returns:
        evaluation result
    """
    args = dict(
        dataset=dataset, predictions=predictions, output_folder=output_folder, **kwargs
    )
    if isinstance(dataset, datasets_orig.COCODataset):
        return coco_evaluation(**args)
    elif isinstance(dataset, datasets_orig.PascalVOCDataset):
        return voc_evaluation(**args)
    elif isinstance(dataset, datasets_custom.iCubWorldDataset):
        return icw_evaluation(**args)
    elif isinstance(dataset, datasets_custom.YCBVideoDataset):
        return ycbv_evaluation(**args)
    else:
        dataset_name = dataset.__class__.__name__
        raise NotImplementedError("Unsupported dataset type {}.".format(dataset_name))
