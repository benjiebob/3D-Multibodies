from datasets.mixed_dataset import MixedDataset
from datasets.base_dataset import BaseDataset
from argparse import Namespace
from utils import spin_options


def dataset_zoo(
    dataset_name="mixed+static+p2",
    use_augmentation=True,
    run_mini=False,
    ambiguous=False,
    length_divider=1.0,
    return_img_orig=False,
    ignore_img=False,
    TRAIN={"rand_sample": -1, "limit_to": -1},
    VAL={"rand_sample": -1, "limit_to": -1},
    TEST={"rand_sample": -1, "limit_to": -1},
    **kwargs
):

    spin_trainoptions = spin_options.SPIN_TRAINOPTIONS
    if dataset_name == "mixed":
        dataset_train = MixedDataset(
            spin_trainoptions,
            is_train=True,
            ambiguous=ambiguous,
            run_mini=run_mini,
            length_divider=length_divider,
            return_img_orig=return_img_orig,
        )
        dataset_val = BaseDataset(
            spin_trainoptions,
            "h36m-p1-corr",
            dataset_key=0,
            is_train=False,
            ambiguous=ambiguous,
            run_mini=run_mini,
            return_img_orig=return_img_orig,
        )
        dataset_test = dataset_val
    elif dataset_name == "mixed+static":
        dataset_train = MixedDataset(
            spin_trainoptions,
            is_train=True,
            ambiguous=ambiguous,
            allow_static_fits=True,
            run_mini=run_mini,
            length_divider=length_divider,
            return_img_orig=return_img_orig,
        )
        dataset_val = BaseDataset(
            spin_trainoptions,
            "h36m-p1-corr",
            dataset_key=0,
            is_train=False,
            ambiguous=ambiguous,
            allow_static_fits=True,
            run_mini=run_mini,
            return_img_orig=return_img_orig,
        )
        dataset_test = dataset_val
    elif dataset_name == "mixed+static+p2":
        dataset_train = MixedDataset(
            spin_trainoptions,
            is_train=True,
            ambiguous=ambiguous,
            allow_static_fits=True,
            run_mini=run_mini,
            length_divider=length_divider,
            return_img_orig=return_img_orig,
            ignore_img=ignore_img,
        )
        dataset_val = None
        dataset_test = {
            "3dpw": BaseDataset(
                spin_trainoptions,
                "3dpw",
                dataset_key=6,
                is_train=False,
                ambiguous=ambiguous,
                allow_static_fits=True,
                run_mini=run_mini,
                return_img_orig=return_img_orig,
                ignore_img=ignore_img,
            ),
            "h36m-p2": BaseDataset(
                spin_trainoptions,
                "h36m-p2",
                dataset_key=0,
                is_train=False,
                ambiguous=ambiguous,
                allow_static_fits=True,
                run_mini=run_mini,
                return_img_orig=return_img_orig,
                ignore_img=ignore_img,
            ),
        }

    elif dataset_name == "all":
        dataset_train = MixedDataset(
            spin_trainoptions,
            is_train=True,
            ambiguous=ambiguous,
            allow_static_fits=True,
            run_mini=run_mini,
            length_divider=length_divider,
            return_img_orig=return_img_orig,
            ignore_img=ignore_img,
        )
        dataset_val = None
        dataset_test = {
            "3dpw": BaseDataset(
                spin_trainoptions,
                "3dpw",
                dataset_key=6,
                is_train=False,
                ambiguous=ambiguous,
                allow_static_fits=True,
                run_mini=run_mini,
                return_img_orig=return_img_orig,
                ignore_img=ignore_img,
            ),
            "h36m-p2": BaseDataset(
                spin_trainoptions,
                "h36m-p2",
                dataset_key=0,
                is_train=False,
                ambiguous=ambiguous,
                allow_static_fits=True,
                run_mini=run_mini,
                return_img_orig=return_img_orig,
                ignore_img=ignore_img,
            ),
            "lsp": BaseDataset(
                spin_trainoptions,
                "lsp",
                dataset_key=0,
                is_train=False,
                ambiguous=ambiguous,
                allow_static_fits=True,
                run_mini=run_mini,
                return_img_orig=return_img_orig,
                ignore_img=ignore_img,
            ),
            "mpi-inf-3dhp": BaseDataset(
                spin_trainoptions,
                "mpi-inf-3dhp",
                dataset_key=0,
                is_train=False,
                ambiguous=ambiguous,
                allow_static_fits=True,
                run_mini=run_mini,
                return_img_orig=return_img_orig,
                ignore_img=ignore_img,
            ),
            "coco": BaseDataset(
                spin_trainoptions,
                "coco",
                dataset_key=0,
                is_train=False,
                ambiguous=ambiguous,
                allow_static_fits=True,
                run_mini=run_mini,
                return_img_orig=return_img_orig,
                ignore_img=ignore_img,
            ),
        }
    elif dataset_name == "h36m_only":
        # dataset_train = BaseDataset(
        #     spin_trainoptions,
        #     "h36m",
        #     dataset_key=0,
        #     is_train=True,
        #     ambiguous=ambiguous,
        #     run_mini=run_mini,
        #     return_img_orig=return_img_orig,
        # )  # TODO: Implement!
        dataset_train = None
        dataset_val = None
        dataset_test = {
            "h36m-p2": BaseDataset(
                spin_trainoptions,
                "h36m-p2",
                dataset_key=0,
                is_train=False,
                ambiguous=ambiguous,
                run_mini=run_mini,
                return_img_orig=return_img_orig,
            )
        }

    elif dataset_name == "3dpw_only":
        # dataset_train = MixedDataset(
        #         spin_trainoptions,
        #         is_train=True,
        #         ambiguous=ambiguous,
        #         allow_static_fits=True,
        #         run_mini=run_mini,
        #         length_divider=length_divider,
        #         return_img_orig=return_img_orig,
        #         ignore_img=ignore_img)
        dataset_train = None
        dataset_val = None
        dataset_test = {
            "3dpw": BaseDataset(
                spin_trainoptions,
                "3dpw",
                dataset_key=6,
                is_train=False,
                ambiguous=ambiguous,
                allow_static_fits=True,
                run_mini=run_mini,
                return_img_orig=return_img_orig,
                ignore_img=ignore_img,
            ),
        }
    else:
        raise ValueError("no such dataset %s" % dataset_name)

    return dataset_train, dataset_val, dataset_test
