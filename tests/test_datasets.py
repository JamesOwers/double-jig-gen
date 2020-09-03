from double_jig_gen.datasets import ABCDataset


def test_abcdataset_getitem():
    dataset = ABCDataset()
    dataset[len(dataset)]
