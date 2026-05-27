import time
from dataset_h5 import GMIPatchH5Dataset, make_dataloader
#from dataset import GMIPatchDataset, make_dataloader

ds = GMIPatchH5Dataset("/lustre/home/mariyam/GMI_1C-R/h5_subsets/train_subset.h5", use_static=True)
#ds = GMIPatchDataset("/lustre/home/mariyam/GMI_1C-R/patches_split_byday_cnnprep", use_static=True)
loader = make_dataloader(ds, batch_size=16, shuffle=True, num_workers=2, pin_memory=True)

t0 = time.perf_counter()
for i, (x, y) in enumerate(loader):
    t1 = time.perf_counter()
    print(i, "batch loaded in", t1 - t0, "sec", x.shape, y.shape)
    t0 = time.perf_counter()
    if i == 9:
        break