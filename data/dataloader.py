import torch

from torchvision.datasets import ImageNet, Omniglot
from torchvision.transforms import ToTensor, Resize, Compose
from torch.utils.data import Dataset, DataLoader


class OmniglotDataset(Dataset):

    def __init__(
            self,
            n=1,
            m=1,
            device='cpu',
            background=True,
    ):
        super(OmniglotDataset).__init__()
        self.device = device
        self.n = n
        self.m = m
        self.ds = Omniglot(
            'datasets/omniglot',
            background=background,
            download=True,
            transform=Compose([Resize(28), ToTensor()])
        )

    def __len__(self):
        return int(len(self.ds) / 20)

    # grabs respective number of images in one class for the support and query set. All images are from the same class per call to __getitem__
    def __getitem__(self, i):
        a = i * 20
        b = a + 20
        x = torch.cat([self.ds[j][0].unsqueeze(0)
                      for j in range(a, b)]).to(self.device)
        mask = torch.randperm(20).to(self.device)
        # returns: (support_set, query_set), target_label of images of this class
        return (x[mask[0:self.n]], x[mask[self.n:self.n + self.m]]), i


class ImageNetDataLoader:

    def __init__(
        self,
        k=20,
        n=1,
        m=1,
        phase='train',
        device='cpu'
    ):
        self.phase = phase
        self.device = device
        self.k = k
        self.n = n
        self.m = m
        self.dl = MiniImageNetDataLoader(
            shot_num=n,
            way_num=k,
            episode_test_sample_num=m
        )
        self.dl.generate_data_list(phase=phase)
        self.dl.load_list(phase=phase)
        self.len = self.dl.getLength(phase=phase)

    def __len__(self):
        # if self.phase == 'train':
        #     length = int((64 * 600) / self.k)
        # if self.phase == 'test':
        #     length = int((20 * 600) / self.k)
        return self.len

    def __getitem__(self, i):
        ss, sl, ts, tl = self.dl.get_batch(phase=self.phase, idx=i)

        ss = torch.tensor(ss).view(self.k, self.n, 3,
                                   84, 84).float().to(self.device)
        sl = torch.tensor(sl).float().to(self.device)
        ts = torch.tensor(ts).view(self.k, self.m, 3,
                                   84, 84).float().to(self.device)
        tl = torch.tensor(tl).float().to(self.device)
        return (ss, sl), (ts, tl)


class Siamese(Dataset):

    def __init__(
            self,
            *datasets,
    ):
        super(Siamese).__init__()
        self.ds = [d for d in datasets]

    def __len__(self):
        return len(self.ds[0])

    def __getitem__(self, i):
        return [ds[i % len(ds)] for ds in self.ds]


if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # train_data = OmniglotDataset(background=True, device=device)
    # train_dataloader = DataLoader(train_data, batch_size=64, shuffle=True)
    # # print(len(train_dataloader.ds), train_dataloader.ds.batch_size)
    # print(f'Batch Size: {train_dataloader.batch_size}')
    # print(
    #     f'Dataloader Length = 964/{train_dataloader.batch_size}: {len(train_dataloader)}')
    # for X, y in train_dataloader:
    #     print(f'Shape of X and dtype: {X.shape}, {X.dtype}')
    #     print(f'Shape of y and dtype: {y.shape}, {y.dtype}')
    #     break

    # train_ds = OmniglotDataset(support_num_exam_per_class=3,
    #                            query_num_exam_per_class=1, background=True, device=device)
    # test_ds = OmniglotDataset(support_num_exam_per_class=1,
    #                           query_num_exam_per_class=3, background=False, device=device)

    k = 3
    n = 2
    m = 4

    train_ds = ImageNetDataLoader(k=k, n=n, m=m, phase='train', device=device)
    test_ds = ImageNetDataLoader(k=k, n=n, m=m, phase='test', device=device)

    ds = Siamese(train_ds, test_ds)

    dl = DataLoader(ds, batch_size=1, shuffle=True, drop_last=True)

    print("Train Dataset Length: ", len(train_ds))
    print("Test Dataset Length: ", len(test_ds))
    print("Dataloader Length: ", len(dl))

    for i, a in enumerate(dl):
        # a[0] is the training dataset with a length of 2
        # a[0][0] is a list containing the support and query set of this batch, a[0][1] are the classes only in this batch
        # a[1] is the testing dataset

        # if i == 0:
        #     print(
        #         f'a[0][0] is a list containing the support and query set of this batch')
        #     # grabs a list of length 2 containing the support and query set
        #     print(a[0][0])
        #     print()
        #     print(f'a[0][0][0] is the support set')  # grabs the support set
        #     print(a[0][0][0])
        #     print()
        #     print(f'a[0][1] are the classes only in this batch')
        #     # grabs the tensor array containing all classes only in this batch
        #     print(a[0][1])
        #     # print(a[0][1][0]) # grabs first class of this batch
        #     print()
        #     classes = a[0][1].numpy()
        #     target_indices = np.array(range(len(a[0][1])))
        #     class_to_index = dict(zip(classes, target_indices))
        #     print(class_to_index)
        #     print(classes[1])
        #     print(class_to_index.get(classes[1]))
        #     print()

        print(
            f'{i:<5}',
            # (batch_size=num_classes, shots=num_examples_per_class, num_channels_per_image=1, 28, 28)
            # a[0][0][0].shape,  # support set
            # a[0][0][1].shape,  # query set
            # # support and query set have the same size, both have K classes w/ N examples per class
            # # a[0][1],
            # a[1][0][0].shape,
            # a[1][0][1].shape,
            # a[1][1],
            a[0][0][0].shape,   # Train Support Set
            a[0][1][0].shape,   # Train Query Set
            a[1][0][0].shape,   # Test Support Set
            a[1][1][0].shape,   # Test Query Set
        )
