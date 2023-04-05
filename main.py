import os
import sys
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader

from dataset import KITTIDataset, KITTITest
from model import Model
import args

PI = 3.1415926535


class Main:
    def __init__(self):
        self.model = Model(args.bbox2d_dim, args.bbox3d_dim)
        self.start_from = 0
        if os.path.exists(args.ckpt):
            print(f'Loading pretrained weights from {args.ckpt}...')
            recordings = torch.load(args.ckpt, map_location='cpu')
            self.model.load_state_dict(recordings['state_dict'])
            self.start_from = recordings['epoch'] + 1
        self.model.to(args.device)
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None

    def train(self):
        if self.train_loader is None:
            train_set = KITTIDataset(args.root, args.img_size, train=True)
            self.train_loader = DataLoader(train_set, args.batch_size, shuffle=True,
                                           num_workers=4, drop_last=False, pin_memory=True)
            print(f'The dataset has {len(train_set)} training data')
        max_aos = 0
        self.model.train()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=args.lr)
        for e in range(self.start_from, args.num_epochs):
            with tqdm(self.train_loader, dynamic_ncols=True) as loader:
                for img, bbox_2d, bbox_3d, orientation in loader:
                    optimizer.zero_grad()
                    img = img.to(args.device)
                    bbox_2d = bbox_2d.to(args.device)
                    bbox_3d = bbox_3d.to(args.device)
                    orientation = orientation.to(args.device)
                    loss, loss_dict = self.model.loss(img, bbox_2d, bbox_3d, orientation,
                                                      weight_3d=args.weight_3d)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), args.grad_clip)
                    optimizer.step()
                    info_dict = {'epoch': e, 'loss': loss.item()}
                    info_dict.update(loss_dict)
                    loader.set_postfix(ordered_dict=info_dict)
                    loader.set_description('Training')
            if e % 5 == 0:
                if not os.path.exists('./assets'):
                    os.makedirs('./assets')
                recordings = {
                    'state_dict': self.model.state_dict(),
                    'epoch': e,
                }
                torch.save(recordings, './assets/ckpt.pth')
                avg_aos = self.val()
                self.model.train()
                if avg_aos > max_aos:
                    max_aos = avg_aos
                    torch.save(recordings, f'./assets/ckpt_{avg_aos}.pth')

    @torch.no_grad()
    def val(self):
        if self.val_loader is None:
            val_set = KITTIDataset(args.root, args.img_size, train=False)
            self.val_loader = DataLoader(val_set, args.batch_size, shuffle=True,
                                         num_workers=4, drop_last=False, pin_memory=True)
            print(f'The dataset has {len(val_set)} validation data')
        self.model.eval()
        total_aos = 0
        with tqdm(self.val_loader, dynamic_ncols=True) as loader:
            for img, bbox_2d, bbox_3d, orientation in loader:
                img = img.to(args.device)
                bbox_2d = bbox_2d.to(args.device)
                orientation = orientation.to(args.device)
                total_aos += self.model.aos(img, bbox_2d, orientation)
                loader.set_description('Validation')
        avg_aos = total_aos / len(val_set)
        print(f'Average Orientation Similarity (AOS): {avg_aos}')
        return avg_aos

    @torch.no_grad()
    def test(self):
        assert not os.path.exists('./outputs'), 'The output folder ./outputs is not empty'
        os.makedirs('./outputs')
        if self.test_loader is None:
            test_set = KITTITest(args.root, args.results_2d_dir, args.img_size)
            print(f'The dataset has {len(test_set)} test data')
        self.model.eval()
        with tqdm(test_set, dynamic_ncols=True) as loader:
            for data in loader:
                img, bbox_2d, results_2d_file, results_2d_row = data
                out_file = f'./outputs/{os.path.basename(results_2d_file)}'
                if not os.path.exists(out_file):
                    os.system(f'cp {results_2d_file} {out_file}')
                if img is not None:
                    img = img.to(args.device)
                    bbox_2d = bbox_2d.to(args.device)
                    orientation = self.model(img, bbox_2d)[1]
                    orientation = torch.clamp(orientation, -PI, PI).cpu().numpy()
                    print(orientation)
                    with open(results_2d_file, 'r') as f:
                        lines = f.readlines()
                        for i in range(img.size(0)):
                            contents = lines[results_2d_row[i]].strip().split(' ')
                            contents[3] = str(orientation[i][0])
                            lines[results_2d_row[i]] = ' '.join(contents) + '\n'
                    with open(out_file, 'w') as f:
                        f.writelines(lines)
                loader.set_description('Testing')


if __name__ == "__main__":
    main = Main()
    assert len(sys.argv) == 2, f'Number of cmd parameters {len(sys.argv) - 1} not supported'
    assert sys.argv[1] in ['train', 'val', 'test'], f'Mode {sys.argv[1]} not supported'
    if sys.argv[1] == 'train':
        main.train()
    elif sys.argv[1] == 'val':
        main.val()
    else:
        main.test()

