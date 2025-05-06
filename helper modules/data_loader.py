import torch, kagglehub as kh, shutil, random, zipfile, torchvision, os, numpy as np, shutil
from pathlib import Path
from PIL import Image
from torch.utils.data import random_split, Dataset, DataLoader
import torchvision.transforms.v2 as T
from itertools import chain
from random import shuffle, sample

gen = torch.Generator().manual_seed(42)

# a pytorch Dataset to hold images
class CellsDataset(Dataset):
  def __init__(self, cells_list:list, transforms:torchvision.transforms.v2.Compose):
    self.ls = cells_list
    self.transforms = transforms

    # get targets
    _targets = list()
    for x in range(len(self.ls)):
      lb = 0 if self.ls[x].parent.name.lower()=='uninfected' else 1
      _targets.append(lb)
    self.targets = np.array(_targets)

  def __len__(self):
    return len(self.ls)

  def __getitem__(self, idx):
    image = self.transforms(Image.open(self.ls[idx]))
    label = self.targets[idx].item()
    return image, label

# define image transforms
_cell_transform = T.Compose([
    T.PILToTensor(),
    T.ToDtype(torch.float32, scale=True),
    T.Resize((128, 128)),
    #imagenet normalization:
    #  * can be gotten from calling model.pretrained_cfg property on
    #    a timm's pretrained model
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# function to download dataset from kaggle, and extract content
def _download_dataset():
  # create download folder
  DOWNLOAD_DIR = Path('malaria')
  DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)

  # download dataset
  cache = kh.dataset_download(handle='iarunava/cell-images-for-detecting-malaria')

  # archive cache
  shutil.make_archive(base_name=DOWNLOAD_DIR/'malaria', format='zip', root_dir=cache)

  # extract zipped file
  with zipfile.ZipFile(file=DOWNLOAD_DIR/'malaria.zip', mode='r') as zipf:
    zipf.extractall(path=DOWNLOAD_DIR)

  # delete duplicate sub-directories
  if Path.cwd().joinpath('malaria/cell_images/cell_images').is_dir():
    pth = Path.cwd().joinpath('malaria/cell_images/cell_images')
    shutil.rmtree(path=pth, ignore_errors=True)

# function to return all image paths as shuffled lists
def _make_cells_list() -> tuple[list, list]:
  # set random seed
  random.seed(0)
  # create paths of infected / uninfected
  parasitized = Path.cwd().joinpath('malaria/cell_images/Parasitized')
  uninfected = Path.cwd().joinpath('malaria/cell_images/Uninfected')

  # lists of infected / parasitized paths
  sick_list = list(parasitized.glob('*.png'))
  healthy_list = list(uninfected.glob('*png'))

  # all cells list
  cell_list = sick_list + healthy_list

  # sample random image paths from parasitized & uninfected directories
  # 80% of random parasitized and unparasitized images will be sampled for training
  # the rest will be split into validation and test datasets
  train_list = chain(
        sample(population=sick_list, k=int(len(sick_list)*0.80)),
        sample(population=healthy_list, k=int(len(sick_list)*0.80)))

  # and make a list of images to be used to training
  train_list = list(train_list)

  # delete the extracted image paths in train_list from cell_list
  for i in train_list:
    if i.is_file():
      cell_list.remove(i)

  # shuffle both lists
  shuffle(cell_list), shuffle(train_list)

  return cell_list, train_list

# function to get dataloaders, and label-mapper dict
def get_dataloaders() -> tuple[DataLoader, DataLoader, DataLoader, dict[int, str]]:
  """
    Splits a dataset into training, validation, and test sets, and creates corresponding DataLoader
    instances for each set. It also returns a dictionary that maps class labels to their respective
    string labels.

    The dataset is split into three subsets with the following proportions:
    - 70% for training
    - 20% for testing
    - 10% for validation

    The resulting DataLoaders are configured with a batch size of 32, use of all available CPU cores
    for parallel data loading, and memory pinning for faster data transfer to the GPU.

    Args:
        None

    Returns:
        tuple: A tuple containing the following elements:
            - train_dl (DataLoader): DataLoader for the training set.
            - test_dl (DataLoader): DataLoader for the test set.
            - val_dl (DataLoader): DataLoader for the validation set.
            - class_mapper (dict): A dictionary that maps integer class labels to string labels,
              where 0 is mapped to 'Uninfected' and 1 is mapped to 'Infected'.

    Example:
        train_dl, test_dl, val_dl, class_mapper = get_dataloaders()
    """
  # download dataset
  _download_dataset()

  # get lists
  cell_list, train_list = _make_cells_list()

  # create a torch Dataset from training images
  tr_set = CellsDataset(cells_list=train_list, transforms=_cell_transform)
  # create a torch Dataset from the rest of images
  val_ts_set = CellsDataset(cells_list=cell_list, transforms=_cell_transform)

  # specify size of train, validation and test sets
  val_size = int(0.75*len(cell_list))
  ts_size = int(len(cell_list) - val_size)

  # test and validation subsets
  ts_set, val_set = random_split(
      dataset=val_ts_set, lengths=[ts_size, val_size], generator=gen)

  # create dataloaders for train, validation, test,
  train_dl = DataLoader(
      dataset=tr_set, batch_size=32, num_workers=os.cpu_count(), pin_memory=True, shuffle=True)
  test_dl = DataLoader(
      dataset=ts_set, batch_size=32, num_workers=os.cpu_count(), pin_memory=True, shuffle=False)
  val_dl = DataLoader(
      dataset=val_set, batch_size=32, num_workers=os.cpu_count(), pin_memory=True, shuffle=False)

  # 0 -> Uninfected, 1 -> Infected
  class_mapper = {0: 'Uninfected', 1:'Infected'}

  return train_dl, test_dl, val_dl, class_mapper
