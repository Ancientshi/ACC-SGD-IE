import os
import pickle
from typing import Optional, Callable, Any, Tuple
from PIL import Image
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torchvision
import torchvision.transforms as transforms


transform_cifar = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])


columns = ['Age','Workclass','fnlgwt','Education','Education num','Marital Status',
           'Occupation','Relationship','Race','Sex','Capital Gain','Capital Loss',
           'Hours/Week','Native country','Income']

def primary(x):
    if x in [' 1st-4th', ' 5th-6th', ' 7th-8th', ' 9th', ' 10th', ' 11th', ' 12th']:
        return ' Primary'
    else:
        return x
    
def native(country):
    if country in [' United-States', ' Cuba', ' 0']:
        return 'US'
    elif country in [' England', ' Germany', ' Canada', 
                     ' Italy', ' France', ' Greece', ' Philippines']:
        return 'Western'
    elif country in [' Mexico', ' Puerto-Rico', ' Honduras', ' Jamaica', 
                     ' Columbia', ' Laos', ' Portugal', ' Haiti', 
                     ' Dominican-Republic', ' El-Salvador', ' Guatemala', ' Peru', 
                     ' Trinadad&Tobago', ' Outlying-US(Guam-USVI-etc)', ' Nicaragua',
                     ' Vietnam', ' Holand-Netherlands' ]:
        return 'Poor' # no offence
    elif country in [' India', ' Iran', ' Cambodia', ' Taiwan', 
                     ' Japan', ' Yugoslavia', ' China', ' Hong']:
        return 'Eastern'
    elif country in [' South', ' Poland', ' Ireland', ' Hungary', 
                     ' Scotland', ' Thailand', ' Ecuador']:
        return 'Poland team'
    else:
        return country

# ------------------------
# Helper functions
# ------------------------

def apply_input_corruption(x: np.ndarray, sigma: float = 0.1) -> np.ndarray:
    """
    Add Gaussian noise to features (or pixel values) for corruption.
    """
    noise = np.random.normal(loc=0.0, scale=sigma, size=x.shape)
    return x + noise

def apply_input_corruption_nlp(x: np.ndarray, sigma: float = 0.1) -> np.ndarray:
    """
    Add Gaussian noise to features (or pixel values) for corruption.
    """
    n_samples = x.shape[0]
    n_features = x.shape[1]
    n_noisy = int(n_features * sigma)
    if n_noisy <= 0:
        return x
    x_noisy = x.copy()
    for i in range(n_samples):
        idx = np.random.choice(n_features, n_noisy, replace=False)
        x_noisy[i, idx] = 1
    return x_noisy



def apply_label_noise(y: np.ndarray, noise_rate: float = 0.1,seed: int = 0) -> np.ndarray:
    """
    Randomly flip labels at the specified noise_rate.
    """
    y_noisy = y.copy()
    n_tr = y_noisy.shape[0]
    np.random.seed(seed)
    ratio=noise_rate
    random_index_list=np.random.choice(n_tr,int(n_tr*ratio),replace=False)
    y_noisy[random_index_list]=1-y_noisy[random_index_list]   
    return y_noisy


# ------------------------
# Base DataModule
# ------------------------
class DataModule:
    def __init__(
        self,
        normalize: bool = True,
        append_one: bool = True
    ):
        self.normalize = normalize
        self.append_one = append_one

    def load(self) -> Tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError

    def fetch(
        self,
        n_tr: int,
        n_val: int,
        n_test: int,
        seed: int = 0,
        args: Any = None,
    ) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
        x, y = self.load()
        # split
        x_tr, x_val, y_tr, y_val = train_test_split(
            x, y, train_size=n_tr, test_size=n_val + n_test, random_state=seed
        )
        x_val, x_test, y_val, y_test = train_test_split(
            x_val, y_val, train_size=n_val, test_size=n_test, random_state=seed + 1
        )

        # apply corruption on training inputs
        if args.corrupted:
            if args.target == '20news':
                x_tr = apply_input_corruption_nlp(x_tr, sigma=args.corruption_sigma)
            else:
                x_tr = apply_input_corruption(x_tr, sigma=args.corruption_sigma)

        # apply label noise on training labels
        if args.noise:
            y_tr = apply_label_noise(y_tr, noise_rate=args.noise_rate,seed=args.seed)

        # normalize
        if self.normalize:
            scaler = StandardScaler()
            scaler.fit(x_tr)
            x_tr = scaler.transform(x_tr)
            x_val = scaler.transform(x_val)
            x_test = scaler.transform(x_test)

        # append bias term
        if self.append_one:
            x_tr = np.c_[x_tr, np.ones(x_tr.shape[0])]
            x_val = np.c_[x_val, np.ones(x_val.shape[0])]
            x_test = np.c_[x_test, np.ones(x_test.shape[0])]

        return (x_tr, y_tr), (x_val, y_val), (x_test, y_test)


# ------------------------
# MNIST Module
# ------------------------
class MnistModule(DataModule):
    def __init__(
        self,
        normalize: bool = True,
        append_one: bool = False
    ):
        super().__init__(
            normalize,
            append_one
        )
        from tensorflow.examples.tutorials.mnist import input_data
        self.input_data = input_data

    def load(self) -> Tuple[np.ndarray, np.ndarray]:
        mnist = self.input_data.read_data_sets('/tmp/data/', one_hot=True)
        ytr = mnist.train.labels
        xtr = mnist.train.images
        # binary classification: digits 1 vs 7
        xtr1 = xtr[ytr[:, 1] > 0]
        xtr7 = xtr[ytr[:, 7] > 0]
        x = np.vstack([xtr1, xtr7])
        y = np.concatenate([
            np.zeros(xtr1.shape[0], dtype=int),
            np.ones(xtr7.shape[0], dtype=int),
        ])
        return x, y


# ------------------------
# News Module
# ------------------------
class NewsModule(DataModule):
    def __init__(
        self,
        normalize: bool = True,
        append_one: bool = False
    ):
        super().__init__(
            normalize,
            append_one
        )

    def load(self) -> Tuple[np.ndarray, np.ndarray]:
        categories = ['comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware']
        train = fetch_20newsgroups(
            subset='train', remove=('headers','footers','quotes'), categories=categories
        )
        test = fetch_20newsgroups(
            subset='test', remove=('headers','footers','quotes'), categories=categories
        )
        vec = TfidfVectorizer(stop_words='english', min_df=0.001, max_df=0.2)
        x_tr = vec.fit_transform(train.data).toarray()
        x_test = vec.transform(test.data).toarray()
        x = np.vstack([x_tr, x_test])
        y = np.concatenate([train.target, test.target])
        return x, y


# ------------------------
# Adult Module
# ------------------------
class AdultModule(DataModule):
    def __init__(
        self,
        normalize: bool = True,
        append_one: bool = False,
        csv_path: str = '../data'
    ):
        super().__init__(
            normalize,
            append_one
        )
        self.csv_path = csv_path

    def load(self) -> Tuple[np.ndarray, np.ndarray]:
        cols = ['Age','Workclass','fnlgwt','Education','Education num',
                'Marital Status','Occupation','Relationship','Race','Sex',
                'Capital Gain','Capital Loss','Hours/Week','Native country','Income']
        train = pd.read_csv(f'{self.csv_path}/adult-training.csv', names=cols)
        test = pd.read_csv(f'{self.csv_path}/adult-test.csv', names=cols, skiprows=1)
        df = pd.concat([train, test], ignore_index=True)

        # basic cleaning
        df.replace(' ?', np.nan, inplace=True)
        df['Income'] = df['Income'].apply(lambda x: 1 if x.strip().startswith('>50K') else 0)
        df['fnlgwt'] = df['fnlgwt'].apply(lambda x: np.log1p(x))
        df['Education'] = df['Education'].apply(primary)
        df['Marital Status'].replace(' Married-AF-spouse',' Married-civ-spouse', inplace=True)
        df['Native country'].fillna(' 0', inplace=True)
        df['Native country'] = df['Native country'].apply(native)

        # one-hot encode categoricals
        cat_cols = df.select_dtypes(include=['object']).columns
        for col in cat_cols:
            dummies = pd.get_dummies(df[col], prefix=col)
            df = pd.concat([df, dummies], axis=1)
            df.drop(col, axis=1, inplace=True)

        x = df.drop('Income', axis=1).values
        y = df['Income'].values
        return x, y
