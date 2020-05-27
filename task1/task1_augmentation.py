import nlpaug.augmenter.word as naw
import nltk
import warnings
warnings.filterwarnings('ignore')

from tqdm import tqdm
import pickle

from sklearn.datasets import fetch_20newsgroups

train_source = fetch_20newsgroups(subset='train')
test_source = fetch_20newsgroups(subset='test')

# data augmentation
train_data = train_source['data']
train_target = train_source['target']

test_data = test_source['data']

aug_syn = naw.SynonymAug(aug_src='wordnet')
train_source_aug = []
train_target_aug = []
for x, y in tqdm(zip(train_data, train_target), total=len(train_data)):
    train_source_aug.append(x)
    train_source_aug.append(aug_syn.augment(x, num_thread=8))
    train_target_aug.append(y)
    train_target_aug.append(y)

pickle.dump(train_source_aug, open('train_source_aug.pkl', 'wb'))
pickle.dump(train_target_aug, open('train_target_aug.pkl', 'wb'))