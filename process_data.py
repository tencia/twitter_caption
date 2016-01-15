import os
import config as c
import numpy as np

# remove repeated captions and their corresponding images
lines = open(c.captions_file, 'r').readlines()
unique_caps = set([])
outputs = []
ims_to_remove = []
for line in lines:
    idx, cap = line.split(',')
    if cap not in unique_caps:
        unique_caps.add(cap)
        outputs.append(line)
with open(c.captions_file, 'w') as wr:
    wr.writelines(outputs)

# remove images that don't have captions
import os
idxs = set([int(line.split(',')[0]) for line in open(c.captions_file, 'r').readlines()])
images = os.listdir(c.images_dir)
for im in images:
    idx = int(im.split('.')[0])
    if idx not in idxs:
        os.remove(os.path.join(c.images_dir, im))

# identify words actually used and store them
lines = open(c.captions_file, 'r').readlines()
captions = [line.strip().split(',')[1].split(' ') for line in lines]
ntotal = len(captions)
caption_words = list(set([word for caption_list in captions for word in caption_list]))
with open(c.words_used_file, 'w') as wr:
    wr.writelines('{}\n'.format(word) for word in caption_words)
words_to_idx = dict((w,i+1) for i,w in enumerate(caption_words))
idx_to_words = dict((i+1,w) for i,w in enumerate(caption_words))
idx_to_words[0] = '<e>'
captions = dict([(int(line.split(',')[0]), line.strip().split(',')[1]) for line in lines])

# create dataset to save as hdf5
from PIL import Image
import os
import utils as u

idx = 0
label = 0
caption_matrix = np.zeros((ntotal, c.max_caption_len), dtype=np.uint16)
img_matrix = np.empty((ntotal, 3, c.img_size, c.img_size), dtype=np.uint8)
for label in captions:
    vector = np.asarray([words_to_idx[w] for w in captions[label].split(' ')], dtype=np.uint16)
    caption_matrix[idx, 0:vector.size] = vector
    im = Image.open(os.path.join(c.images_dir, '{}.jpg'.format(label)))
    img_matrix[idx] = u.arr_from_img_storage(im)
    idx += 1
    if idx % 400 == 0:
        print 'loaded img {}'.format(idx)

ntrain = int(ntotal * .95)
indices_dict = {'train': (0, ntrain), 'test': (ntrain, ntotal)}
u.save_hd5py({'images': img_matrix, 'captions': caption_matrix}, c.twimg_hdf5_file,
        indices_dict)
