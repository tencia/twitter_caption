import utils as u
import config as c

import private_config as pc
from PIL import Image
import requests
import numpy as np
from StringIO import StringIO
import json
import string
import os
import oauth2 as oauth
import urllib2 as urllib

from httplib import IncompleteRead

words_set = set([l.strip() for l in open(c.words_20k_file, 'r').readlines()])
exclude = set(string.punctuation + string.digits)

_debug = 0

oauth_token    = oauth.Token(key=pc.access_token_key, secret=pc.access_token_secret)
oauth_consumer = oauth.Consumer(key=pc.api_key, secret=pc.api_secret)

signature_method_hmac_sha1 = oauth.SignatureMethod_HMAC_SHA1()

http_method = "GET"


http_handler  = urllib.HTTPHandler(debuglevel=_debug)
https_handler = urllib.HTTPSHandler(debuglevel=_debug)

'''
Construct, sign, and open a twitter request
using the hard-coded credentials above.
'''
def twitterreq(url, method, parameters):
  req = oauth.Request.from_consumer_and_token(oauth_consumer,
                                             token=oauth_token,
                                             http_method=http_method,
                                             http_url=url, 
                                             parameters=parameters)

  req.sign_request(signature_method_hmac_sha1, oauth_consumer, oauth_token)

  headers = req.to_header()

  if http_method == "POST":
    encoded_post_data = req.to_postdata()
  else:
    encoded_post_data = None
    url = req.to_url()

  opener = urllib.OpenerDirector()
  opener.add_handler(http_handler)
  opener.add_handler(https_handler)

  response = opener.open(url, encoded_post_data)

  return response

# build caption only of valid words. if caption length < 3, return empty
def caption(txt):
    valids = []
    sec = txt.strip().lower().split(' ')
    for s in sec:
        if s == 'rt' or '@' in s or 'http' in s:
            continue
        s = ''.join(ch for ch in s.replace('#', '') if ch not in exclude)
        if s in words_set:
            valids.append(s)
            continue
        return []
    if len(valids) > 3:
        return valids
    return []

# fetch [limit] new tweets
def fetchsamples(limit):
  ret = []
  url = "https://stream.twitter.com/1/statuses/sample.json"
  parameters = []
  while len(ret) < limit:
      try:
          response = twitterreq(url, "GET", parameters)
          for line in response:
            ret.append(line.strip())
            if len(ret) % 100 == 0:
                print len(ret)
            if len(ret) >= limit:
              break
      except IncompleteRead:
          pass
  return ret

# filter tweets for images / good captions and output them to file
def output_valid_data(lines):
    try:
        j=[json.loads(l) for l in lines]
        valids = {}
        for j_idx,j1 in enumerate(j):
            if 'entities' in j1 and 'media' in j1['entities']:
                m = j1['entities']['media'][0]
                if 'jpg' in m['media_url'].lower():
                    cap = caption(j1['text'])
                    if len(cap) > 0 and m['media_url'] not in valids:
                        valids[m['media_url']] = ' '.join(cap[:c.max_caption_len])
        with open(os.path.join(c.raw_dir, 'data_{}.csv'.format(np.random.randint(1000000))),
                'w') as wr:
            wr.writelines(['{},{}\n'.format(u,cap) for u,cap in valids.iteritems()])
    except ValueError:
        pass

# check saved image URLs for 'natural' images and crop and save locally
def process_data(threshold = 0.18):
    for data_file in os.listdir(c.raw_dir):
        data = [l.strip().split(',') for l in
                open(os.path.join(c.raw_dir, data_file), 'r').readlines()]
        existing_images = os.listdir(c.images_dir)
        # start idx at 1+(highest existing index)
        idx = 0 if not existing_images else \
                max([int(s.split('.')[0]) for s in existing_images if 'jpg' in s]) + 1
        captions = {}
        for img_url, label in data:
            response = requests.get(img_url)
            if len(response.content) > 0 and 'html' not in response.content:
                img = Image.open(StringIO(response.content))
                w,h = img.size
                x_shift, y_shift = 0,0
                if w > h:
                    x_shift = (w-h)/2
                    w=h
                elif h > w:
                    y_shift = (h-w)/2
                    h=w
                img = img.crop((x_shift, y_shift, x_shift+w, y_shift+h))\
                        .resize((c.img_size, c.img_size), Image.ANTIALIAS)
                img_arr = np.array(img)
                # we want natural images (not text or screenshots) so filter out those which
                # have a large number of pixels concentrated at one value, which is
                # indicative of a big section of one color
                max_pixel_ratio = np.histogram(img_arr, bins=255)[0].max()*1./img_arr.size
                if max_pixel_ratio < threshold:
                    img.save(os.path.join(c.images_dir, '{}.jpg'.format(idx)))
                    captions[idx] = label
                    idx += 1
        with open(c.captions_file, 'a') as wr:
            wr.writelines(['{},{}\n'.format(i,cap) for i,cap in captions.iteritems()])
        os.remove(os.path.join(c.raw_dir, data_file))

if __name__ == '__main__':
  import numpy as np
  while True:
    lines = fetchsamples(1000)
    output_valid_data(lines)
    process_data()
