#!/usr/bin/env ipython

from tqdm import tqdm
import requests
import regex as re
import os

os.chdir(os.path.dirname(__file__))
print(os.getcwd())

langs = ['de', 'en', 'es', 'fr', 'ja', 'zh']
dats = ['train', 'dev', 'test']

for dat in dats:
	for lang in langs:
		url = "https://amazon-reviews-ml.s3-us-west-2.amazonaws.com/json/"+dat+"/dataset_"+lang+"_"+dat+".json"
		print(url)
		
		# If file already exists
		if(os.path.isfile(os.sep.join(re.split('/', url)[-2:]))): print("Already exists!!"); continue
		
		# Streaming, so we can iterate over the response.
		req = requests.get(url, stream=True)
		
		# Total size in bytes.
		total_size = int(req.headers.get('content-length', 0))
		block_size = 1024 #1 Kibibyte
		
		bar = tqdm(total=total_size, unit='iB', unit_scale=True)
		
		with open(os.sep.join(re.split('/', url)[-2:]), 'wb') as foo:
			for data in req.iter_content(block_size):
				bar.update(len(data))
				foo.write(data)
		bar.close()
		
		if total_size != 0 and bar.n != total_size: print("ERROR, something went wrong")
		
	print("done")