import os
import numpy as np
#assumes labelled data ra stored into a positive and negative folder
#returns two lists one with the text per file and another with the corresponding class 
def loadLabeled(path):

	rootdirPOS =path+'/pos'
	rootdirNEG =path+'/neg'
	data=[]
	Class=[]
	count=0
	for subdir, dirs, files in os.walk(rootdirPOS):
		
		for file in files:
			with open(rootdirPOS+"/"+file, 'r') as content_file:
				content = content_file.read() #assume that there are NO "new line characters"
				data.append(content)
	tmpc1=np.ones(len(data))
	for subdir, dirs, files in os.walk(rootdirNEG):
		
		for file in files:
			with open(rootdirNEG+"/"+file, 'r') as content_file:
				content = content_file.read() #assume that there are NO "new line characters"
				data.append(content)
	tmpc0=np.zeros(len(data)-len(tmpc1))
	Class=np.concatenate((tmpc1,tmpc0),axis=0)
	return data,Class

from bs4 import BeautifulSoup 
import re

def review_to_wordlist(review):
    '''
    Meant for converting each of the IMDB reviews into a list of words.
    '''
    # First remove the HTML.
    review_text = BeautifulSoup(review,"lxml").get_text()

    # Use regular expressions to only include words.
    review_text = re.sub("[^a-zA-Z]"," ", review_text)

    # Convert words to lower case and split them into separate words.
    words = review_text.lower().split()
    
    # Return a list of words
    return(words)