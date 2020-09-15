import numpy as np
import csv
from paths import BERUFSGRUPPEN_PATH,DICT_PATH
from pos_tags import *
import sys
import re
from multiword import create, load

f = open(BERUFSGRUPPEN_PATH, 'r')
BERUFSGRUPPEN = np.array(list(csv.reader(f, skipinitialspace=True)))
BERUFSGRUPPEN = {int(entry[0]) : int(entry[1]) for entry in BERUFSGRUPPEN}
f.close()

# BERUFSGRUPPEN #############

def get_berufsgruppen_for_entry(entry_id):
	entry_id = int(entry_id.replace("'",''))
	if(entry_id in BERUFSGRUPPEN):
		decimal = BERUFSGRUPPEN[entry_id]
	else:
		print(entry_id)
		decimal = 0
	bitcode = '{0:08b}'.format(decimal)
	while len(bitcode) < 32: bitcode = str(0) + bitcode
	bitarray = [int(b) for b in bitcode]
	return bitarray

def get_berufsgruppen(entries):
	return [get_berufsgruppen_for_entry(entry[1]) for entry in entries]

# POS #######################

def pos_to_one_hot(pos):
	pos = [pos_tags[p] for p in pos]
	len_pos = len(pos)
	one_hot = np.zeros((len_pos, len(pos_tags)))
	one_hot[np.arange(len_pos),pos] = 1
	return one_hot

def get_pos(entries):
	return [pos_to_one_hot(entry) for entry in entries]

# WORD FEATURES ##############
r = 0
def get_word_features_for_entry(entry):
	global r
	print(r)
	r+=1
	return [get_features_for_word(token) for token in entry]

phrases = None
n = 0
def get_word_features(entries):
	global phrases, n, r
	phrases, n, n2 = load()
	n = 0
	for k,v in phrases.vocab.items():
		if(str(k).count('_') < 1):
			n += v

	r = 0
	return [get_word_features_for_entry(entry) for entry in entries]

# suffixes Nomen -> -heit, -keit, -schaft, -ung, -wesen
def has_noun_suffix(word):
	noun_suffixes = ['heit', 'keit', 'schaft', 'ung', 'wesen']
	return int(any(word.endswith(ending) for ending in noun_suffixes))	

# suffixes Adjective -> -lig, lich-, lisch, -artig, -haft, -iv, -sam
def has_adjective_suffix(word):
	adjective_suffixes = ['lig', 'lich', 'lisch', 'artig', 'haft', 'iv', 'sam']
	return int(any(word.endswith(ending) for ending in adjective_suffixes))

def contains_digits(word):
	return int(any(char.isdigit() for char in word))

# contains special chars
def contains_special_chars(word):
	return int(not word.isalnum())

# first letter is capitalized
def first_letter_caps(word):
	return int(word[0].isupper())

# other than first letter is capt
def other_than_first_caps(word):
	if(len(word) > 1):
		return int(any(char.isupper() for char in word[1:]))
	else:
		return 0

# all caps
def all_caps(word):
	return int(word.isupper())

# ends with 'sch' -> Sprache
def ends_with_sch(word):
	word = word.replace('-kenntnisse','').replace('kenntnisse','').replace('-Kenntnisse','').replace('Kenntnisse','').replace('-kenntnissen','').replace('kenntnissen','').replace('-Kenntnissen','').replace('Kenntnissen','')
	return int(word.endswith('sch'))

# wortlänge
def longer_than_5(word):
	return int(len(word) > 5)

# enthält umlaute
def contains_umlaute(word):
	return int(any(char in ['ä', 'ü', 'ö', 'ß'] for char in word))

f = open(DICT_PATH, 'r',encoding='ISO-8859-1')
dictionary = [re.sub('\s+','',word) for word in f.readlines()]
f.close()

def is_in_dict(word):
	return word in dictionary or word.lower() in dictionary

def matches(term1, term2):
    return (term1 in term2 and len(term1) > len(term2)*2/3) or (term2 in term1 and len(term2)> len(term1)*2/3)

def match_case(term1, term2):
    return matches(term1.lower(), term2.lower())
    
def get_word_occ(word):
	return phrases.vocab[bytes(word, encoding='utf-8')] / n

def get_features_for_word(word):
	word_features = [
		has_noun_suffix(word),
		has_adjective_suffix(word),
		contains_digits(word),
		contains_special_chars(word),
		first_letter_caps(word),
		other_than_first_caps(word),
		all_caps(word),
		ends_with_sch(word),
		longer_than_5(word),
		contains_umlaute(word), 
		is_in_dict(word),
		get_word_occ(word)#'{:.20f}'.format(get_word_occ(word))
		]
	return word_features



if __name__ == '__main__':
  print(get_word_features([['HTML5', 'Python3'],['Deutsch']]))
 