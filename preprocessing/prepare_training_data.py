import numpy as np
import csv
from clean_text import clean_text
from multiword import get_multiword_tokens
from labels import *
from word2vec import W2V 
import pandas as pd
import sys
import time
from os import path, makedirs
from paths import TRAINING_DATA_PATH, PREP_TRAINING_DATA_PATH, CONTROL_IDS_PATH, ROOT_PATH, POS_TRAINING_DATA_PATH, TRAINING_DATA_BASE_PATH
import training_features
np.set_printoptions(suppress=True)
np.set_printoptions(threshold=sys.maxsize)

from nltk.corpus import stopwords
nltk_stopwords = stopwords.words('german')

f = open(CONTROL_IDS_PATH, 'r')
CONTROL_IDS = np.array(list(csv.reader(f, skipinitialspace=True))).flatten()
f.close()

vec_dim = 100
lower_case = False
skipgram = True
w2v = W2V(skipgram=skipgram, punctuation_marks=False, stoppwords=False, multiword=False, lower_case=lower_case, vector_dim=vec_dim)

def read_training_data():
	f = open(TRAINING_DATA_PATH, 'r')
	entries = list(csv.reader(f, skipinitialspace=True))
	f.close()
	print('Found ', len(entries), " training entries.")
	return entries

def get_path_extras(multiword, punc_marks, stopwords, window, use_softskills=True, use_langs=True):
	path_extras = ''
	if(multiword): path_extras += '_multi'
	if(punc_marks): path_extras += '_punc'
	if(stopwords): path_extras += '_stop'
	if(not use_softskills): path_extras += '_noSoft'
	if(not use_langs): path_extras += '_noLangs'
	if(not isinstance(window, int) and len(window) == 2): path_extras += '_win' + str(window[0]) + '-' + str(window[1])
	return path_extras

def read_clean_tokens(multiword, punc_marks, stopwords, window):
	path_extras = get_path_extras(multiword, punc_marks, stopwords, window)
	if(not path.exists(PREP_TRAINING_DATA_PATH+path_extras+'.csv')):	
		print('no clean tokens found. Cleaning...')	
		clean_training_data(multiword, punc_marks, stopwords, window)
	f = open(PREP_TRAINING_DATA_PATH+path_extras+'.csv', 'r')
	entries = list(csv.reader(f, skipinitialspace=True))
	f.close()
	f = open(POS_TRAINING_DATA_PATH+path_extras+'.csv', 'r')
	pos = list(csv.reader(f, skipinitialspace=True))
	f.close()
	if(lower_case):
		lc_entries = []
		for entry in entries:
			lc_entries.append([t.lower() for t in entry])
		entries = lc_entries
	print('Found ', len(entries), " cleaned entries.")
	return entries, pos

def write_clean_tokens_to_file(data, pos, multiword, punc_marks, stopwords, window):
	path_extras = get_path_extras(multiword, punc_marks, stopwords, window)
	pd.set_option('display.max_rows', len(data))
	df = pd.DataFrame([data])
	df.to_csv(PREP_TRAINING_DATA_PATH+path_extras+'.csv', mode='a',float_format='%.5f', na_rep="NAN!", header=False, index=False,line_terminator="")
	df = pd.DataFrame([pos])
	df.to_csv(POS_TRAINING_DATA_PATH+path_extras+'.csv', mode='a',float_format='%.5f', na_rep="NAN!", header=False, index=False,line_terminator="")
	pd.reset_option('display.max_rows')

def get_current_normalized_skill(skill, punc_marks, stopwords, multiword):
	norm = " ".join(clean_text(skill, remove_stopwords=not stopwords, remove_punctuation_marks=not punc_marks)[0])
	if(multiword and len(norm.split())>1):
		norm, _ = get_multiword_tokens(norm.split())
		return norm
	else:
		return norm.split()

def create_labels_for_tokens(tokens, skills, skill_labels, punc_marks, stopwords, multiword, use_softskills, use_langs):
	label = [len(labels)-1] * len(tokens)
	skill_index = 0
	prev_skill_index = 0
	token_index = 0

	iterations = 0 # needed because skills are sometimes in wrong order
	max_iterations = 10
	norm_skill = get_current_normalized_skill(skills[skill_index], punc_marks, stopwords, multiword)
	### iterate over skills
	while(skill_index < len(skills) and iterations < max_iterations): 
		### iterate over tokens
		while token_index < len(tokens): 
			if(label[token_index] == len(labels)-1):
				# single word skill
				if((len(norm_skill) == 1) and (skill_in_token(norm_skill[0], tokens[token_index]) or skill_in_token(skills[skill_index], tokens[token_index]))):
					if((skill_labels[skill_index]!='Softskill' or use_softskills) and (skill_labels[skill_index]!='Sprache' or use_langs)):
						label[token_index] = labels[skill_labels[skill_index]]
					skill_index += 1
					last_found_token_index = token_index
					if skill_index >= len(skills): break
					norm_skill = get_current_normalized_skill(skills[skill_index], punc_marks, stopwords, multiword)
				# multi word skill
				elif (len(norm_skill) > 1 and (skill_in_token(norm_skill[0], tokens[token_index]) or (skill_in_token(skills[skill_index], tokens[token_index])))):
					i = 0
					while (i<len(norm_skill) and (token_index < len(tokens)) and ((skill_in_token(norm_skill[i],tokens[token_index])) or norm_skill[0] in ['-','/',',','.'])):
						if((labels[skill_labels[skill_index]]!='Softskill' or use_softskills) or (labels[skill_labels[skill_index]]!='Sprache' or use_langs)):
							label[token_index] = labels[skill_labels[skill_index]]
						token_index += 1
						i += 1
					skill_index += 1					
					last_found_token_index = token_index
					if skill_index >= len(skills): break
					norm_skill = get_current_normalized_skill(skills[skill_index], punc_marks, stopwords, multiword)
				if skill_index >= len(skills): break
			token_index += 1
		token_index=0
		if(skill_index == prev_skill_index): # skip skill if not found
			skill_index += 1
			if skill_index >= len(skills): break			
			norm_skill = get_current_normalized_skill(skills[skill_index], punc_marks, stopwords, multiword)

		prev_skill_index = skill_index
		iterations+=1
	# make sure stopwords are labeled as noSkill
	for i, token in enumerate(tokens):	
		if(token in nltk_stopwords or (token in ['-','/',',','.']) or (token.isdigit())):
			label[i] = len(labels)-1
	return label, tokens

def skill_in_token(skill, token):
	return ((skill.lower() in token.lower() and len(token) >= len(skill)/2) or (token.lower() in skill.lower() and len(token) >= len(skill)/2)) 

def labels_to_one_hot(label):
	len_label = len(label)
	one_hot = np.zeros((len_label, len(labels)))
	one_hot[np.arange(len_label),label] = 1
	return one_hot

def no_skill_labels_to_one_hot(label):
	one_hot = [[0,1] if l == len(labels)-1 else [1,0] for l in label]
	return np.array(one_hot)

def one_hot_to_labels(one_hot):
	return np.argmax(one_hot, axis=1, out=None)

def label_to_string(num_labels):
	label_list = list(labels.keys())
	return [label_list[label-1] for label in num_labels]

def train_w2v_model(data, punctuation_marks, stoppwords, multiword):
	global w2v
	new = []
	for entry in data:
		tokens = [x for x in entry]
		for token in tokens:
			if(not w2v.is_word_in_model(token)):
				new.append(tokens)
				break
	print(len(new), new[0:5])
	if(len(new) > 0):
		w2v.train_model(new, skipgram=skipgram, punctuation_marks=punctuation_marks, stoppwords=stoppwords, multiword=multiword, lower_case=lower_case, vector_dim=vec_dim)

def read_word_vectors(tokens):
	global w2v
	vectors = [w2v.get_w2v(token) for token in tokens]
	return np.array(vectors)

def prep_label_pipeline(noSkill, tokens, entry, punc_marks, stopwords, multiword, use_softskills, use_langs):
	skills = entry[3].replace('[', '').replace(']', '').replace("'", '').strip().split(',')
	skills = [skill.strip() for skill in skills]
	label = entry[4].replace('[', '').replace(']', '').replace("'", '').strip().split(',')
	label = [l.strip() for l in label]
	skill_labels, tokens = create_labels_for_tokens(tokens, skills, label, punc_marks, stopwords, multiword, use_softskills, use_langs)

	if(not noSkill):
		return labels_to_one_hot(skill_labels)
	else:
		return no_skill_labels_to_one_hot(skill_labels)

def split_test_train(vectors, one_hot_labels, entries, pos, word_features, berufsgruppen):
	x_train = []
	y_train = []
	x_test = []
	y_test = []
	pos_test = []
	pos_train = []
	wf_test = []
	wf_train = []
	bg_test = []
	bg_train = []
	test_entries = []
	assert len(vectors) == len(one_hot_labels)
	for i in range(len(vectors)):
		anzeigen_fk = entries[i][1]
		if(str(anzeigen_fk) in CONTROL_IDS):
			if(len(pos) > 0):
				pos_test.append(pos[i])
			if(len(word_features) > 0):
				wf_test.append(word_features[i])
			if(len(berufsgruppen) > 0):
				bg_test.append(berufsgruppen[i])
			x_test.append(vectors[i])
			y_test.append(one_hot_labels[i])
			test_entries.append(entries[i])
		else:
			if(len(pos) > 0):
				pos_train.append(pos[i])
			if(len(word_features) > 0):
				wf_train.append(word_features[i])
			if(len(berufsgruppen) > 0):
				bg_train.append(berufsgruppen[i])
			x_train.append(vectors[i])
			y_train.append(one_hot_labels[i])
	return x_train, y_train, x_test, y_test, test_entries, pos_test, pos_train, wf_test, wf_train, bg_test, bg_train

def generate_windows(vectors, label, window, balanced, entries, noSkill, pos_tags, word_features, berufsgruppen, pos_window, wf_window, train=False):
	x = []
	pos = []
	wf = []
	bg = []
	y = []
	x_noSkill = []
	pos_noSkill = []
	wf_noSkill = []
	bg_noSkill = []
	y_noSkill = []
	anzeigen_fks = []
	window1 = 0
	window2 = 0
	if(isinstance(window, int)):
		window1 = window
		window2 = window
	elif(len(window) == 2):
		window1 = window[0]
		window2 = window[1]
	else:
		print('wrong window input', window)
	print('window', window)

	vector_length = len(vectors[0][0])
	zero_v = [0] * vector_length
	zero_wf = [0]
	if(len(word_features) > 0): zero_wf = [0] * len(word_features[0][0])
	zero_pos = [0]
	if(len(pos_tags) > 0): zero_pos = [0] * len(pos_tags[0][0])

	assert len(vectors) == len(label)
	if(len(pos_tags) > 0): assert len(vectors) == len(pos_tags)
	if(len(word_features) > 0): assert len(vectors) == len(word_features)
	if(len(berufsgruppen) > 0): assert len(vectors) == len(berufsgruppen)

	for i in range(len(vectors)):
		if(len(vectors[i])> window2*2+1):
			for center_v in range(0,len(vectors[i])):
				window_x = []
				window_pos = []
				window_wf = []
				# append vectors
				for w in range(-window1, window2+1):
					if(center_v+w < 0 or center_v+w >= len(vectors[i])):
						window_x.append(zero_v)
						if(len(word_features) > 0 and -wf_window <= w and wf_window >= w):
							window_wf.append(zero_wf)
						if(len(pos_tags) > 0 and -pos_window <= w and pos_window >= w):
							window_pos.append(zero_pos)
					else:
						window_x.append(vectors[i][center_v+w])
						if(len(word_features) > 0 and -wf_window <= w and wf_window >= w):
							window_wf.append(word_features[i][center_v+w])
						if(len(pos_tags) > 0 and -pos_window <= w and pos_window >= w):
							window_pos.append(pos_tags[i][center_v+w])
				if(train and balanced and not noSkill and label[i][center_v][len(labels)-1] == 1): # noSkill
					x_noSkill.append(window_x)
					y_noSkill.append(label[i][center_v])
					if(len(pos_tags) > 0): pos_noSkill.append(window_pos)
					if(len(word_features) > 0): wf_noSkill.append(window_wf)
					if(len(berufsgruppen) > 0): bg_noSkill.append(berufsgruppen[i])
				else:	
					x.append(window_x)
					y.append(label[i][center_v])
					if(len(pos_tags) > 0): pos.append(window_pos)
					if(len(word_features) > 0): wf.append(window_wf)
					if(len(berufsgruppen) > 0): bg.append(berufsgruppen[i])
				if(not train and len(entries) != 0):
					anzeigen_fks.append(entries[i][1])
				else:
					anzeigen_fks.append(i)
	return x, y, x_noSkill, y_noSkill, anzeigen_fks, pos, wf, bg, pos_noSkill, wf_noSkill, bg_noSkill

def generate_one_hot_labels(noSkill, clean_tokens, entries, punc_marks, stopwords, multiword, use_softskills, use_langs, window):	
	one_hot_labels = []
	extras = get_path_extras(multiword, punc_marks, stopwords, window, use_softskills, use_langs)
	if(not noSkill):
		if(not path.exists(TRAINING_DATA_BASE_PATH +  '/one_hot_labels'+extras+'.npy')):		
			one_hot_labels = [prep_label_pipeline(noSkill, clean_tokens[i], entries[i], punc_marks, stopwords, multiword, use_softskills, use_langs) for i in range(len(clean_tokens))]
			np.save(TRAINING_DATA_BASE_PATH +"/one_hot_labels"+extras+".npy", one_hot_labels)	
		else: 
			one_hot_labels = np.load(TRAINING_DATA_BASE_PATH + "/one_hot_labels"+extras+".npy", allow_pickle=True)
	else:		
		if(not path.exists(TRAINING_DATA_BASE_PATH +  '/no_skill_one_hot_labels'+extras+'.npy')):		
			one_hot_labels = [prep_label_pipeline(noSkill, clean_tokens[i], entries[i], punc_marks, stopwords, multiword, use_softskills, use_langs) for i in range(len(clean_tokens))]	
			np.save(TRAINING_DATA_BASE_PATH +"/no_skill_one_hot_labels"+extras+".npy", one_hot_labels)
		else: 
			one_hot_labels = np.load(TRAINING_DATA_BASE_PATH + "/no_skill_one_hot_labels"+extras+".npy", allow_pickle=True)

	np.savez('/home/franzi/Documents/all_labeled_skills', skill_labels=one_hot_labels, tokens=clean_tokens)	
	return one_hot_labels

def prep_training_data(dirpath, noSkill, balanced, window, use_pos, use_berufsgruppen, use_word_features, multiword, punc_marks, stopwords, pos_window, wf_window, use_softskills, use_langs):
	global w2v
	entries = read_training_data()
	clean_tokens, pos = read_clean_tokens(multiword, punc_marks, stopwords, window)
	one_hot_labels = []
	vectors = []
	pos_tags = []
	word_features = []
	berufsgruppen = []

	one_hot_labels = generate_one_hot_labels(noSkill, clean_tokens, entries, punc_marks, stopwords, multiword, use_softskills, use_langs, window)

	extras = get_path_extras(multiword, punc_marks, stopwords, window,  use_softskills, use_langs)	
	print(TRAINING_DATA_BASE_PATH +"/vectors"+extras+".npy")
	if(not path.exists(TRAINING_DATA_BASE_PATH + '/vectors'+extras+'.npy')):
		train_w2v_model(clean_tokens, punctuation_marks=punc_marks, stoppwords=stopwords, multiword=multiword)
		vectors = [read_word_vectors(clean_tokens[i]) for i in range(len(clean_tokens))]
		np.save(TRAINING_DATA_BASE_PATH +"/vectors"+extras+".npy", vectors)
	else:
		vectors = np.load(TRAINING_DATA_BASE_PATH + '/vectors'+extras+'.npy', allow_pickle=True)
	if(use_word_features):
		word_features = training_features.get_word_features(clean_tokens)
	if(use_pos):
		pos_tags = training_features.get_pos(pos)
	if(use_berufsgruppen):
		berufsgruppen = training_features.get_berufsgruppen(entries)

	# split training and test data
	x_train, y_train, x_test, y_test, test_entries, pos_test, pos_train, wf_test, wf_train, bg_test, bg_train = split_test_train(vectors, one_hot_labels, entries, pos_tags, word_features, berufsgruppen,)
	# generate training data
	x_train, y_train, x_noSkill, y_noSkill, _,pos_train, wf_train, bg_train,  pos_noSkill_train, wf_noSkill_train, bg_noSkill_train = generate_windows(x_train, y_train, window, balanced, [], noSkill, pos_train, wf_train, bg_train, pos_window, wf_window,train=True) 
	# generate test data
	x_test, y_test, _, _, test_afks,pos_test, wf_test, bg_test, _, _, _ = generate_windows(x_test, y_test, window, balanced, test_entries, noSkill, pos_test, wf_test, bg_test, pos_window, wf_window)
	save_files(dirpath, balanced, x_train, y_train, x_test, y_test, x_noSkill, y_noSkill, test_afks, pos_test, wf_test, bg_test, pos_train, wf_train, bg_train,  pos_noSkill_train, wf_noSkill_train, bg_noSkill_train)
	features = [pos_test, wf_test, bg_test, pos_train, wf_train, bg_train,  pos_noSkill_train, wf_noSkill_train, bg_noSkill_train]
	return np.array(x_train), np.array(y_train), np.array(x_test), np.array(y_test), np.array(x_noSkill), np.array(y_noSkill), np.array(test_afks), features

def clean_training_data(multiword, punc_marks, stopwords, window):
	entries = read_training_data()
	for i in range(0,len(entries)):
		print(i)
		tokens, pos_tags = clean_text(entries[i][2], remove_stopwords=not stopwords, remove_punctuation_marks=not punc_marks)
		if(multiword):
			tokens, pos_tags = get_multiword_tokens(tokens, pos_tags)
		write_clean_tokens_to_file(tokens, pos_tags, multiword, punc_marks, stopwords, window)
	print('done')

def get_data_path(noSkill=False, balanced=False, berufsgruppen=False, pos=False, stopwords=False, punc_marks=False, word_features=False, multiword=False, window=4, pos_window=0, wf_window=0, use_softskills=True, use_langs=True):#, vec_dim, lower_case, skipgram):
	dirname = '/data'
	if(noSkill): dirname += '_noSkill'
	if(balanced): dirname += '_balanced'
	if(berufsgruppen): dirname += '_bg'
	if(pos): dirname += '_pos'
	if(stopwords): dirname += '_stop'
	if(punc_marks): dirname += '_puncMarks'
	if(word_features): dirname += '_wFeatures'
	if(multiword): dirname += '_multi'
	if(isinstance(window, int)):
		dirname += '_' + str(window)
	elif(len(window) == 2):
		dirname += '_' + str(window[0])+ str(window[1])
	else:
		print('wrong window input', window)
	if(pos_window != 0): dirname += '_pos'+str(pos_window)
	if(wf_window != 0): dirname += '_wf'+str(wf_window)
	if(not use_softskills): dirname += '_noSoft'
	if(not use_langs): dirname += '_noLangs'
	if(vec_dim != 200): dirname += '_vecDim' +str(vec_dim)
	if(lower_case): dirname += '_lowerCase'
	if(not skipgram): dirname += '_cbow'
	return TRAINING_DATA_BASE_PATH+dirname

def save_files(dirpath, balanced, x_train, y_train, x_test, y_test, x_noSkill, y_noSkill, test_afks, pos_test, wf_test, bg_test, pos_train, wf_train, bg_train,  pos_noSkill_train, wf_noSkill_train, bg_noSkill_train):
	makedirs(dirpath, exist_ok=True)
	np.savez(dirpath + "/x_train", x_train=x_train, wf_train=wf_train, pos_train=pos_train, bg_train=bg_train)
	np.savez(dirpath + "/y_train", y_train=y_train)
	np.savez(dirpath + "/x_test", x_test=x_test, wf_test=wf_test, pos_test=pos_test, bg_test=bg_test)
	np.savez(dirpath + "/y_test", y_test=y_test)
	np.savez(dirpath + "/test_afks", test_afks=test_afks)
	if(balanced): 
		np.savez(dirpath + "/x_noSkill", x_noSkill=x_noSkill, pos_noSkill_train=pos_noSkill_train, wf_noSkill_train=wf_noSkill_train, bg_noSkill_train=bg_noSkill_train)
		np.savez(dirpath + "/y_noSkill", y_noSkill=y_noSkill)

def load_files(dirpath, balanced):
	data = np.load(dirpath + "/x_train.npz", allow_pickle=True)
	x_train = data['x_train']
	wf_train = data['wf_train']
	pos_train = data['pos_train']
	bg_train = data['bg_train']
	y_train = np.load(dirpath + "/y_train.npz", allow_pickle=True)['y_train']
	data = np.load(dirpath + "/x_test.npz", allow_pickle=True)
	x_test = data['x_test']
	wf_test = data['wf_test']
	pos_test = data['pos_test']
	bg_test = data['bg_test']
	y_test = np.load(dirpath + "/y_test.npz", allow_pickle=True)['y_test']
	test_afks = np.load(dirpath + "/test_afks.npz", allow_pickle=True)['test_afks']
	x_noSkill = []
	y_noSkill = []
	pos_noSkill_train = []
	wf_noSkill_train = []
	bg_noSkill_train = []
	if(balanced):
		data = np.load(dirpath + "/x_noSkill.npz", allow_pickle=True)
		x_noSkill = data['x_noSkill']
		pos_noSkill_train = data['pos_noSkill_train']
		wf_noSkill_train = data['wf_noSkill_train']
		bg_noSkill_train = data['bg_noSkill_train']
		y_noSkill = np.load(dirpath + "/y_noSkill.npz", allow_pickle=True)['y_noSkill']
	features = [pos_test, wf_test, bg_test, pos_train, wf_train, bg_train,  pos_noSkill_train, wf_noSkill_train, bg_noSkill_train]
	return x_train, y_train, x_test, y_test, x_noSkill, y_noSkill, test_afks, features

def get_info_for_afk(afk):	
	f = open(TRAINING_DATA_PATH, 'r')
	entries = list(csv.reader(f, skipinitialspace=True))
	f.close()
	rel = []
	for i in range(len(entries)):
		if(int(entries[i][1]) == afk):
			rel.append(entries[i])
	return rel

def get_training_data(noSkill=False, balanced=False, berufsgruppen=False, pos=False, stopwords=False, punc_marks=False, word_features=False, multiword=False, window=4, pos_window=0, wf_window=0, use_softskills=True, use_langs=True, vec_dim=100, skipgram=True, lower_case=False):

	dirpath = get_data_path(noSkill, balanced, berufsgruppen, pos, stopwords, punc_marks, word_features, multiword, window, pos_window, wf_window, use_softskills, use_langs)	
	print('using data from ', dirpath)
	if(not path.exists(dirpath + '/x_train.npz')):
		print('processing data...')
		x_train, y_train, x_test, y_test, x_noSkill, y_noSkill, test_afks, features = prep_training_data(dirpath, noSkill, balanced, window, pos, berufsgruppen, word_features, multiword, punc_marks, stopwords, pos_window, wf_window, use_softskills, use_langs)

		print(len(x_train), len(y_train), len(x_test), len(y_test), len(x_noSkill), len(y_noSkill))
		if(balanced):			
			return x_train, y_train, x_test, y_test, x_noSkill, y_noSkill, test_afks, features
		return x_train, y_train, x_test, y_test, test_afks, features
	else:
		print('loading data...')
		x_train, y_train, x_test, y_test, x_noSkill, y_noSkill, test_afks, features = load_files(dirpath, balanced)
		print(len(x_train), len(y_train), len(x_test), len(y_test), len(x_noSkill), len(y_noSkill))
		if(balanced):
			return x_train, y_train, x_test, y_test, x_noSkill, y_noSkill, test_afks, features
		return x_train, y_train, x_test, y_test, test_afks, features
