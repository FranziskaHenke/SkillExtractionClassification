from nltk.parse import CoreNLPParser
import numpy as np
import re
import spacy
import csv
import Levenshtein
import time
import nltk

from nltk.corpus import stopwords
stopwords = stopwords.words('german')
from paths import DICT_PATH, ABBREV_PATH, STOPLIST_PATH
import stanza
from stanza.utils.resources import DEFAULT_MODEL_DIR

#stanza.download('de', DEFAULT_MODEL_DIR)

f = open(DICT_PATH, 'r',encoding='ISO-8859-1')
dictionary = [re.sub('\s+','',word) for word in f.readlines()]
f.close()

f = open(ABBREV_PATH, 'r',encoding='utf-8')
abbrevs = list(csv.reader(f))#, quoting=csv.QUOTE_NONE))
f.close()
abbrev_keys = [row[0].strip().replace('"', "") for row in abbrevs]
abbrev_keys_lower = [row[0].strip().lower() for row in abbrevs]

nlp = spacy.load('de_core_news_sm')
stanza_nlp = stanza.Pipeline(models_dir=DEFAULT_MODEL_DIR, lang='de', processors='tokenize,pos', tokenize_pretokenized=True, verbose=False)

def separate_trailing_punctuation(tokens):
	out_text = ""
	for token in tokens:
		if(token[-1] in ['.',',',';', ":","?"]) and len(token) > 1:
			out_text += token[:-1]+" "+token[-1]
		else:
			out_text += token
		out_text += " "
	return out_text.split()

def tokenize(text):
	tokens = replace_abbrevs(text.split())
	text = " ".join(tokens)
	pos = []
	if(len(tokens) > 0):
		doc = stanza_nlp(text)
		tokens = [l.text for sent in doc.sentences for l in sent.words]
		pos = [l.xpos if l.xpos != None else 'XY' for sent in doc.sentences for l in sent.words]
	return tokens, pos

def is_in_dict(token):
	return (token in dictionary or token.lower() in dictionary)

def check_if_words_exist_in_dict(tokens):
	in_dict = []
	for i, token in enumerate(tokens):
		in_dict.append((token in dictionary or token.lower() in dictionary))
	return in_dict

def lemmatize(tokens, pos):
	result = []
	for i,token in enumerate(tokens):
		if((pos[i] in ['NN','ADJA']) or pos[i][0] == 'V') and str(token).isalpha():
			doc = nlp(str(token))
			result.append(doc[0].lemma_)
		else:
			result.append(token)
	return result

def resolveHyphensAndSlashes(tokens):
	resolved = []

	for i,token in enumerate(tokens):
		found = False
		# left hyphen, e.g. Verantwortungs- und Kooperationsbereitschaft
		if ((str(token) == '-') and (i+2 < len(tokens) and (tokens[i+1] in ['und','&','oder', '+', '/','bzw.']) and (i > 0) and (tokens[i+2] not in ['in', 'innen', 'er', 'e']))) :
			hyphenatedWord = tokens[i-1]
			relatedWord = tokens[i+2]
			for i in range(2, len(relatedWord)):
				if (hyphenatedWord + relatedWord[i:] in dictionary):
					resolved.pop()
					resolved.append(hyphenatedWord + relatedWord[i:])
					found = True
					break
				if(i == len(relatedWord)-1):	
					resolved.append('-')
					found = True

		# right hyphen, Selbstmanagement & -reflexion
		elif ((str(tokens[i-1]) == '-') and (tokens[i-2] in ['und','&','oder','/','+','bzw.']) and (i > 2) and (i < len(tokens) )) :
			hyphenatedWord = tokens[i]
			relatedWord = tokens[i-3]
			for i in range(0, len(relatedWord)-2):
				if (relatedWord[:i] + hyphenatedWord in dictionary):
					resolved.pop()
					resolved.append(relatedWord[:i] + hyphenatedWord)
					found = True
					break
				if(i == len(relatedWord)-1):
					resolved.append('-')
					found = True
		if(not found):
			resolved.append(token)
	return resolved

def replace_abbrevs(tokens):
	i = 0
	while i < len(tokens):
		token = tokens[i]
		if(token.lower() in abbrev_keys_lower):
			tokens[i] = abbrevs[abbrev_keys_lower.index(token.lower())][1].replace('"', '')
		elif(i+1 < len(tokens)) and (token.lower() + tokens[i+1].lower() in abbrev_keys_lower):
			tokens[i] = abbrevs[abbrev_keys_lower.index(token.lower() + tokens[i+1].lower())][1].replace('"', '')
			tokens = np.delete(tokens, [i+1])
		i += 1
	return tokens

def replace_spelling_errors(tokens, in_dict):
	for i, token in enumerate(tokens):
		if(not in_dict[i] and len(token) > 4):
			lowest_ratio = 0
			sub_dict = [i for i in dictionary if i.startswith(token[:1].upper()) or i.startswith(token[:1].lower())]
			for entry in sub_dict:
				ratio = Levenshtein.ratio(token.lower(), entry.lower())
				if(ratio >= 0.9 and ratio > lowest_ratio):
					tokens[i] = entry
					in_dict[i] = True
					lowest_ratio = ratio
					if(ratio == 1.0):
						break
	return tokens, in_dict

def remove_stopwords_from_tokens(tokens, pos):
	cleaned_tokens = []
	cleaned_pos = []
	for i,token in enumerate(tokens):
		if((pos[i] in ['NN','ADJA'] or pos[i][0] == 'V') or not token.lower() in stopwords):
			cleaned_tokens.append(token)
			cleaned_pos.append(pos[i])
	return cleaned_tokens, cleaned_pos

def removeBrackets(text):
	text = text.replace("(", " ").replace(")", " ").replace("[", " ").replace("]", " ")
	return text

def split_at_special_char(text):
	text = text.replace("&", " & ").replace(" -", " - ").replace("- ", " - ").replace("? ", " ? ").replace("! ", " ! ").replace(", ", " , ")
	return text

def removeSpecialChars(text):
	text = text.replace("•", " ").replace(">", " ").replace("■", " ").replace("►", " ").replace("...", " ").replace("❚", " ").replace("●", " ").replace("▪", " ").replace("|", " ").replace("", " ").replace("", " ").replace("✓", " ").replace("'", " ").replace('"', ' ').replace("·", " ").replace("–", " ").replace("", " ").replace(" **", " ").replace(" *", " ").replace("'", " ").replace('"', " ").replace('',' ').replace("»", " ").replace('‐', ' ').replace("", " ").replace(" –", " ").replace("„", " ").replace("“", " ").replace("", " ").replace("♦", " ")
	return text

def remove_standalone_specialchars_from_tokens(tokens, pos):
	indexes = [i for i, token in enumerate(tokens) if token.strip() in ['&','$','%','=','*','§',"'",'"','+',';',':', '-', '""', "''", '``', '`', "**", "...", "−", "--", '–', "…", '—','/']  ]
	tokens = np.delete(tokens, indexes)
	pos = np.delete(pos, indexes)
	return tokens, pos

def remove_punct_marks(tokens, pos):
	indexes = [i for i, token in enumerate(tokens) if token.strip() in ['?','!',',','.'] ]
	tokens = np.delete(tokens, indexes)
	pos = np.delete(pos, indexes)
	return tokens, pos

def normalize_punctuation_marks(tokens):
	tokens = [ token.strip().replace("?", ".").replace("!", ".") if len(token) == 1 else token for token in tokens ]
	return tokens

def slashRule(text):
	# split word at slash if all "subtokens" are longer than 2 -> don't split I/O + m/w/d, but do split Informatik/Wirtschaft Koch/Köchin, C/C++
	tokens = text.split()
	text = ""
	for token in tokens:
		sub_tokens = token.split("/")
		sub_too_short = False
		sub_non_alpha = False
		for sub_token in sub_tokens:
			if len(sub_token) <= 2:
				sub_too_short = True
			if not sub_token.replace('-','').isalpha():
				sub_non_alpha = True
		if( ( sub_too_short and sub_non_alpha)): token = token.replace('/', ' / ')
		text += token + " "
	return text

def removeMWD(text):
	text = text.replace("(m/w/d)", "").replace("(m/w)", "")
	return text


def removeGenderEndings(tokens):
	endings = ["/-in", "/-innen","/-er","/-e","\*-in","\*-innen","\*-er","\*-e","/in", "/innen","/er","/e","\*in","\*innen","\*er","\*e", "/r", "/-r", "\*r", "\*-r"]
	clean_tokens = []
	for token in tokens:
		for ending in endings:
			if(token.endswith(ending) and len(token) > len(ending)):
				token = re.sub(ending+'$', '', token)
		clean_tokens.append(token)
	return clean_tokens

def remove_first_special_char(text):
	if(not text[0].isalnum()): return text[1:]
	return text

def clean_text(text, clean_only=False, remove_stopwords=True, remove_punctuation_marks=True):

	text = remove_first_special_char(text)
	text = removeBrackets(text)	
	text = removeSpecialChars(text)
	text =  " ".join(separate_trailing_punctuation(text.split()))
	text = slashRule(text)
	text = split_at_special_char(text)	
	tokens = text.split()
	tokens = resolveHyphensAndSlashes(tokens)
	tokens = removeGenderEndings(tokens)
	pos_tags = tokens
	if(not clean_only):
		tokens, pos_tags = tokenize(' '.join(tokens))
	else:
		pos_tags = tokens
	if(remove_stopwords):
		tokens, pos_tags = remove_stopwords_from_tokens(tokens, pos_tags)
	if(remove_punctuation_marks):
		tokens, pos_tags = remove_punct_marks(tokens, pos_tags)
	else:
		tokens = normalize_punctuation_marks(tokens)
	tokens, pos_tags = remove_standalone_specialchars_from_tokens(tokens, pos_tags)
	return list(tokens), pos_tags

if __name__ == '__main__':
    text = """Studium/Berufserfahrung: Abgeschlossenes betriebswrtschaftliches Hochschul-/Fachhochschulstudium/Duale (m/w/d) Hochschule Sehr gute Kenntnisse und Erfahrungen im Bereich Planung und I/O Controlling mit Schwerpunkt i/O-Produktcontrolling .NET C# C++ Alternativ: Fundierte Kenntnisse in Einkaufsprozessen und/oder Prozessen der Kostenplanung Sprachkenntnisse: Gute bis sehr gute Englischkenn. Was Sie mitbringen sollten: - Laufendes Studium im Bereich Informatik oder C/.NET/C++ ein vergleichbarer Studiengang - Sicherer Umgang mit Microsoft Umgebungen - Grundkenntnisse in der Entwicklung moderner Javascript-Anwendungen basierend auf React.js oder Node.js oder - Erste Erfahrungen in einer Programmiersprache, idealerweise in Java, C/C++ oder C# - Eine lösungsorientierte Arbeitsweise mit einem hohen Maß an Eigenverantwortung, Zuverlässigkeit und Flexibilität - Gute Sprachkenntnisse in Deutsch und Englisch"""
    text = """Du hast Erfahrung im Umgang mit .Net und C#, kennst dich mit 5G aus, sowie MS-Office."""
    print(clean_text(text))