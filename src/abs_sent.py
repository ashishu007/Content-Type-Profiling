import re
import json
import spacy
import neuralcoref
import pandas as pd
from spacy.matcher import Matcher
from itertools import groupby

class BaseMerger():
	def __init__(self, nlp, ent_type, pos='X'):
		self.matcher = Matcher(nlp.vocab)
		self.ent_type = ent_type
		self.pos = pos

	def __call__(self, doc):
		matches = self.matcher(doc)
		spans = []
		for match_id, start, end in matches:
			spans.append(doc[start:end])
		with doc.retokenize() as retokenizer:
			for span in spans:
				span[0].ent_type_ = self.ent_type
				span[0].pos_      = self.pos
			try:
				retokenizer.merge(span)
			except:
				print('--not mergining token')
		return doc

class NumberComparisonMerger(BaseMerger):
	def __init__(self, nlp):
		super().__init__(nlp, ent_type='CARDINAL_COMPARISON')
		self.matcher.add('XY', None, [{'IS_DIGIT': True},{'LOWER': '-'},{'IS_DIGIT': True}])
		for preposition in ['for', 'to', 'by', 'in']:
			pattern = [{'IS_DIGIT': True},{'LOWER': '-'},{'LOWER': preposition},{'LOWER': '-'},{'IS_DIGIT': True}]
			self.matcher.add(f'X{preposition}Y', None, pattern)

class ShotBreakdownMerger(BaseMerger):
	def __init__(self, nlp):
		super().__init__(nlp, ent_type='SHOT_BREAKDOWN')
		self.matcher.add('SHOT-BREAKDOWN', None, [{'ORTH': 'ShotBreakdown'}])


class TreeExtracter:
	# Nodes matching this NER/PoS will be 'abstracted' (replace with tag-label)
	special_ents = {
		'DATE':         set(['SYM', 'NUM', 'PROPN']),
		'TIME':         set(['SYM', 'NUM', 'PROPN']),
		'PERCENT':      set(['SYM', 'NUM', 'PROPN']),
		'MONEY':        set(['SYM', 'NUM', 'PROPN']),
		'ORDINAL':      set(['SYM', 'NUM', 'PROPN']),
		'CARDINAL':     set(['SYM', 'NUM', 'PROPN']),
		'PERSON':       set(['PROPN']),
		'NORP':         set(['PROPN']),
		'FAC':          set(['PROPN']),
		'ORG':          set(['PROPN']),
		'GPE':          set(['PROPN']),
		'LOC':          set(['PROPN']),
		'PRODUCT':      set(['PROPN']),
		'EVENT':        set(['PROPN']),
		'WORK_OF_ART':  set(['PROPN']),
		'LAW':          set(['PROPN']),
		'LANGUAGE':     set(['PROPN']),
	}

	def __init__(self):
		# Spacy with neural coreference resolution
		self.nlp = spacy.load('en_core_web_lg', disable=[])
		neuralcoref.add_to_pipe(self.nlp)
		self.nlp.add_pipe(NumberComparisonMerger(self.nlp))
		self.nlp.add_pipe(ShotBreakdownMerger(self.nlp), last=True)

	def spacy_parses_to_abstract_text(self, sents):
		tokens = []
		for sent in sents:
			for token in sent:
				if token.ent_type_:
					tokens.append(f'{token.pos_}-{token.ent_type_}')
			else:
				## Need the whole token, instead of just lemma for template purposes
				# not now, because using original text for template extraction
				tokens.append(f'{token.pos_}-{token.lemma_}')
				# tokens.append(f'{token.pos_}-{token}')

		# Remove consecutive duplicates.  PROPN-GPE PROPN-ORG PROPN-ORG => PROPN-GPE PROPN-ORG (e.g. Portland Trail Blazers)
		return ' '.join([x[0] for x in groupby(tokens)])

	def raw_sentence(self, sents):
		return ' '.join([sent.text for sent in sents])

	def spacy_sent_tokenize(self, doc):
		# print(f'{doc.text}')
		sents = []
		all_sents = []
		valid_stop = False
		for sent in doc.sents:
			sents.append(sent)
			valid_stop = True if sent[-1].text in ['.', '?', '!'] else False
			if valid_stop:
				all_sents.append(self.raw_sentence(sents))
				sents = []
		return all_sents

	def process_one_summary(self, summary):
		if summary != '':
			summary = re.sub('\(\d+-\d+ \w{0,3}, \d+-\d+ \w{0,3}, \d+-\d+ \w{0,3}\)', '(ShotBreakdown)', summary)
			summary = re.sub('\s+', ' ', summary)

			orig_doc = self.nlp(summary)
			doc = self.nlp(orig_doc._.coref_resolved)

			sents         = []
			sent_num      = 0
			valid_stop    = False
			abs_sents     = []

			for sent in doc.sents:
				sents.append(sent)
				valid_stop = True if sent[-1].text in ['.', '?', '!'] else False
				if valid_stop:
					raw_sentence = self.raw_sentence(sents)
					abstract = self.spacy_parses_to_abstract_text(sents)
					abs_sents.append({
						"raw_coref": raw_sentence,
						"abs": abstract
					})
					sents = []
					sent_num += 1

		return abs_sents

def main():

	print('constructing...')
	extractor = TreeExtracter()
	nlp = spacy.load('en_core_web_lg', disable=[])
	print('constructed!!!')

	# for _, part in enumerate(['test', 'train', 'valid']):
	for _, part in enumerate(['train']):
		if part == 'train':
			seasons = [14, 15, 16]
			# seasons = [14]
		elif part == 'valid':
			seasons = [17]
		else:
			seasons = [18]

		print(part, seasons)

		raw_sents = {
			'season': [],
			'game_id': [], 
			'raw': [], 
			'raw_coref': [], 
			'abs': []
		}

		for _, season in enumerate(seasons):
			print(season)
			js = json.load(open(f'./sportsett/data/initial/jsons/{season}.json', 'r'))
			summs = [' '.join(i['summary']) for i in js]

			for idx, summ in enumerate(summs):

				if idx % 100 == 0:
					print(idx)

				out = extractor.process_one_summary(summ)

				doc = nlp(summ)
				all_sents = extractor.spacy_sent_tokenize(doc)
				# print(idx, len(out), len(all_sents))
				# print(out, all_sents)

				if len(out) == len(all_sents):
					for idx1, sent in enumerate(all_sents):
						raw_sents['season'].append(season)
						raw_sents['game_id'].append(idx)
						raw_sents['raw'].append(sent)
						raw_sents['raw_coref'].append(out[idx1]['raw_coref'])
						raw_sents['abs'].append(out[idx1]['abs'])
				else:
					print(idx, len(out), len(all_sents), "ignoring")

		df = pd.DataFrame(raw_sents)
		# print(df.shape)
		# print(df)

		df.to_csv(f'./sportsett/data/initial/csvs/{part}.csv', index=False)

if __name__ == '__main__':
	main()
