# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

#Project #0 CPSC 420
#Author: Zoe Cooper

#Import statements
from collections import Counter
from random import choice
import re

class Bucket:
	def __init__(self):
		self.next_word = Counter(); 
		
	def add_next_word(self, word):
		self.next_word[word] += 1

#Make NGram object

class NGram:

#order = 2 : bigram, 3 : trigram


	def get_buckets(self, order, string):
		Buckets = dict()
		words = re.findall(r"([\w]+)", string)
		token = []
		next_word = ''
		for i in  range(len(words) - order):
			token = []
			next_word = words[i + order]
			for j in range(order):
				token.append(words[i + j]) 
			if next_word not in Buckets:
				Buckets[tuple(token)] = Bucket() 
				Buckets[tuple(token)].add_next_word(next_word) 
			else:
				Buckets[tuple(token)].add_next_word(next_word) 
		return Buckets
		
	def topword(self, n, bucket):
		temp = list(reversed(sorted(bucket.next_word, key=bucket.next_word.get))) 
		if len(temp[:n]) > 0:
			return choice(temp[:n]) 
		else:
			return "" 
	
	'''
	@Corpus text is the nietzche file
	
	'''		
	def generated_ngrams(self, order, filename, amount):
		with open('corpus.txt', 'r') as f:
			string = f.read()
		string = re.sub('[,\.?"-\'!:;]', '', string) 
		lower = string.lower() #No capitals
		clean = re.sub('[^a-z, \n]+', '', lower) #Fixes punctuation
		string = re.sub('[,\.?"-\'!:;]', '', clean)
		Buckets = {}
		Buckets = self.get_buckets(order, string)
		
		first_token = choice(list(Buckets.copy().keys())) #Had to do copy bc python 3
		this_token = first_token 
		generated_words = []
		generated_words += list(this_token) 
		next_word = ""
		
		for i in range(amount):
			next_word = self.topword(1, Buckets[this_token]) 
			temp = list(this_token)
			temp.pop(0)
			temp.append(next_word)
			next_token = tuple(temp) 
			this_token = next_token
			generated_words += [next_word] 
			
		print(' '.join(generated_words))
	
ngram = NGram()
ngram.generated_ngrams(1, 'corpus.txt', 100) #You can change for unigrams, bigrams, trigrams, or less than 100 words generated or whatever.
		
