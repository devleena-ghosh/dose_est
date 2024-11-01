import heapq
from __builtin__ import min as _min, max as _max
from itertools import tee, izip, imap, count
from operator import itemgetter

InitPriority = 10

class HeapQ:
	def __init__(self, initial = None):
		if initial:
			self._data = [(InitPriority, item) for item in initial]
			heapq.heapify(self._data)
		else:
			self._data = []
			
	def isEmpty(self):
		return len(self._data) == 0

	def enqueue(self, item):
		heapq.heappush(self._data, item)

	def dequeue(self):
		return heapq.heappop(self._data)

	def push(self, item):
		heapq.heappush(self._data, (InitPriority, item))

	def pop(self):
		value = heapq.heappop(self._data)
		val = value[1]
		#print('pop: ', value, str(val))
		return val
		
	def size(self):
		return len(self._data)
		
	def heapify(self):
		return heapq.heapify(self._data)
	
	def heapreplace(self, item):
		return heapq.heapreplace(self._data, item)
	
	def heappushpop(self, item):
		return heapq.heappushpop(self._data, item)

	#def nsmallest(n, iterable, key=None):
		#"""Find the n smallest elements in a dataset.

		#Equivalent to:  sorted(iterable, key=key)[:n]
		#"""
		#if key is None:
			#return heapq.nsmallest(n, iterable)
		#in1, in2 = tee(iterable)
		#it = izip(imap(key, in1), count(), in2)                 # decorate
		#result = heapq.nsmallest(n, it)
		#return map(itemgetter(2), result)                       # undecorate

	#def nlargest(n, iterable, key=None):
		#"""Find the n largest elements in a dataset.

		#Equivalent to:  sorted(iterable, key=key, reverse=True)[:n]
		#"""
		#if key is None:
			#return heapq.nlargest(n, iterable)
		#in1, in2 = tee(iterable)
		#it = izip(imap(key, in1), count(), in2)                 # decorate
		#result = heapq.nlargest(n, it)
		#return map(itemgetter(2), result)                       # undecorate


	#def min(iterable, key=None):
		#if key is None:
			#return _min(iterable)
		#it = iter(iterable)
		#try:
			#min_elem = it.next()
		#except StopIteration:
			#raise ValueError('min() arg is an empty sequence')
		#min_k = key(min_elem)
		#for elem in it:
			#k = key(elem)
			#if k < min_k:
				#min_elem = elem
				#min_k = k
		#return min_elem

	#def max(iterable, key=None):
		#if key is None:
			#return _max(iterable)
		#it = iter(iterable)
		#try:
			#max_elem = it.next()
		#except StopIteration:
			#raise ValueError('max() arg is an empty sequence')
		#max_k = key(max_elem)
		#for elem in it:
			#k = key(elem)
			#if k > max_k:
				#max_elem = elem
				#max_k = k
		#return max_elem
		
#class Item:
	#def __init__(self, key, value):
		#self.key = key
		#self.value = value
	
	#def __str__(self):
		#s = '('+self.key+', '+self.value+')'
		
	#def __eq__(self, item):
		#return item.key == self.key and item.value == self.value
