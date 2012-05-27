import numpy
import scipy
from scipy import sparse
#from scipy.sparse import lil_matrix
from scipy.io.mmio import mminfo, mmread, mmwrite
import sys, os
import random

'''
	@author gibbons4@stanford

	Contact with any questions.
	reads in an MM format matrix and spits out the matrix and another three vectors
'''

def createInput(input_file, a_file, x_file, q, N) :
	#print 'reading the matrix'
	a = mmread(input_file)
	#print 'done'

	afile = open(a_file, 'w+')
	xfile = open(x_file, 'w+')

	elements = zip(a.row, a.col, a.data) 
	n = len(elements)
	p = max(a.row)
	#generate the vector x
	x = []
	#n = int(sys.argv[2])
	#elemRange = int(sys.argv[3])
	for i in range(0, q) :
		x.append(random.uniform(-1, 1))


	print >>afile, n, p, q, N

	#a
	for i,j,v in elements :
		print >> afile, v,

	print >> afile, ''

	#s
	s = random.sample(range(n), p)
	s.sort()
	print >>afile, 0,
	for i in range(1, p-1) :
	  print >> afile, s[i],
	print >> afile, n
	
	#x
	for xx in x:
		print >>xfile, xx,
	print >> xfile, ''
	
	#k
	#k = random.sample(range(q), n)
	#k.sort()
	for i in range(0, n) :
	  print >> afile, int(random.uniform(0, q)),
	print >> afile, ''


if len(sys.argv) != 2:
	print "Usage: python readMM.py dir"
	sys.exit(0)

files = os.listdir(sys.argv[1])
for f in files :
	DIR = f[:f.find('.')] + '/'
	os.mkdir(DIR)
	print DIR, f
	createInput(sys.argv[1] + '/' + f, DIR+'a.txt', DIR+'x.txt', random.randint(1000, 1000000), int(random.uniform(5, 100)));

