import numpy
import scipy
from scipy import sparse
#from scipy.sparse import lil_matrix
from scipy.io.mmio import mminfo, mmread, mmwrite
import sys
import random

'''
	@author gibbons4@stanford

	Contact with any questions.
	reads in an MM format matrix and spits out the matrix and another three vectors
'''

afile = open('a.txt', 'w')
xfile = open('x.txt', 'w')

if len(sys.argv) != 4:
	print "Usage: python readMM.py file.mtx q N"
	sys.exit(0)

#print 'reading the matrix'
#print sys.argv[1]
x = mmread(sys.argv[1])
#print 'done'
elements = zip(x.row, x.col, x.data) 
n = len(elements)
p = max(x.row)
q = int(sys.argv[2])
#generate the vector a
x = []
#n = int(sys.argv[2])
#elemRange = int(sys.argv[3])
for i in range(0, n) :
	x.append(random.uniform(-1, 1))

#p = int(sys.argv[4])
N = int(sys.argv[3])

print >>afile, n, p, q, N

#x
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

#a
for xx in x:
	print >>xfile, xx,
print >> xfile, ''

#k
#k = random.sample(range(q), n)
#k.sort()
for i in range(0, n) :
  print >> afile, int(random.uniform(0, q)),
print >> afile, ''
