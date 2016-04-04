import os
import numpy as np
from random import randint
import math
import scipy
from scipy import linalg
import struct
#Planck Best fit variances taken from http://planck.caltech.edu/pub/2015results/Planck_2015_Results_XIII_Cosmological_Parameters.pdf page 16 1st column

Variables = [
['ombh2', 0.00023,'ombh2'],
['omch2',0.022,'omch2'],
['scalar_spectral_index(1)',0.0062,'ns'],
['hubble',0.96,'hubble'],
['re_optical_depth',0.019,'tau'],
['scalar_amp(1)',0.036E-09,'As']
]

##############
### Priors ###
##############

def getVariance(n):
	global Variables
	variance = Variables[n][1]**2
	return variance

def makePrior():
        global Variables
	global use
        nParams = len(Variables)
	priors = np.zeros((nParams,nParams))
	for n in use:
		priors[n][n] = 1/(getVariance(n))
	return priors

##############################
### Covariance Matrix Part ###
##############################

"""code here pulls out the data from the covariance matrix file"""

def covSize(file): #gets details on size and bins
	with open(file, mode='rb') as data:
        	filecontent = data.read()
	metadata = struct.unpack("iiii",filecontent[:16])
	return metadata

def getCovariance(file): #saves the covariance matrix
        f = open(file, "rb")
        f.seek(16, os.SEEK_SET)
        matrix = np.fromfile(f,dtype=np.float)
        matrixb= np.reshape(matrix,(np.sqrt(np.size(matrix)),np.sqrt(np.size(matrix))),'F')
        return np.asmatrix(matrixb)

def getInv(file):
	icov = np.loadtxt(file)
	matrix = []
	metadata = []
	for n in range(4,len(icov)):
		matrix.append(icov[n])
	for m in range(4):
		metadata.append(int(icov[m]))
	matrixb= np.asmatrix(np.reshape(matrix,(np.sqrt(np.size(matrix)),np.sqrt(np.size(matrix))),'F'))
	tuple = (metadata,matrix)
        return tuple

#####################
### PMF Templates ###
#####################

def readIn(file):
        list=[]
        with open(str(file),'r') as data:
                data = data.readlines()
        for x in range(len(data)):
                list.append(data[x].split())
        return list

def readTemp(reference,template):
        template = readIn(template)
	tempLength = len(template)
	refLength = len(reference)
        for line in template:
                del line[5]
                for n in range(len(line)):
                        line[n] = float(line[n])
        if refLength >= tempLength: 
		for n in range(refLength-tempLength):
			template.append([0,0,0,0,0])
	if tempLength > refLength:
		for n in range(tempLength-refLength):
			del template[len(reference)]
        return template 


####################
### Finding Dl's ###
####################

"""this code finds the Dl's for each parameter"""

def runCamb(file): #runs any .ini in camb.
        os.system("./cosmomc_jul15_pmf/camb/camb ./fisherOutputs/%s.ini" % file)


def modIni(Xi,pm,filename,Xiname,sigmaFactor,covFile): #varies paramter in .ini file and saves it
	k=sigmaFactor*Xi*pm
	b=[]
	cov = getInv(covFile)[0]
	with open('fileCAMB.ini','r') as file:
		a = file.readlines()
	for x in range(len(a)):
		a[x] = a[x].split()
		if a[x][0] == Xiname:
			a[x][2] = float(a[x][2]) + float(k)
		if a[x][0] == "l_max_scalar":
                        a[x][2] = int(cov[2]*cov[1]-cov[3]+1)
			print (str(a[x][0]) + " " + str(a[x][1]) + " " +str(a[x][2]))
                b.append(str(a[x][0]) + " " + str(a[x][1]) + " " +str(a[x][2])+ '\n')
	with open('fisherOutputs/'+ str(filename) + str(cov[1]*cov[2])+'.ini','w') as file:
		file.writelines(b)

def trimL(array,covFile): #removes first lmin-1 data points from Dl matrix.
        cov = getInv(covFile)[0]
	lmin = cov[3]
	array = array.tolist()
        for n in range(lmin-1):
                del array[0]
        return array

def diffClArray(file1,file2,Xi,Xiname,sigmaFactor,covFile): #finds local derivatives for each Dl wrt a parameter.
        cov = getInv(covFile)[0]
	l = cov[1]*cov[2]
	k=sigmaFactor*Xi
	modIni(Xi,1,file1,Xiname,sigmaFactor,covFile)
	runCamb(file1 + str(cov[1]*cov[2]))
	array1 = np.loadtxt('plik_plus_r0p01_lensedtotCls.dat')
	modIni(Xi,-1,file2,Xiname,sigmaFactor,covFile)
	runCamb(file2 + str(cov[1]*cov[2]))
	array2 = np.loadtxt('plik_plus_r0p01_lensedtotCls.dat')
	array3 = (array1 - array2)/(2*k)
	array3 = trimL(array3,covFile)
	np.savetxt('fisherOutputs/Diff%s' % Xiname + str(int(l)) + '.dat', array3)
	return array3

def useDiffs(covFile): # loads Dls from a file instead of making them through CAMB
	global Variables
        cov = getInv(covFile)[0]
	l = cov[1]*cov[2]
	diffs = []
	for n in range(len(Variables)):
		diffs.append(np.loadtxt('fisherOutputs/Diff%s' % str(Variables[n][0]) + str(int(l))+'.dat'))
	return diffs
	
def getDls(newDiffs,sigmaFactor,covFile): # finds or loads Dls for all parameters
	global Variables
	diffs = []
	if newDiffs == True: # if true, make new Dls
		for x in range(len(Variables)):
			diffs.append(diffClArray(Variables[x][2]+"p",Variables[x][2]+"m",Variables[x][1],Variables[x][0],sigmaFactor,covFile))
	if newDiffs == False: # if false, pre-load Dls
		diffs=useDiffs(covFile)
	diffs.append(readTemp(diffs[0],'vector_b1mpc=2p5nG_nb=-2p9_beta=20p7233.txt'))
	diffs.append(readTemp(diffs[0],'tensor_b1mpc=2p5nG_nb=-2p9_beta=20p7233.txt'))
	return diffs

#############################
### Matrix Multiplication ###
#############################

"""converts Dls from 6 vectors to a matrix, produces a bin matrix
with a built in window function and multiplies with the covariance matrix"""

def lister(array):
	if type(array) is not list:
		array = array.tolist()
	return array

def flatten(matrix): #converts individual paramater Dls into a vector
	flat = []
	matrix = lister(matrix)
	for variable in matrix:
		variable = lister(variable)
		for row in variable:
			del row[4]
        		del row[1]
        		del row[0]
		diffs = np.reshape(variable,-1,order='F')
		flat.append(diffs)
	return np.asmatrix(flat)

def binMatrix(covFile): #makes a bin matrix.
        cov = getInv(covFile)[0]
	file = np.loadtxt('fisherOutputs/Diffombh2' + str(cov[1]*cov[2]) + '.dat')
	l = len(file)
	matrix = np.zeros((cov[1],l))
	for n in range(cov[1]):
		matrix[n][n*cov[2]:(n+1)*cov[2]]=cov[2]**-1
	#print """WARNING: if the number of l's and the number of bins are not evenly divisible you will lose some of your data off the end. It's not a big deal for small bin sizes, but use big ones at your own risk!"""
	return matrix

def extendBin(matrix): #turns a matrix into a block diagonal of itself times over.
	matrix_new = scipy.linalg.block_diag(matrix,matrix)
	return np.asmatrix(matrix_new)

def fisher(newDiffs,sigmaFactor,covFile):
        cov = getInv(covFile)[0]
	A=flatten(getDls(newDiffs,sigmaFactor,covFile))
	AT=A.transpose()
	B=extendBin(binMatrix(covFile))
	BT=B.transpose()
	C=getInv(covFile)[1]
	C=np.asmatrix(np.reshape(C,(np.sqrt(np.size(C)),np.sqrt(np.size(C))),'F'))
	return A*BT*C*B*AT

def addPriors(priorMatrix,fisherMatrix):
	return priorMatrix + fisherMatrix

def forecast(matrix):
	inverse = np.linalg.inv(matrix)
	fc = []
	for n in range(len(inverse)):
		fc.append(math.sqrt(inverse[n,n]))
	return fc	

#############################
### Single Parameter Test ###
#############################

def singleDiff(n,covFile): # choose parameter n and load its derivatives from file.
        global Variables
        cov = getInv(covFile)[0]
        diffs = []
        diffs.append(np.loadtxt('fisherOutputs/Diff%s' % str(Variables[n][0]) + str(cov[1]*cov[2])+'.dat'))
        return diffs[0]

def singleFisher(file,n):
        global Variables
        cov = getInv(covFile)[0]
        A=np.asmatrix(flatten([singleDiff(n,file)]))
	AT=A.transpose()
        B=np.asmatrix(extendBin(binMatrix(file)))
        BT=B.transpose()
        C=np.asmatrix(getInv(file)[1])
	return A*BT*C*B*AT

################
### Run Code ###
################

def runMe():
	loc = '/sptcloud/data/dsutton/covs/iexpcov_28877d_07.3ukarcmin_knee400_fwhm10.0_minell20_maxell1200_deltaell25_cal0.050_beam0.125.bin'
	testInverseCovarianceMatrix = getInv(loc)
	testFisherMatrix = fisher(False,0.2,loc)
	print testFisherMatrix
	testForecast = forecast(testFisherMatrix)
	print testForecast

if __name__ == '__main__':
	runMe()






