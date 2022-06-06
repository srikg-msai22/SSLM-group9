import pandas
import re


def clean(s):
	s=s.replace("\\","")
	s=s.replace("[","")
	s=s.replace("]","")
	s=s.replace("\'","")
	s=s.replace("\"","")
	s=s.split(', ')
	final=[]
	for n in s:
		n=n.strip()
		n=re.sub(r'_\(.*?\)','',n)
		final.append(n)
	return final

def accuracy():

	data = pandas.read_csv("resultsdata_5_t5small.csv")

	# print(data.at[0,"model predicted tuple"])
	# print(data.at[0,"original tuple"])

	correct=0
	accuracy = 0
	total=data.shape[0]

	for i in range(data.shape[0]):
		orig = clean(data.at[i,"original tuple"])
		pred = clean(data.at[i,"model predicted tuple"])
		c=0
		t=len(orig)
		for i in orig:
			if i in pred: 
				c += 1
		# if c==0:
			# print(orig,pred)
		if c==t:
			correct += 1
		accuracy += c/t
	print(correct/total, accuracy/total)
