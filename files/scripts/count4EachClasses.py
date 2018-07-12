#coding = utf-8

import os

basePath = "/home/xlw/Git-Repo/Domain-Adversarial/files/data/Label"

path4Label = os.path.join(basePath, "Label_6052SAMPLE.txt")

with open(path4Label, "rb") as fileP:
  fileLines = fileP.readlines()
  fileLines = [fileLine[: -3].split('\t') for fileLine in fileLines]
  fileLinesFirstCol = [fileLine[0] for fileLine in fileLines]
  fileLinesSecondCol = [fileLine[1] for fileLine in fileLines]
  fileLinesThirdCol = [fileLine[2] for fileLine in fileLines]
  fileLinesLabels = [fileLine[2].split(';') for fileLine in fileLines]

print fileLines[-12: ]
#print fileLines[-12][2][0]
print fileLinesFirstCol[-12: ]
print fileLinesSecondCol[-12: ]
print fileLinesThirdCol[-12: ]
print fileLinesLabels[-12: ]

dic4Labels = {}
dic4CountofLabel = {}  # How many instances for each of class

for fileLinesLabel in fileLinesLabels:
  num4CountofLabel = len(fileLinesLabel)
  if dic4CountofLabel.has_key(num4CountofLabel):
    dic4CountofLabel[num4CountofLabel] += 1
  else:
    dic4CountofLabel[num4CountofLabel] = 1
  for iele in fileLinesLabel:
    if dic4Labels.has_key(iele[0]):
      dic4Labels[iele[0]] += 1
    else:
      dic4Labels[iele[0]] = 1

for key in dic4Labels:
  print "Key is %s, the count is %d." % (key, dic4Labels[key])
for key in dic4CountofLabel:
  print "There are %d instances with %d labels." % (dic4CountofLabel[key], key)
