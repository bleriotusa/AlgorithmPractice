__author__ = 'Michael'
import re
file = open('names.rtfd/TXT.rtf',errors='ignore')
f = file.read()
names = re.findall(r'\n(\w+ \w+)}}', f)
output = open('Names.txt', 'w')
for num, name in enumerate(names): print(num, name)
# for num, name in enumerate(names): output.write(name + '\n')