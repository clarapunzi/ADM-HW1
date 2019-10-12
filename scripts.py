# ===== PROBLEM1 =====

# INTRODUCTION

# Exercise 1 - Introduction - Say "Hello, World!" With Python

print("Hello, World!")


# Exercise 2 - Introduction - Python If-Else

import math
import os
import random
import re
import sys

n = int(input().strip())
if n%2!=0:               # n odd
    print('Weird')
elif n in range(2,6):    # 2 <= n < 6   
    print('Not Weird')
elif n in range(6,21):   # 6 <= n < 21
    print('Weird')
else:                    # n >= 21
    print('Not Weird')
    
    
# Exercise 3 - Introduction - Arithmetic Operators

a = int(input())
b = int(input())
c1, c2, c3 = a+b, a-b, a*b
print(c1,c2,c3, sep='\n')


# Exercise 4 - Introduction - Python: Division

a = int(input())
b = int(input())
int_div, fl_div = a//b, a/b
print(int_div,fl_div,sep='\n')


# Exercise 5 - Introduction - Loops

n, i = int(input()), 0
while i<n:
    sq = i*i
    print(sq)   # print the square of each number smaller than the input n
    i += 1
    
    
# Exercise 6 - Introduction - Write a function

def is_leap(year):
    if year%4!=0:   #4 doesn't divide the year
        leap = False
    elif year%100 == 0 and year%400!=0:   #the year can be evenly divided by 4 and 100 but not by 400
        leap=False
    else:          #the year can be evenly divided by 4, 100 and 400
        leap=True
    return leap


# Exercise 7 - Introduction - Print Function

n, i = int(input()), 1
while i <= n:
    print(i, end='', flush=True)
    i += 1
    
    
# DATA TYPES

# Exercise 8 - Basic data types - List Comprehensions

x, y, z, n = map(int, input().split())
print([[i,j,k] for i in range(x+1) for j in range(y+1) for k in range(z+1) if i+j+k!=n])


# Exercise 9 - Basic data types - Find the Runner-Up Score!

n = int(input())
arr = map(int, input().split())
arr_s = sorted(arr, reverse=True)  #sort the array in descending order, so the first elemente is the highest
m, i = arr_s[0], 1
while arr_s[i] == m and i<n:    #the loop stop when it finds the second-highest score
    i+=1
print(arr_s[i])


# Exercise 10 - Basic data types - Nested Lists

l, gr = [], []  
for _ in range(int(input())):
    name = input()
    score = float(input())
    l.append([name, score])  #create a list with names and scores
    gr.append(score)         #creat a list with just scores
second_low_gr = sorted(gr)[1]
for x,y in sorted(l):        #print the names with the second lowest score in alphabetically order
    if y == second_low_gr:
        print(x)
        

# Exercise 11 - Basic data types - Finding the percentage

n = int(input())
student_marks = {}
for _ in range(n):
    name, *line = input().split()
    scores = list(map(float, line))
    student_marks[name] = scores   #create a dictionary with the marks of each student
query_name = input()
perc = student_marks[query_name]   #consider all the marks of the selected student
s = 0
for e in perc:    #calculate the percentage of the marks
    s = s+e
s = s/3              
print(format(s,'.2f'))


# Exercise 12 - Basic data types - Lists

N = int(input())
l=[]
for _ in range(N):
    line = input().split()
    com = line[0]
    if len(line)==2:      #commands to perform each possible operation
        e = int(line[1])
    if len(line)==3:
        i = int(line[1])
        e = int(line[2])
    if com!='print':
        if com=='insert':
            l.insert(i,e)
        elif com=='remove':
            l.remove(e)
        elif com=='append':
            l.append(e)
        elif com == 'sort':
            l.sort()
        elif com=='pop' and len(l)!=0:
            l.pop()
        elif com=='reverse':
            l.reverse()
    else:
        print(l)

        
# Exercise 13 - Basic data types - Tuples

n = int(input())
integer_list = map(int, input().split())
t=tuple(integer_list)
print(hash(t))  #print the hash value of the tuple


# STRINGS
        
# Exercise 14 - Strings - sWAP cASE

def swap_case(s):
    return s.swapcase()


# Exercise 15 - Strings - String Split and Join

def split_and_join(line):
    line=line.split(' ')
    line='-'.join(line)
    return line


# Exercise 16 - Strings - What's Your Name?

def print_full_name(a, b):
    print('Hello '+a+' '+b+'! You just delved into python.')
    
    
# Exercise 17 - Strings - Mutations

def mutate_string(string, position, character):
    return string[:position]+character+string[position+1:]


# Exercise 18 - Strings - Find a string:

def count_substring(string, sub_string):   #Count the number of occurrences of the substring in the original string
    l1, l2, n = len(string), len(sub_string), 0
    for i in range(l1-l2+1):
        if string[i:i+l2] == sub_string and cases(string[i:i+l2], sub_string):
            n+=1
    return  n

def cases(s,t):   #Check if both characters of two strings with same index have the same case (upper or lower)
    for i in range(len(s)):
        if s[i].isupper() != t[i].isupper():
            return False
    return True

if __name__ == '__main__':
    string = input().strip()
    sub_string = input().strip()    
    count = count_substring(string, sub_string)
    print(count)
    

# Exercise 19 - Strings - String Validators

s = input()
l=5*[False]
for e in s: 
    if e.isalnum()==True:   #check if the string has any alphanumeric characters
       l[0]=True
    if e.isalpha()==True:   #check if the string has any alphabetical characters
       l[1]=True
    if e.isdigit()==True:   #check if the string has any digits
       l[2]=True
    if e.islower()==True:   #check if the string has any lowercase characters
       l[3]=True
    if e.isupper()==True:   #check if the string has any uppercase characters
       l[4]=True
print(*l,sep='\n')


# Exercise 20 - Strings - Text Alignment

thickness = int(input())
c = 'H'
#Top Cone
for i in range(thickness):
    print((c*i).rjust(thickness-1)+c+(c*i).ljust(thickness-1))
#Top Pillars
for i in range(thickness+1):
    print((c*thickness).center(thickness*2)+(c*thickness).center(thickness*6))
#Middle Belt
for i in range((thickness+1)//2):
    print((c*thickness*5).center(thickness*6))    
#Bottom Pillars
for i in range(thickness+1):
    print((c*thickness).center(thickness*2)+(c*thickness).center(thickness*6))    
#Bottom Cone
for i in range(thickness):
    print(((c*(thickness-i-1)).rjust(thickness)+c+(c*(thickness-i-1)).ljust(thickness)).rjust(thickness*6))
    

# Exercise 21 - Strings - Test Wrap

def wrap(string, max_width):
    return textwrap.fill(string, max_width)


# Exercise 22 - Strings - Designer Door Mat

n, m = map(int, input().split())
for i in range(n//2):
    print(('.|.'*(2*i+1)).center(m,'-'))   #Top semi-square
print('WELCOME'.center(m,'-'))
for i in range(n//2-1,-1,-1):    
    print(('.|.'*(2*i+1)).center(m,'-'))   #Bottom semi-square


# Exercise 23 - Strings - String Formatting

def print_formatted(number):
    l = len(bin(number)[2:])   #Width of the binary value of the number
    for i in range(number):
        print(str(i+1).rjust(l,' '),str(oct(i+1)[2:]).rjust(l,' '),str(hex(i+1)[2:]).upper().rjust(l,' '), str(bin(i+1)[2:]).rjust(l,' '),sep=' ')
        
        
# Exercise 24 - Strings - Alphabet Rangoli

import string
def print_rangoli(size):
    s = 4*int(size)-3
    alpha = string.ascii_lowercase
    for i in list(range(size))[::-1] + list(range(1, size)):
        print('-'.join(alpha[size-1:i:-1] + alpha[i:size]).center(s, '-'))
        
        
# Exercise 25 - Strings - Capitalize!

def solve(s):
    for x in s[:].split():
        s = s.replace(x, x.capitalize())
    return s


# Exercise 26 - Strings - The Minion Game

def minion_game(s):
    vowels, l = 'AEIOU', len(s)
    score_kevin = sum([l-i for i in range(l) if s[i] in vowels])   #Possible combinations starting with a vowel
    score_stuart = sum([l-i for i in range(l) if s[i] not in vowels])   #Possible combinations starting with a consonant
    if score_stuart>score_kevin:
        print('Stuart', score_stuart)
    elif score_kevin>score_stuart:
        print('Kevin', score_kevin)
    else:
        print('Draw')

        
# Exercise 27 - Strings - Merge the tools!

def merge_the_tools(string, k):
    for i in range(0,len(string),k):   #consider subsegments of lenght k of the given string
        s=''
        for j in string[i:i+k]:   #in each subsegment, just take the not repeating characters
            if j not in s:
                s+=j    
        print(s)
        

#SETS

# Exercise 28 - Sets - Introduction to Sets

def average(array):
    array = set(array)
    return sum(array)/len(array)


# Exercise 29 - Sets - No Idea!

l = map(int, input().split())
array, a, b = list(input().split()), set(input().split()), set(input().split())
s=0
for x in array:
    if x in a:
        s+=1
    elif x in b:
        s-=1
print(s)


# Exercise 30 - Sets - Symmetric Difference

n, set_n = int(input()), set(map(int, input().split()))
m, set_m = int(input()), set(map(int, input().split()))
print(*sorted(set_n.difference(set_m).union(set_m.difference(set_n))), sep='\n')   #Print the terms that exists in om or n but not in both


# Exercise 31 - Sets - Set .add() Operation

n=int(input())
print(len(set([str(input()) for i in range(n)])))   #Print the number of different inputs


# Exercise 32 - Sets - Set .discard(), .remove() & .pop()

n = int(input())
s = set(map(int, input().split()))
for i in range(int(input())):    #List the different commands that can appear
    comm=input().split()
    if comm[0]=='pop' and len(s)!=0:
        s.pop()
    elif comm[0]=='remove' and comm[1] in s:
        s.remove(int(comm[1]))
    else:
        s.discard(int(comm[1]))
print(sum(s))


# Exercise 33 - Sets - Set .union() Operation

n, en = int(input()), set(input().split())
b, fr = int(input()), set(input().split())
print(len(en.union(fr)))


# Exercise 34 - Sets - Set .intersection() Operation

n, en = int(input()), set(input().split())
b, fr = int(input()), set(input().split())
print(len(en.intersection(fr)))


# Exercise 35 - Sets - Set .difference() Operation

n, en = int(input()), set(input().split())
b, fr = int(input()), set(input().split())
print(len(en.difference(fr)))


# Exercise 36 - Sets - Set .symmetric_difference() Operation

n, en = int(input()), set(input().split())
b, fr = int(input()), set(input().split())
print(len(en.symmetric_difference(fr)))

# Exercise 37 - Sets - Set Mutation

l=int(input())
a=set(map(int, input().split()))
n=int(input())
for i in range(n):     #Operations that will possibly be executed
    comm=input().split()
    new_set=map(int, set(input().split()))
    if comm[0]=='update':
        a.update(new_set)
    if comm[0]=='intersection_update':
        a.intersection_update(new_set)
    if comm[0]=='difference_update':
        a.difference_update(new_set)
    if comm[0]=='symmetric_difference_update':
        a.symmetric_difference_update(new_set)
print(sum(a))


# Exercise 38 - Sets - The Captain's Room

k=int(input())
rooms=list(map(int,input().split()))
set_rooms=set(rooms)
print((sum(set_rooms)*k-sum(rooms))//(k-1))   #Sum as if there were k people in the Captain room and then compare with the data given


# Exercise 39 - Sets - Check Subset

n=int(input())
for i in range(n):
    x, a, y, b = input(), set(input().split()), input(), set(input().split())
    print(a.issubset(b))
    
    
# Exercise 40 - Sets - Check Strict Superset

a, n = set(input().split()), int(input())
res=True
for i in range(n):
    b=set(input().split())
    if a.issuperset(b)==False:
        res=False
        break
print(res)   #Print TRUE iff a is a superset of all the sets in input


#COLLECTIONS

# Exercise 41 - Collections - collections.Counter()

n_shoes = int(input())
sizes = list(map(int, input().split()))
n_cust= int(input())
money=0
for i in range(n_cust):
    p=list(map(int, input().split()))
    if p[0] in sizes:
        money+=p[1]   #if Raghu can sell a pair of shoes, add the money he earns
        sizes.remove(p[0])   #and remove the item sold
print(money)


# Exercise 42 - Collections - DefaultDict Tutorial

from collections import defaultdict
d = defaultdict(list)
n, m = map(int, input().split())
for i in range(n):
    d[input()].append(i+1)
for i in range(m):
    x=input()
    if x in d:   #check if each word in group B is also in group A (d)
        print(' '.join(map(str,d[x])))   #if so, print the indexes in A of the occurences 
    else:
        print('-1')
        

# Exercise 43 - Collections - Collections.namedtuple()

from collections import namedtuple
n, columns, tot = int(input()), input(), 0
stud=namedtuple('Student', columns)
for _ in range(n):
    stud2 = stud(*input().split())
    tot+= int(stud2.MARKS)     #Sum the marks of all the students
print('{:.2f}'.format(tot/n))

        
# Exercise 44 - Collections - Collections.OrderedDict()

from collections import OrderedDict
d = OrderedDict()
n = int(input())
for i in range(n):
    l=input().rpartition(' ')   #for each new input, check if the product is already in the dictionary and update the price
    if l[0] in d:
        d[l[0]]+=int(l[-1]) 
    else:
        d[l[0]]=int(l[-1])  
for e in d: 
    print(e, d[e])     
    

# Exercise 45 - Collections - Word Order

from collections import OrderedDict
d = OrderedDict()
n=int(input())
for i in range(n):
    word=input()
    if word in d:
        d[word]+=1
    else:
        d[word]=1
print(len(d))       #Print the number of different words in input
print(*d.values())  #Print the number of occurance of each word in input


# Exercise 46 - Collections - Collections.deque()

from collections import deque
n, d = int(input()), deque()
for _ in range(n):
    comm, *arg = input().split()
    getattr(d,comm)(*arg)   #Execute the command in input
[print(int(e), end=' ') for e in d]


# Exercise 47 - Collections - Company Logo

import math
import os
import random
import re
import sys
import collections 

string = sorted(collections.Counter(input()).items(), key= lambda x: (-x[1],x[0]))[:3]      #Sort in descending order of occurrance (-x[1]) and in alphabetical order for letters with same occurrance count. Then take the first three
print("\n".join(x[0]+" "+str(x[1]) for x in string))


# Exercise 48 - Collections - Piling up!

t = int(input())
for _ in range(t):
    n, l = int(input()), list(map(int, input().split()))  #n is the number of cubes, l is the list of the side-lenghts of the cubes
    i = 1
        while i<n-1 and (l[i]<=l[i-1] or l[i]<=l[i+1]):   #If there is an element in the list that is bigger then both the previous and the 
        i+=1                                              #netx element, then it isn't possible to stack the cubes
    print('Yes' if i==n-1 else 'No') 
    

#DATA AND TIME

# Exercise 49 - Date time - Calendar Module

import calendar
m, d, y = map(int, input().split())
print((calendar.day_name[calendar.weekday(y,m,d)]).upper())

# Exercise 50 - Date time - Time Delta

from datetime import datetime
import math
import os
import random
import re
import sys

def time_delta(t1, t2):
    date_format = '%a %d %b %Y %H:%M:%S %z'    #Specify the format of the timestamps
    d1 =int(datetime.strptime(t1,date_format).timestamp())
    d2 =int(datetime.strptime(t2,date_format).timestamp())
    return str(abs(d1-d2))    #Return the absolute difference of the two timestamps
    
if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')
    t = int(input())
    for t_itr in range(t):
        t1 = input()
        t2 = input()
        delta = time_delta(t1, t2)
        fptr.write(delta + '\n')
    fptr.close()
    
        
#EXCEPTIONS

# Exercise 51 - Exceptions

cases=int(input())
for i in range(cases):
    try:
        a,b = map(int, input().split())
        print(a//b)
    except ZeroDivisionError as e:
        print('Error Code:', e)
    except ValueError as e:
        print('Error Code:', e)    
    

#BUILT-INS

# Exercise 52 - Built-ins - Zipped!

(n_stud, x_subj) = map(int, input().split())
l = []
for i in range(x_subj):
    l.append(map(float, input().split()))
for i in zip(*l):
    print("{0:.1f}".format(sum(i)/x_subj))   #Print the average score of each student
    

# Exercise 53 - Built-ins - Athlete Sort

import math
import os
import random
import re
import sys

if __name__ == '__main__':
    n, m = map(int, input().split())
    arr = [[int(x) for x in input().split()] for _ in range(n)]
    k = int(input())
    arr.sort(key = lambda x: x[k])   #Sort the athlets based on the Kth  attribute
    for row in arr:
        print(*row)
        
        
# Exercise 54 - Built-ins - ginortS

s=input()
new_order='abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ1357902468'
print(*sorted(s, key=new_order.index), sep='')


#PYTHON FUNCTIONALS

# Exercise 55 - Map and Lambda Function

cube = lambda x: x**3

def fibonacci(n):    #Function that return the list of the first n Fibonacci numbers
    if n == 0:
        return []
    if n == 1:
        return [0]
    l = [0,1]
    for i in range(2,n):
         l.append(l[i-2] + l[i-1])
    return(l)

if __name__ == '__main__':
    n = int(input())
    print(list(map(cube, fibonacci(n))))  #Print the list of the cube of each Fibonacci number

    
# REGEX AND PARSING

# Exercise 56 - Regex - Detect Floating Point Number

import re 
t = int(input())
for _ in range(t):
    print(bool(re.match(r'^[+-]?[0-9]*\.[0-9]+$', input())))
    
    
# Exercise 57 - Regex - Re.split()

import re
regex_pattern = r"[,.]"
print("\n".join(re.split(regex_pattern, input())))


# Exercise 58 - Regex - Group(), Groups() & Groupdict()

import re
rep = re.search(r'([a-zA-Z0-9])\1', input())   #rep is TRUE the first time re.search() find a repeated alphanumeric character 
print(rep.group(1) if rep else -1)
    
    
# Exercise 59 - Regex - Re.findall() & Re.finditer()

import re
s = input()
c, v = 'bcdfghjklrmnpqrstvwxyzBCDFGHJKLNPQRSTVWXYZ','aeiouAEIOU'
m = re.findall(r'(?<=[%s])([%s]{2,})(?=[%s])'%(c,v,c), s)  #Search the substrings with at least two vowels and lying in between consonants
if m:  
    print('\n'.join(m))
else:
    print(-1)
    

# Exercise 60 - Regex - Re.start() & Re.end()

import re
s, k = input(), input()
for (n, x) in enumerate(s):
    if re.match(k, s[n:]):       #Search the occurences of the substring k in s
        print((n, n+len(k)-1))   #When found, print the starting and ening indexes
if re.search(k,s) == None:
    print((-1,-1))
    

# Exercise 61 - Regex - Regex Substitution

import re

for _ in range(int(input())):
    s = re.sub(r' &&(?= )', ' and', input())   #Substitute " && " with " and "
    s = re.sub(r' \|\|(?= )', ' or', s)    #Substitute " || " with " or "
    print(s)

    
# Exercise 62 - Regex - Validating Roman Numerals

import re
regex_pattern = r"M{0,3}(C[MD]|D?C{0,3})(X[CL]|L?X{0,3})(I[VX]|V?I{0,3})$"  #All the possible way to get a valide roman numeral <=3999
print(str(bool(re.match(regex_pattern, input()))))


# Exercise 63 - Regex - Validating phone numbers

import re
for _ in range(int(input())):
    print('YES' if re.match(r'[789]\d{9}$', input()) else 'NO') 
    

# Exercise 64 - Regex - Validating and Parsing Email Addresses

import re

n = int(input())
for _ in range(n):
    name, email = input().split(' ')
    if re.match(r'<[A-Za-z]+(\w|-|\.|_)+@[A-Za-z]+\.[A-Za-z]{1,3}>', email):   #Correct format
        print(name, email)
        

# Exercise 65 - Regex - Hex Color Code

import re
for _ in range(int(input())):
    m = [x for x in re.findall('[\s:](#[a-f0-9]{6}|#[a-f0-9]{3})[\s:;,)]', input(), re.I)]
    for x in m:
        print(x)

        
# Exercise 66 - Regex - HTML Parser - Part 1

from html.parser import HTMLParser
import re

class MyHTMLParser(HTMLParser):
    def handle_starttag(self, tag, attrs):   #Define the format of the start tags
        print('Start :', tag)
        for e in attrs:
            print('->', e[0], '>', e[1])
    def handle_endtag(self, tag):     #Define the format of the end tags
        print('End   :', tag)
    def handle_startendtag(self, tag, attrs):    #Define the format of the empty tags
        print('Empty :', tag)
        for e in attrs:
            print('->', e[0], '>', e[1])

n = int(input())
code = '\n'.join([input() for _ in range(n)])

while re.search(r'(?:<!--)\w(?:-->)', code):   #Exclude the comments
    code = code[:re.search(r'(?:<!--)\w(?:-->)', code).start()]+code[re.search(r'(?:<!--)\w(?:-->)', code).end():]  

parser = MyHTMLParser()
parser.feed(code)


# Exercise 67 - Regex - HTML Parser - Part 2

from html.parser import HTMLParser

class MyHTMLParser(HTMLParser):
    def handle_data(self, data):
        if data =='\n':
            return
        print('>>> Data', data, sep='\n')   #Format of non-empty data
    def handle_comment(self, comment):
        if '\n' in comment:
            print('>>> Multi-line Comment')   #Format of single-line comments
        else: 
            print('>>> Single-line Comment')   #Format of multi-line comments
        print(comment)    
html = ""       
for i in range(int(input())):
    html += input().rstrip()
    html += '\n'
    
parser = MyHTMLParser()
parser.feed(html)
parser.close()


# Exercise 68 - Regex - Detect HTML Tags, Attributes and Attribute Values

from html.parser import HTMLParser
import re

class MyHTMLParser(HTMLParser):  
    def handle_starttag(self, tag, attrs):  #Define how to print the tags, with the eventual attributes and values
        print(tag)
        for e in attrs:
            print('->', e[0], '>', e[1])

n = int(input())
code = '\n'.join(input() for _ in range(n))
code = re.sub(r'(?:<!--)(\w)(?:-->)', r'', code, re.MULTILINE)  #Exclude the comments

parser = MyHTMLParser()
parser.feed(code)


# Exercise 69 - Regex - Validating UID

import re

def f(s):
    if len(s) != 10:   #There must be exactly 10 characters 
        return 'Invalid'
    else:
        if not re.search(r'([A-Z].*){2,}', s):  #At least 2 uppercase alphabetic characters
            return 'Invalid' 
        if not re.search(r'([0-9].*){3,}', s):  #At least 3 digits
            return 'Invalid' 
        if not re.search(r'[a-zA-Z0-9]{10}', s):  #Only alphanumeric characters
            return 'Invalid'
        if re.search(r'(.).*\1', s):  #No repetitions
            return 'Invalid'
        return 'Valid'

for _ in range(int(input())):
    print(f(input()))
    
    
# Exercise 70 - Regex - Validating Credit Card Numbers

import re

for _ in range(int(input())):
    card = input()
    if re.match(r'^[456](\d{15}|\d{3}(?:-?\d{4}){3})$', card) and not re.search(r'(\d)\1{3,}', card.replace('-','')):  #Correct format
        print('Valid')
    else:
        print('Invalid')
        

# Exercise 71 - Regex - Validating Postal Codes

import re

P = input()
regex_integer_in_range = r"^[1-9]\d{5}$"   #Match integers from 100000 to 999999
regex_alternating_repetitive_digit_pair = r"(\d)(?=\d\1)"   #Match alternating repetitive digits

print (bool(re.match(regex_integer_in_range, P)) 
and len(re.findall(regex_alternating_repetitive_digit_pair, P)) < 2)


# Exercise 72 - Regex - Matrix Script

import math
import os
import random
import re
import sys

first_multiple_input = input().rstrip().split()
n = int(first_multiple_input[0])
m = int(first_multiple_input[1])
matrix = []
for _ in range(n):   #Fill the matrix
    matrix_item = input()
    matrix.append(matrix_item)
line=''
a = zip(*matrix)
for e in a:   #Put in a line all elements of the matrix, reading top to bottom, left to right
    line += ''.join(e) 
#Replace symbols and spaces between alphanumeric characters with a single space:
line = re.sub(r'(?<=[a-zA-Z0-9])[!@#$%& ]{1,}(?=[a-zA-Z0-9])', ' ', line)
print(line)
    
    
# XML

# Exercise 73 - Xml - XML 1 - Find the Score

import sys
import xml.etree.ElementTree as etree

def get_attr_number(node):
    s=0
    for child in node.iter():
        s = s + len(child.attrib)   #Sum all the attributes in each branch of the tree
    return s

if __name__ == '__main__':
    sys.stdin.readline()
    xml = sys.stdin.read()
    tree = etree.ElementTree(etree.fromstring(xml))
    root = tree.getroot()
    print(get_attr_number(root))

    
# Exercise 74 - Xml - XML2 - Find the Maximum Depth

import xml.etree.ElementTree as etree

maxdepth = 0
def depth(elem, level):
    global maxdepth
    if level == maxdepth:  #Update the maxdepth while going deeper
        maxdepth += 1
    for child in elem:
        depth(child, level+1)  #Recursion on the elements of the next levele (if there are)

if __name__ == '__main__':
    n = int(input())
    xml = ""
    for i in range(n):
        xml =  xml + input() + "\n"
    tree = etree.ElementTree(etree.fromstring(xml))
    depth(tree.getroot(), -1)
    print(maxdepth)
    
    
    
#CLOSURES AND DECORATORS

# Exercise 75 - Closures and decorators - Standardize Mobile Number Using Decorators

def wrapper(f):
    def fun(l):
        f('+91 '+n[-10:-5]+' '+n[-5:] for n in l)  #Rewrite the number in the correct format
    return fun

@wrapper
def sort_phone(l):
    print(*sorted(l), sep='\n')

if __name__ == '__main__':
    l = [input() for _ in range(int(input()))]
    sort_phone(l) 
    
    
# Exercise 76 Exercise 77 - Numpy -  - Closures and decorators - Decorators 2

import operator
def person_lister(f):
    def inner(people):
        return map(f,sorted(people, key=lambda x: int(x[2])))   #Sort the names by age
    return inner

@person_lister
def name_format(person):
    return ("Mr. " if person[3] == "M" else "Ms. ") + person[0] + " " + person[1]   #Rewrite the names in the standard format

if __name__ == '__main__':
    people = [input().split() for i in range(int(input()))]
    print(*name_format(people), sep='\n')
    
       
# NUMPY       

# Exercise 77 - Numpy - Arrays

import numpy

def arrays(arr):
    return numpy.array(arr[::-1],float)

arr = input().strip().split(' ')
result = arrays(arr)
print(result)


# Exercise 78 - Numpy - Shape and Reshape

import numpy
print(numpy.array(input().split(' '), int).reshape(3,3))


# Exercise 79 - Numpy - Transpose and Flatten

import numpy

n, m = map(int, input().split(' '))
arr = numpy.array([input().split() for _ in range(n)], int)
print(numpy.transpose(arr), arr.flatten(), sep='\n')    #Print the flatten copy of the transpose of the array in input


# Exercise 80 - Numpy - Concatenate

import numpy
n, m, p = map(int, input().split())

arr1 = numpy.array([input().split() for _ in range(n)], int)
arr2 = numpy.array([input().split() for _ in range(m)], int)
print(numpy.concatenate((arr1, arr2), axis=0))   #Concatenate the two arrays in input along the axis 0


# Exercise 81 - Numpy - Zeros and Ones

import numpy

dims = tuple(map(int, input().split())) 
print(numpy.zeros(dims, dtype = numpy.int))  #Print an array of zeros of the given shape
print(numpy.ones(dims, dtype = numpy.int))   #Print an array of ones of the given shape


# Exercise 82 - Numpy - Eye and Identity

import numpy
numpy.set_printoptions(legacy='1.13')  #Set the correct spacing in between elements of the array 

n, m = map(int, input().split())
print(numpy.eye(n, m, k=0))  #Print an array n x m with ones on the main diagonal and zeros elsewhere (if n = m it's just the identity)


# Exercise 83 - Numpy - Array Mathematics

import numpy

n, m = map(int, input().split())
a = numpy.array([input().split() for _ in range(n)], int)
b = numpy.array([input().split() for _ in range(n)], int)
print(a+b, a-b, a*b, a//b, a%b, a**b, sep='\n')


# Exercise 84 - Numpy - Floor, Ceil and Rint

import numpy
numpy.set_printoptions(sign=' ')   #Set the correct spacing in between elements of the array

a = numpy.array(input().split(), float)
print(numpy.floor(a), numpy.ceil(a), numpy.rint(a), sep='\n')


# Exercise 85 - Numpy - Sum and Prod

import numpy

n, m = map(int, input().split())
arr = numpy.array([input().split() for _ in range(n)], int)
print(numpy.prod(numpy.sum(arr, axis=0)))  #Take the sum of the values by column (axis 0) and print the product of them


# Exercise 86 - Numpy - Min and Max

import numpy

n, m = map(int, input().split())
a = numpy.array([input().split() for _ in range(n)], int)
print(numpy.max(numpy.min(a, axis=1)))   #Consider the minimum in each row and then print their maximum


# Exercise 87 - Numpy - Mean, Var, and Std

import numpy
numpy.set_printoptions(sign=' ')    #Set the correct spacing in between elements of the array

n, m = map(int, input().split())
a = numpy.array([input().split() for _ in range(n)], int)

print(numpy.mean(a, axis=1), numpy.var(a, axis=0), numpy.around(numpy.std(a), 12), sep='\n')  


# Exercise 88 - Numpy - Dot and Cross

import numpy
numpy.set_printoptions(sign=' ')   #Set the correct spacing in between elements of the array

n = int(input())
a = numpy.array([input().split() for _ in range(n)], int)
b = numpy.array([input().split() for _ in range(n)], int)

print(numpy.dot(a,b))  #Matrix multiplication


# Exercise 89 - Numpy - Inner and Outer

import numpy

a = numpy.array(input().split(), int)
b = numpy.array(input().split(), int)

print(numpy.inner(a,b), numpy.outer(a,b), sep='\n')


# Exercise 90 - Numpy - Polynomials

import numpy
#Evaluate the polynomial with the given coefficient at a given point:
print(numpy.polyval(numpy.array(input().split(), float),float(input())))  


# Exercise 91 - Numpy - Linear Algebra

import numpy
numpy.set_printoptions(legacy='1.13')

n = int(input())
a = numpy.array([input().split() for _ in range(n)], float)

print(round(numpy.linalg.det(a), 2))   #Print the determinant of the given matrix


        
# ===== PROBLEM2 =====

# Exercise 92 - Challenges - Birthday Cake Candles

import math
import os
import random
import re
import sys

def birthdayCakeCandles(ar):
    return ar.count(max(ar))   #Count the occurences of the highest value in the list in input

if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')
    ar_count = int(input())
    ar = list(map(int, input().rstrip().split()))
    result = birthdayCakeCandles(ar)
    fptr.write(str(result) + '\n')
    fptr.close()
    
    
# Exercise 93 - Challenges - Kangaroo

import math
import os
import random
import re
import sys

def kangaroo(x1, v1, x2, v2):
    if x1==x2:  #Same strting point
        return 'YES'
    if v1==v2 and x1!=x2:  #If they have same speed but different starting point they will never meet!
        return 'NO'
    if v1>=v2:  #Check if there exist a positive integer k that solve the equation x1 + k*v1 = x2 + k*v2 
        if (x2-x1)%(v1-v2)==0 and (x2-x1)//(v1-v2)>=0:
            return 'YES'
    else:
        if (x1-x2)%(v2-v1)==0 and (x1-x2)//(v2-v1)>=0:
            return 'YES'
    return 'NO'

if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')
    x1V1X2V2 = input().split()
    x1 = int(x1V1X2V2[0])
    v1 = int(x1V1X2V2[1])
    x2 = int(x1V1X2V2[2])
    v2 = int(x1V1X2V2[3])
    result = kangaroo(x1, v1, x2, v2)
    fptr.write(result + '\n')
    fptr.close()
    
    
# Exercise 94 - Challenges - Viral Advertising

import math
import os
import random
import re
import sys

def viralAdvertising(n):
    people, likes  = 5, 0
    for _ in range(n):
        new = math.floor(people/2)
        likes += new   #Update the number of people who like the advertisement
        people = new*3   #Update the number of people who receive the advertisement
    return likes

if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')
    n = int(input())
    result = viralAdvertising(n)
    fptr.write(str(result) + '\n')
    fptr.close()

    
# Exercise 95 - Challenges - Recursive Digit Sum

import math
import os
import random
import re
import sys

def superDigit(n, k):
    def sumDigit(p):   #The inner function sum recursively the digits of a number until it only has one digit
        if len(p)==1:
            return p
        else:
            p1 = sum(int(i) for i in p)
            return sumDigit(str(p1))
    p = sum(int(i) for i in n)*k
    return sumDigit(str(p))

if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')
    nk = input().split()
    n = nk[0]
    k = int(nk[1])
    result = superDigit(n, k)
    fptr.write(str(result) + '\n')
    fptr.close()    
   
 
# Exercise 96 - Challenges - Insertion Sort - Part 1

import math
import os
import random
import re
import sys

def insertionSort1(n, arr):
    x = arr[-1]
    for i in range(n-1,0,-1):  #Go through the array from right to left
        if arr[i-1] < x:  #The number in the position with lower index is smaller than the last
            arr[i] = x    #Then the right position for x is exactly at the next index
            print(*arr)
            break
        else:      #The number in the position with lower index is bigger than the last
            arr[i] = arr[i-1]   #Then copy the bigger in the first position on the right
            print(*arr)
    if arr[0]>x:   #Check if also the first element of the list is bigger than the last
        arr[0]=x
        print(*arr)

if __name__ == '__main__':
    n = int(input())
    arr = list(map(int, input().rstrip().split()))
    insertionSort1(n, arr)
    
    
# Exercise 97 - Challenges - Insertion Sort - Part 2

import math
import os
import random
import re
import sys

def insertionSort2(n, arr):
    for i in range(1,n):
        x = arr[i]
        if x > arr[i-1]:  #At each iteration, check if the element with index i is bigger than all the previous
            print(*arr)
        else:             #If not, put it in the correct position
            j=i-2
            while j >= 0 and arr[j] > x:  
                j -=1
            arr = arr[:j+1]+[x]+arr[j+1:i]+arr[i+1:]
            print(*arr)

if __name__ == '__main__':
    n = int(input())
    arr = list(map(int, input().rstrip().split()))
    insertionSort2(n, arr)