

import math
import os
import random
import re
import sys

if __name__ == '__main__':
    n = int(input().strip())
if n % 2 == 0 and n > 20:
    print("Not Weird")
elif n % 2 == 0 and n >= 6 and n <= 20:
    print("Weird")
elif n % 2 == 0 and n >= 2 and n <= 5:
    print("Not Weird")
elif n % 1 == 0:
    print("Weird")
else:
    print("")   
    
# square of a number , first get as list, unpack and display line by line
n = int(input())
sq_num = []
for i in range(0,n,1):
    sq_num.append(i * i)
for i in sq_num:
    print(i, end="\n")
    
    
# leap year check
def is_leap(year):
    if year % 400 == 0:
        return True
    if year % 100 == 0:
        return False
    if year % 4 == 0:
        return True
    else:
        return False
year = int(input())
print(is_leap(year))

# print "123" if n = 3
if __name__ == '__main__':
    n = int(input())
    for i in range(1,n+1,1):
        print(str(i), end = '')

# second largest number in the given array

if __name__ == '__main__':
    n = int(input())
    arr = map(int, input().split())
    s = set(arr)
    print(sorted(s)[-2])        

# List comprehension

x, y, z, n = int(input()), int(input()), int(input()), int(input())
print ([[a,b,c] for a in range(0,x+1) for b in range(0,y+1) for c in range(0,z+1) if a + b + c != n ])


# print average for specific name

if __name__ == '__main__':
    n = int(input())
    student_marks = {}
    for _ in range(n):
        name, *line = input().split() # string input 
        scores = list(map(float, line))
        student_marks[name] = scores
    query_name = input()
    avg = sum(student_marks[query_name])/float(len(student_marks[query_name]))
    print(avg)
    
# sets

# print (distinct) average for given set of numbers

def average(array):
    s = set(array)
    ave = 0
    for i in s:
        ave = sum(s)/len(s)
    return ave

if __name__ == '__main__':
    n = int(input())
    arr = list(map(int, input().split()))
    result = average(arr)
    print(result)
    
# get set of some numbers from user
(_, A) = (input(),set(map(int, input().split())))

# i want to get n number of elements 
n = int(input())
for i in range(n):
    print(i)
    
    
    
# Finding the symmetric difference 

n = int(input())
s_n = set(map(int, input().split()))
m = int(input())
s_m = set(map(int, input().split()))
sor = sorted(s_n.symmetric_difference(s_m))
for i in range(len(sor)):
    print(sor[i])
    
# print distinct country counts

N = int(input())
country = []
for i in range(N):
    country.append(input())
print(len(set(country)))

# union operations of set


n = int(input())
n_1=list(map(int, input().strip().split()))
m = int(input())
m_1=list(map(int, input().strip().split()))
print(len(set(n_1).union(set(m_1))))

# intersection operations
n = int(input())
n_1=list(map(int, input().strip().split()))
m = int(input())
m_1=list(map(int, input().strip().split()))
print(len(set(n_1).intersection(set(m_1))))

#difference

n = int(input())
n_1=list(map(int, input().strip().split()))
m = int(input())
m_1=list(map(int, input().strip().split()))
print(len(set(n_1).difference(set(m_1))))

# remove,discard, pop in sets

n = int(input())
s = set(map(int, input().split()))
N = int(input())
empty = []

for i in range(N):
    x = input().split()
    empty.append(x)
 
for i in range(len(empty)):
    if empty[i][0] == 'remove':
        s.remove(int(empty[i][1]))
    elif empty[i][0] == 'pop':
        s.pop()
    elif empty[i][0] == 'discard':
        s.discard(int(empty[i][1]))
        
print(sum(s))

# insert/append/discard/.. using python
# use of eval, join functions

n = int(input())
l = []
for _ in range(n):
    s = input().split()
    cmd = s[0]
    args = s[1:]
    if cmd !="print":
        cmd += "("+ ",".join(args) +")"
        eval("l."+cmd)
    else:
        print(l)
        
# using sets to perform set operations 
# used getattr(), see how input is obtained 

(_, A) = (input(),set(map(int, input().split())))

for _ in range(int(input())):
    (command, newSet) = (input().split()[0],
        set(map(int, input().split()))
    )
    getattr(A, command)(newSet)

print(sum(A))

# Captain room number

d=int(input())  
a=map(int, input().split())  #store all to array
s1=set();  #all unique room number
s2=set();  #all unique room number occur more than once
for i in a:
    if  i in s1:
        s2.add(i)
    else:
        s1.add(i)
s3=s1.difference(s2)
print(s3.pop())


# A is subset of B??

n = int(input())
for _ in range(n):
    _, A = int(input()),set(map(int,input().split()))
    _, B = int(input()),set(map(int,input().split()))
    print(A.issubset(B))
  
# A contains B??  

a = set(input().split())
print(all(a > set(input().split()) for _ in range(int(input()))))

# polar coordinates

import cmath
print(*cmath.polar(complex(input())), sep='\n')

# Find angle and print degree 

import math
AB, BC = float(input()), float(input())
print(str(int(round(math.degrees(math.atan2(AB,BC)))))+chr(176)) 

# print triangle --> interesting

for i in range(1,int(input())+1):
   print((10**i//9)**2)
   
# print triangle --> other one 


for i in range(1,int(input())):
    print((10**(i)//9)*i)


#divmod in python

a = int(input())
b = int(input())
print(a//b)
print(a%b)
print(divmod(a,b))

# print pow (2 , 3 attrributes)

a = int(input())
b = int(input())
m = int(input())

print(pow(a,b))
print(pow(a,b,m))

# a ^b + c ^d

a = int(input())
b = int(input())
c = int(input())
d = int(input())

print(pow(a,b) + pow(c,d))

# cartesian product

from itertools import product
A = list(map(int, input().split()))
B = list(map(int, input().split()))
prod = product(A,B)

for e in prod:
    print(e, end = " ")
    
#  permutations


import itertools
ll, n = input().split()

for i in itertools.permutations(sorted(ll),int(n)):
    print("".join(i))
    
# combinations

from itertools import combinations
string, n = input().split()
for i in range(1,int(n)+1):
    data = ["".join(sorted(i)) for i in combinations(string,i)]
    data.sort()
    print("\n".join(data))
    
    
# combinations with replacements

import itertools
ll, n = input().split()

for i in itertools.combinations_with_replacement(sorted(ll),int(n)):
    print("".join(i))
 
# compress the string - itertools.groupby

import itertools
a = input()
x = itertools.groupby(a)
for k,g in x:
    print((len(list(g)),int(k)), end = ' ')   


# Iterators and iterables

from itertools import combinations
input()
combos = list(combinations(input().split(), int(input())))
print(combos)
count = 0
for i in combos:
    if "a" in i:
        count+=1
print(round(count/len(combos),3))

# Maximize it! tricky question to understand and solve

from itertools import product
K,M = map(int,input().split())
nums = []
for _ in range(K):
    row = map(int,input().split()[1:])
    nums.append(map(lambda x:x**2%M, row))
print(max(map(lambda x: sum(x)%M, product(*nums))))

# perfect square (if n = 20, perfect squares till n are 1, 4, 9, 16)

A = int(input())
n1 = 1
n2 = n1 * n1
n1 = (n1 * 2) + 1
while ((n2 <= A)) :
    print(n2, end= " ")
    n2 = n2 + n1
    n1 += 2

# len of digits/ count of digits

for _ in range(int(input())):
    print(len(str(int(input()))))
    
# sum of given digits

sum = 0
for _ in range(int(input())):
    sum = 0
    for i in input():
        sum = sum + int(i)
    print(sum)
    

# Collections:
# using counter --> very interesting

no_of_shoes = int(input())
sizes = Counter(map(int,input().split()))
no_of_customers = int(input())
earnings = 0

for i in range(no_of_customers):
    size,price = map(int,input().split())
    if sizes[size] > 0:
        sizes[size] -= 1
        earnings += price
        
print(earnings)

# DefaultDict --> astonishing>>

from collections import defaultdict
n,m = list(map(int,input().split()))
d = defaultdict(list)
for i in range(n):
    d[input()].append(i+1)
for i in range(m):
    print(*d[input()] or [-1])
    
# Date time difference 

#!/bin/python3

import math
import os
import random
import re
import sys
from datetime import datetime as dt


# Complete the time_delta function below.
format_string = "%a %d %b %Y %H:%M:%S %z"
def time_delta(t1, t2):
    first = dt.strptime(t1,'%a %d %b %Y %H:%M:%S %z')
    second = dt.strptime(t2,'%a %d %b %Y %H:%M:%S %z')
    return str(abs(int((first-second).total_seconds())))

if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    t = int(input())

    for t_itr in range(t):
        t1 = input()

        t2 = input()

        delta = time_delta(t1, t2)

        fptr.write(delta + '\n')

    fptr.close()
    
# using zip 

n, x = map(int, input().split()) 

sheet = []
for _ in range(x):
    sheet.append(map(float, input().split()) ) 

for i in zip(*sheet): 
    print(round(sum(i)/len(i),1))
    
    
# using input and eval -->have a look again

ui = input().split()
x = int(ui[0])
print(eval(input()) == int(ui[1]))

# using eval to print

eval(input())

# sort 

#!/bin/python3

import math
import os
import random
import re
import sys
from operator import itemgetter


if __name__ == '__main__':
    nm = input().split()

    n = int(nm[0])

    m = int(nm[1])

    arr = []

    for _ in range(n):
        arr.append(list(map(int, input().rstrip().split())))

    k = int(input())
    arr = sorted(arr, key=itemgetter(k))
    for item in arr:
        print(' '.join(map(str, item)))


# any or all --> palindromic integer

N,n = int(input()),input().split()
print(all([int(i)>0 for i in n]) and any([j == j[::-1] for j in n]))

# addition, subtraction, division, floor division, power, mod of two arrays

import numpy
n , m = input().split()
a = []
b = []
for i in range(int(n)):
    my_array = input().split()
    a.append(my_array)
for j in range(int(n)):
    my_array1 = input().split()
    b.append(my_array1)    
lisa = numpy.array(a,int)
lisb = numpy.array(b,int)
print(numpy.add(lisa,lisb))
print(numpy.subtract(lisa,lisb))
print(numpy.multiply(lisa,lisb))
print(numpy.floor_divide(lisa,lisb))
print(numpy.mod(lisa,lisb))
print(numpy.power(lisa,lisb))

# printing the product of array

import numpy
n, m = input().split()
a = []
for i in range(int(n)):
    my_array = input().split()
    a.append(my_array)
lisa = numpy.array(a,int)
print(numpy.prod(numpy.sum(lisa, axis=0), axis=0))

# alternate way for the above problem

import numpy
N, M = map(int, input().split())
A = numpy.array([input().split() for _ in range(N)],int)
print(numpy.prod(numpy.sum(A, axis=0), axis=0))


# printing min/max of array
N, M = map(int, input().split())
A = numpy.array([input().split() for _ in range(N)],int)
print(numpy.max(numpy.min(A, axis=1), axis=0))

# print mean, var, std
import numpy
N, M = map(int, input().split())
A = numpy.array([input().split() for _ in range(N)],int)
print(numpy.mean(A, axis = 1))
print(numpy.var(A, axis = 0))
stdA = numpy.std(A, axis = None)
print(round(stdA,11))

# matrix multiplication - dot()
import numpy
N = int(input())
A = numpy.array([input().split() for _ in range(N)],int)
B = numpy.array([input().split() for _ in range(N)],int)
print(numpy.dot(A, B))

# inner product and outer product

import numpy
A = list(map(int, input().split()))
B = list(map(int, input().split()))
print(numpy.inner(A, B))
print(numpy.outer(A, B))

# polynomials

poly = list(map(float, input().split()))
x = int(input())
n = len(poly)
result = 0
for i in range(n):
    Sum = poly[i]
    for j in range(n - i - 1):
        Sum = Sum * x
    result = result + Sum
print(result)

# determinant - linear algebra
N = int(input())
A = []
for i in range(N):
    A.append(list(map(float, input().split())))
print(round(numpy.linalg.det(A),2))

# Exception handling

for _ in range(int(input())):
    try:
        a,b = map(int,input().split())
        print(a // b)
    except Exception as e:
        print("Error Code:",e)
        
# regex + exception handling

import re;
N = int(input())
for _ in range(N):
    try:
        re.compile(input()) # used to compile regex patterns, if any exception occurs it throws it!
        Output = True
    except re.error:
        Output = False
    print(Output)

# check for float number pattern

import re;
N = int(input())
for _ in range(N):
    s = input()
    print(bool(re.match(r'^[-+]?[0-9]*\.[0-9]+$',s)))
    
# using re.split()

regex_pattern = r"[.|,]"	# Do not delete 'r'.

import re
print("\n".join(re.split(regex_pattern, input())))

# using groups, groupdict in regex

import re
m = re.search(r'([a-zA-Z0-9])\1+', input().strip())
print(m.group(1) if m else -1)

# printing factors --> scaler

def main():
    # YOUR CODE GOES HERE
    # Please take input and print output to standard input/output (stdin/stdout)
    # E.g. 'input()/raw_input()' for input & 'print' for output
    N = int(input())
    count = 0
    for i in range(1, N + 1):
        if N % i == 0:
            count += 1
    print(count)
    return 0

if __name__ == '__main__':
    main()
    
# convert binary to decimal --> scaler
def main():
    # YOUR CODE GOES HERE
    # Please take input and print output to standard input/output (stdin/stdout)
    # E.g. 'input()/raw_input()' for input & 'print' for output
    n = int(input())
    pwr = 0
    decimal = 0
    while n > 0:
        last = n %10
        n = n // 10
        decimal = decimal + (last * 2 ** pwr)
        pwr += 1
    print(decimal)
    return 0

if __name__ == '__main__':
    main()
    
# factorial of number

def main():
    # YOUR CODE GOES HERE
    # Please take input and print output to standard input/output (stdin/stdout)
    # E.g. 'input()/raw_input()' for input & 'print' for output
    N = int(input())
    factorial = 1
    # if N == 0:
    #     print(1)
    # else:
    for i in range(1, N + 1):
        factorial = factorial * i
    print(factorial)
    return 0

if __name__ == '__main__':
    main()
    
# print armstrong numbers for given range

def main():
    # YOUR CODE GOES HERE
    # Please take input and print output to standard input/output (stdin/stdout)
    # E.g. 'input()/raw_input()' for input & 'print' for output
    lower = 1
    upper = int(input())
    print(1)
    for num in range(lower, upper + 1):
        order = len(str(num))
        sum = 0

        temp = num
        while temp > 0:
            digit = temp % 10
            sum += digit ** order
            temp //= 10

        if num == sum:
            if order > 1:
                print(num)
    return 0

if __name__ == '__main__':
    main()
    
# regex - vowels between consonants

import re
v = "aeiou"
c = "qwrtypsdfghjklzxcvbnm"
print(*re.findall("(?=[%s]([%s]{2,})[%s])"%(c,v,c),input(), re.I) or [-1], sep="\n")

# regex using sub() - replaces '&&' with 'and', '||' with 'or'

import re 
for _ in range(int(input())):
    str_ = input()
    str_ = re.sub(r"(?<= )(&&)(?= )", "and", str_)
    print(re.sub(r"(?<= )(\|\|)(?= )", "or", str_))

# regex checking mobile number pattern
import re
N = int(input())
for _ in range(N):
    pattern = r'^[789]\d{9}$'
    print ( 'YES' if re.match(pattern,input()) else 'NO')
    
    
# check whether email is valid or not:

import re
import email.utils
pattern = re.compile(r"^[a-zA-Z][\w\-.]*@[a-zA-Z]+\.[a-zA-Z]{1,3}$")
for _ in range(int(input())):
    u_name, u_email = email.utils.parseaddr(input())
    if pattern.match(u_email):
          print(email.utils.formataddr((u_name, u_email)))

# Hex color code check

import re
for _ in range(int(input())):
    matches = re.findall(r':?.(#[0-9a-fA-F]{6}|#[0-9a-fA-F]{3})', input())
    if matches:
        print(*matches, sep='\n')
        
# HTML parser - 1

from html.parser import HTMLParser
class MyHTMLParser(HTMLParser):
    def handle_starttag(self, tag, attrs):        
        print ('Start :',tag)
        for ele in attrs:
            print ('->',ele[0],'>',ele[1])
            
    def handle_endtag(self, tag):
        print ('End   :',tag)
        
    def handle_startendtag(self, tag, attrs):
        print ('Empty :',tag)
        for ele in attrs:
            print ('->',ele[0],'>',ele[1])
            
MyParser = MyHTMLParser()
MyParser.feed(''.join([input().strip() for _ in range(int(input()))]))

# HTML parser - Part 2

from html.parser import HTMLParser

class MyHTMLParser(HTMLParser):
    def handle_comment(self, data):
        if "\n" in data:
            print(">>> Multi-line Comment  ", data, sep="\n")
        else:
            print(">>> Single-line Comment  ", data, sep="\n")
    def handle_data(self, data):
        if data != "\n":
            print(">>> Data", data, sep="\n")
html = ""       
for i in range(int(input())):
    html += input().rstrip()
    html += '\n'
    
parser = MyHTMLParser()
parser.feed(html)
parser.close()

# HTML tags, attributes, values

class MyHTMLParser(HTMLParser):
    def handle_starttag(self, tag, attrs):        
        print (tag)
        for ele in attrs:
            print ('->',ele[0],'>',ele[1])
            
    def handle_endtag(self, tag):
        pass
        
    def handle_startendtag(self, tag, attrs):
        print (tag)
        for ele in attrs:
            print ('->',ele[0],'>',ele[1])
            
MyParser = MyHTMLParser()
MyParser.feed(''.join([input().strip() for _ in range(int(input()))]))

# validating UID

import re

no_repeats = r"(?!.*(.).*\1)"
two_or_more_upper = r"(?=(?:.*[A-Z]){2,})"
three_or_more_digits = r"(?=(?:.*\d){3,})"
ten_alphanumerics = r"[a-zA-Z0-9]{10}"
filters = no_repeats, two_or_more_upper, three_or_more_digits, ten_alphanumerics

for uid in [input() for _ in range(int(input()))]:
    if all([re.match(f, uid) for f in filters]):
        print("Valid")
    else:
        print("Invalid")
        

# Validating Credit Card Numbers

import re

is_grouping = re.compile(r'^(?:.{4}\-){3}.{4}$').match
is_consecutive = re.compile(r'(.)\1{3}').search
is_valid = re.compile(r'^[456]\d{15}$').match

for _ in range(int(input())):
    card_no = input()
    if is_grouping(card_no):
        card_no = card_no.replace('-', '')
    if is_valid(card_no) and not is_consecutive(card_no):
        print('Valid')
    else:
        print('Invalid')
        
# MAtrix script using regex

import math
import os
import random
import re
import sys

first_multiple_input = input().rstrip().split()

n = int(first_multiple_input[0])

m = int(first_multiple_input[1])

matrix = []

for _ in range(n):
    matrix_item = input()
    matrix.append(matrix_item)
encoded_string = "".join([matrix[j][i] for i in range(m) for j in range(n)])
pat = r'(?<=[a-zA-Z0-9])[^a-zA-Z0-9]+(?=[a-zA-Z0-9])'
print(re.sub(pat,' ',encoded_string))

# classes --> dealing with complex numbers

import math

class Complex(object):
    def __init__(self, real, imaginary):
        self.real=real
        self.imaginary=imaginary
        
    def __add__(self, no):
        return Complex(self.real+no.real,self.imaginary+no.imaginary)
    def __sub__(self, no):
        return Complex(self.real-no.real,self.imaginary-no.imaginary)
        
    def __mul__(self, no):
        r=self.real*no.real-self.imaginary*no.imaginary
        i=self.real*no.imaginary+self.imaginary*no.real
        return Complex(r,i)

    def __truediv__(self, no):
        d=no.real**2+no.imaginary**2
        n=self*Complex(no.real,-1*no.imaginary)
        return Complex(n.real/d,n.imaginary/d)


    def mod(self):
        d=self.real**2+self.imaginary**2
        return Complex(math.sqrt(d),0)
    def __str__(self):
        if self.imaginary == 0:
            result = "%.2f+0.00i" % (self.real)
        elif self.real == 0:
            if self.imaginary >= 0:
                result = "0.00+%.2fi" % (self.imaginary)
            else:
                result = "0.00-%.2fi" % (abs(self.imaginary))
        elif self.imaginary > 0:
            result = "%.2f+%.2fi" % (self.real, self.imaginary)
        else:
            result = "%.2f-%.2fi" % (self.real, abs(self.imaginary))
        return result

if __name__ == '__main__':
    c = map(float, input().split())
    d = map(float, input().split())
    x = Complex(*c)
    y = Complex(*d)
    print(*map(str, [x+y, x-y, x*y, x/y, x.mod(), y.mod()]), sep='\n')
    
# classes - Class 2 - Find the Torsional Angle

import math

class Points(object):
    def __init__(self, x, y, z):
        self.x=x
        self.y=y
        self.z=z

    def __sub__(self, no):
        return  Points((self.x-no.x),(self.y-no.y),(self.z-no.z))
    def dot(self, no):
        return (self.x*no.x)+(self.y*no.y)+(self.z*no.z)
    def cross(self, no):
        return Points((self.y*no.z-self.z*no.y),(self.z*no.x-self.x*no.z),(self.x*no.y-self.y*no.x))
    def absolute(self):
        return pow((self.x ** 2 + self.y ** 2 + self.z ** 2), 0.5)

if __name__ == '__main__':
    points = list()
    for i in range(4):
        a = list(map(float, input().split()))
        points.append(a)

    a, b, c, d = Points(*points[0]), Points(*points[1]), Points(*points[2]), Points(*points[3])
    x = (b - a).cross(c - b)
    y = (c - b).cross(d - c)
    angle = math.acos(x.dot(y) / (x.absolute() * y.absolute()))

    print("%.2f" % math.degrees(angle))

# debugging progrm 1
def is_vowel(letter):
    return letter in ['a', 'e', 'i', 'o', 'u', 'y']

def score_words(words):
    score = 0
    for word in words:
        num_vowels = 0
        for letter in word:
            if is_vowel(letter):
                num_vowels += 1
        if num_vowels % 2 == 0:
            score += 2
        else:
            score += 1
    return score


n = int(input())
words = input().split()
print(score_words(words))

# debugging program 2




# using deque 
from collections import deque
def check(d):
    while d:
        big = d.popleft() if d[0]>d[-1] else d.pop()
        if not d:
            return "Yes"
        if d[-1]>big or d[0]>big:
            return "No"
    
for i in range(int(input())):
    int(input())
    d = deque(map(int,input().split()))
    print(check(d))
        
# using map and lambda function - fibonnacci series

cube = lambda x: x ** 3

def fibonacci(n):
    result = [0,1]
    for i in range(2,n):
        result.append(result[i-1]+result[i-2])
    return result[:n]

if __name__ == '__main__':
    n = int(input())
    print(list(map(cube, fibonacci(n))))
    

# validating email - using functions

def fun(s):
    try:
        username, url = s.split("@")
        website, extension = url.split(".")
    except ValueError:
        return False
    
    if username.replace("-", "").replace("_", "").isalnum() is False:
        return False
    elif website.isalnum() is False:
        return False
    elif len(extension) > 3:
        return False
    else:
        return True    

def filter_mail(emails):
    return list(filter(fun, emails))

if __name__ == '__main__':
    n = int(input())
    emails = []
    for _ in range(n):
        emails.append(input())

filtered_emails = filter_mail(emails)
filtered_emails.sort()
print(filtered_emails)

# python functionals - reduce function

from fractions import Fraction
from functools import reduce
import operator
def product(fracs):
    t = reduce(operator.mul , fracs)
    return t.numerator, t.denominator

if __name__ == '__main__':
    fracs = []
    for _ in range(int(input())):
        fracs.append(Fraction(*map(int, input().split())))
    result = product(fracs)
    print(*result)
    
# Closures and decorators - 

def wrapper(f):
    def fun(l):
        f(["+91 "+c[-10:-5]+" "+c[-5:] for c in l])
    return fun

@wrapper
def sort_phone(l):
    print(*sorted(l), sep='\n')

if __name__ == '__main__':
    l = [input() for _ in range(int(input()))]
    sort_phone(l) 
