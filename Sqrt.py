__author__ = 'Michael'
"""
Sqrt
Given an integer number N, compute its square root without using any math library functions and print the result to
standard output.
Please round the result downwards to the nearest integer (e.g both 7.1 and 7.9 are rounded to 7)
Expected complexity: O(logN), O(1)
Example input:
N: 17
Example output:
4
"""
class Track_Calls:
    def __init__(self,f):
        self._f = f
        self.calls = 0

    def __call__(self,*args,**kargs):  # bundle arbitrary arguments
        self.calls += 1
        return  self._f(*args,**kargs) # unbundle arbitrary arguments

class Illustrate_Recursive:
    def __init__(self,f):
        self._f     = f
        self._trace = False
    def illustrate(self,*args,**kargs):
        self._indent = 0
        self._trace  = True
        answer       = self.__call__(*args,**kargs)
        self._trace  = False
        return answer
    def __call__(self,*args,**kargs):
        if self._trace:
            if self._indent == 0:
                print('Starting recursive illustration'+30*'-')
            print (self._indent*"."+"calling", self._f.__name__+str(args)+str(kargs))
            self._indent += 2
        answer = self._f(*args)
        if self._trace:
            self._indent -= 2
            print (self._indent*"."+self._f.__name__+str(args)+str(kargs)+" returns", answer)
            if self._indent == 0:
                print('Ending recursive illustration'+30*'-')
        return answer


@Illustrate_Recursive
def compute_sqrt(n):
    # Write your code here
    # To print results to the standard output you can use print
    # Example: print "Hello world!"
    answers = range(1, n + 1)
    print(find_sqrt(answers, n))

@Illustrate_Recursive
def find_sqrt(answers, n):
    # print(answers)
    middle = answers[int(len(answers) / 2)]
    # print(middle, middle ** 2)
    if middle ** 2 == n or len(answers) == 1:
        return middle
    elif middle ** 2 < n:
        return find_sqrt(answers[int(len(answers) / 2):], n)
    elif middle ** 2 > n:
        return find_sqrt(answers[:int(len(answers) / 2)], n)


compute_sqrt(17)
print(compute_sqrt(17))