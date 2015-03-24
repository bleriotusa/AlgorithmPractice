
class Track_Calls:
    def __init__(self,f):
        self._f = f
        self.calls = 0
    
    def __call__(self,*args,**kargs):  # bundle arbitrary arguments
        self.calls += 1
        return  self._f(*args,**kargs) # unbundle arbitrary arguments


# These versions of memoize require only *args to be able to cache arguments 

class Memoize:
    def __init__(self,f):
        self._f = f
        self.cache = {}
    def __call__(self,*args):
        if args in self.cache:
            return self.cache[args]
        else:
            answer = self._f(*args)
            self.cache[args] = answer
        return answer


def memoize(f):
    cache = {}
    def wrapper(*args):
        if args in cache: 
            return cache[args]
        else:
            answer = f(*args)
            cache[args] = answer
        return answer
    return wrapper

def prints(f):
    def printer(*args):
        print('function called')
        return f(*args)
    return printer

class Prints:
    def __init__(self,f):
        self._f = f
    def __call__(self,*args):
        print('function called, with args {}'.format(args))
        return self._f(*args)


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
    


if __name__ == '__main__':
    import sys 
#     print('Default recursion limit =', sys.getrecursionlimit())
    sys.setrecursionlimit(5000)
#     print('Default recursion limit =',sys.getrecursionlimit())

    @Track_Calls
    #Memoize
    #@memoize
    @Illustrate_Recursive
    @prints
    def factorial(n):
        if n == 0:
            return 1
        else:
            return n*factorial(n-1)
           
#     print(factorial.illustrate(10))
#     factorial(5)
#     print(factorial.calls)

    @Track_Calls
    @Memoize
#     @memoize
#     @Illustrate_Recursive
    @Prints

    def fib(n):
        assert n>=0, 'fib cannot have negative n('+str(n)+')'
        if    n == 0: return 1
        elif  n == 1: return 1
        else:         return fib(n-1) + fib(n-2)
    print(fib(30))
#     print(fib.calls)
    
#     for i in range(5500):
#         print(i,fib(i),fib.calls)
#         fib.calls = 0
#     print(fib.calls)
#     print(fib.illustrate(5))
    @Track_Calls
    def cut_rod(p, n):
        if n == 0:
            return 0
        q = float("-infinity")
        for i in range(1,n+1):
            q = max(q, p[i] + cut_rod(p, n-i-1))
        return q
    print('answer',cut_rod([1,5,8,9,10,17,17,20,24,30], 4))
    print('calls',cut_rod.calls)
    @Track_Calls
    def cut_rod2(p,n):
        if n==0:
            return 0
        q = float("-infinity")
        for i in range(1,n+1):
            q = max(q, p[i], cut_rod2(p,n-i))
        return q
#     print(cut_rod2([1,5,8,9,10,17,17,20,24,30], 9))
#     print(cut_rod2.calls)
