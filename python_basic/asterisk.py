def practiceAsterisk(first, second, third : int = None, fourth = None ) -> int :
# first, second : positional arguments / third, fourth : keyword arguments
    dict = {}
    dict['first'], dict['second'] = first, second
    dict['third'] = third if third is not None else 'Nothing'
    dict['fourth'] = fourth if fourth is not None else 'Nothing'
    print(dict)

def practicePositionalInVariadic(*args):
    print(args)
    list = [i for i in args]
    print(list)


def practicekeywordInVariadic(**kwargs):
    print(kwargs)

def practiceBothInVariadic(*args, **kwargs):
    print(args)
    print(kwargs)

def practiceUnpacking(*numbers, **people):
    print(numbers)
    print(people)

def function(*list):
    print(list)

practiceAsterisk(1, second = 2)
practiceAsterisk(1,2,3)
practiceAsterisk(1, 2, third = 3)
practiceAsterisk(1, 2, 3, fourth = 4)
practiceAsterisk(1, 2, 3, 4)

practicePositionalInVariadic('first', 'second', 'third', 'fourth')
practicekeywordInVariadic(first = 'homeless', second = 'child', third = 'parents', fourth = 'brider')
practiceBothInVariadic(1, 2, 'hello', fourth = 'baby', fifth = 'lover')

dict = {'hello': 'DS.wook', 'bye': 'Lee Yu'}
list = [1,2,3,4,5]
list_1 = [8,9]
# print(**dict) error
print(*list)
practiceUnpacking(*list)
practiceUnpacking(**dict)
practiceUnpacking(*list, **dict)

function(list, list_1)