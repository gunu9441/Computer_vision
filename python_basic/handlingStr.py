string = 'banana'
print('string: {}'.format(string))
print('lenght: {}'.format(len(string)))
print()

string = 'I have a pen'
print('string: {}'.format(string))

splitedString = string.split()
print('split string: {}'.format(splitedString))
print()

string = 'I-have-a-pen'
print('string: {}'.format(string))

splitedString = string.split('-')
print('split string: {}'.format(splitedString))
print()

string = 'I have a pen'
print('string: {}'.format(string))

upperString = string.upper()
print('upper string: {}'.format(upperString))
print()

string = 'I have a pen'
print('string: {}'.format(string))

lowerString = string.lower()
print('lower string: {}'.format(lowerString))
print()

a = '01-sample.png'
b = '02-sample.jpg'
c = '03-sample.jpg'
print('a: {} \nb: {}\nc: {}\n'.format(a, b, c))

print('Does a start with 01?: {}'.format(a.startswith('01')))
print('Does b end with png?: {}'.format(b.endswith('.png')))
print('Does c end with pdf?: {}'.format(c.endswith('.pdf')))
print()

mylist = [a, b, c]
print('list: {}'.format(mylist))
result = [i for i in mylist if i.endswith('.jpg')]
print('string which contains \'.jpg\': {}'.format(result))
print()

a = '01-sample.png'
b = '02-sample.jpg'
c = '03-sample.jpg'
print('a: {} \nb: {}\nc: {}\n'.format(a, b, c))

mylist = [a, b, c]
print('list: {}'.format(mylist))

png_list = []
png_list = [i.replace('jpg', 'png') for i in mylist]
print('png list: {}'.format(png_list))
print()

string = '  01-sample.png'
print('list        : {}'.format(string))

print('strip string: {}'.format(string.strip()))