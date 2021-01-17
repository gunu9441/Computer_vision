# Print 

## Print Function

### 문자열, 정수형 동시 출력
print('fist', 'second', 'third', 1, 2, 3)

### 원소 사이에 '-' 추가
print('fist', 'second', 'third', 1, 2, 3, sep='-')

### 다른 열에 출력
print('first line')
print('second line')

### 같은 열에 출력
print('first line', end=" ---> ")
print('second line')

## Print Method Of String
print("Hello")
print('Hello')
print("Hello 'yejin'")
print('Hello "yejin"')
print('Hello \'yejin\'')
print("Hello \"yejin\"")
print("-" * 14)
print('-' * 14)

## Print according to a Format
### allocate specific string using print format
print('My name is {}'.format('gunu'))
print('My name is {1}, Your name is {0}'.format('yejin', 'gunu')) 
print('My girl friend is {0}, Your name is {0}'.format('yejin', 'gunu')) 

### use end function in print format
print('My name is {mine}, Your name is {yours}'.format(
    yours = 'yejin', mine = 'gunu'
    ), end = '--->')
print(' ♥')

## Coordinate Printing String length
### String: 왼쪽   정렬
### Number: 오른쪽 정렬
print('My name is {:10}, Your name is {:10}!'.format('gunu', 'yejin'))
print('My age is {:10}, Your age is {:10}'.format(22, 21))

### :< 왼쪽 정렬, :> 오른쪽 정렬, :^ 가운데 정렬
print('My name is {:<10}, Your name is {:>10}!'.format('gunu', 'yejin'))
print('My age is {:^10}, Your age is {:<10}'.format(22, 21))

