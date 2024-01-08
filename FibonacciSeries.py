n = int(input('Enter no. of numbers you would like to see from fibonacci series: '))
first = 0
second = 1
print(first)
print(second)
for i in range(n):
  third = first + second
  print(third)
  first = second
  second = third
