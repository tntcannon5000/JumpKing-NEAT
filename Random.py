def getLargestNumber(num):
  """
  Finds the largest number possible by swapping adjacent digits with the same parity.

  Args:
      num: A string representation of the number.

  Returns:
      The largest number possible by swapping digits in num.
  """
  n = len(num)
  for i in range(1, n - 1):
    if (int(num[i]) % 2 == int(num[i-1]) % 2) and (num[i] > num[i+1]):
      num = num[:i] + num[i+1] + num[i] + num[i+2:]
  return num

print(getLargestNumber('0082663'))