#!/usr/bin/env python
# coding: utf-8

# # LeetCode link of my account
# https://leetcode.com/nupriyasaxena/

# In[42]:


#1768. Merge Strings Alternately

class Solution:
    def mergeAlternately(self, word1: str, word2: str) -> str:
        return ''.join(a + b for a, b in zip_longest(word1, word2, fillvalue=''))


# In[27]:


#389. Find the Difference

class Solution:
    def findTheDifference(self, s: str, t: str) -> str:
        cnt = Counter(s)
        for c in t:
            cnt[c] -= 1
            if cnt[c] < 0:
                return c


# In[28]:


#28. Find the Index of the First Occurrence in a String

class Solution:
    def strStr(self, haystack: str, needle: str) -> int:
        try: 
            index = haystack.find(needle)
            return index
        except ValueError:
            return -1


# In[29]:


#242. Valid Anagram

class Solution:
    def isAnagram(self, s: str, t: str) -> bool:
        if len(s) != len(t):
            return False
        chars = [0] * 26
        for i in range(len(s)):
            chars[ord(s[i]) - ord('a')] += 1
            chars[ord(t[i]) - ord('a')] -= 1
        return all(c == 0 for c in chars)


# In[31]:


#459. Repeated Substring Pattern

class Solution:
    def repeatedSubstringPattern(self, s: str) -> bool:
        return (s + s).index(s, 1) < len(s)


# In[33]:


#283. Move Zeroes

class Solution:
    def moveZeroes(self, nums: list[int]) -> None:
        """
        Do not return anything, modify nums in-place instead.
        """
        left, n = 0, len(nums)
        for right in range(n):
            if nums[right] != 0:
                nums[left], nums[right] = nums[right], nums[left]
                left += 1


# In[36]:


#66. Plus One

class Solution:
    def plusOne(self, digits: list[int]) -> list[int]:
        n = len(digits)
        for i in range(n - 1, -1, -1):
            if digits[i] < 9:
                digits[i] += 1
                return digits
            
            digits[i] = 0

        return [1] + digits
        


# In[37]:


#1822. Sign of the Product of an Array

class Solution:
    def arraySign(self, nums: list[int]) -> int:
        res = 1
        
        for num in nums:
            res *= num
        
        if res == 0: return 0
        elif res > 0 : return 1
        else: return -1


# In[38]:


#1502. Can Make Arithmetic Progression From Sequence

def checkAP(arr, n):
    if (n == 1): return True
 
    arr.sort()
    d = arr[1] - arr[0]
    for i in range(2, n):
        if (arr[i] - arr[i-1] != d):
            return False
 
    return True
arr = [ 20, 15, 5, 0, 10 ]
n = len(arr)
print("Yes") if(checkAP(arr, n)) else print("No")


# In[40]:


#896. Monotonic Array

class Solution:
    def isMonotonic(self, nums: list[int]) -> bool:
        incr = all(a <= b for a, b in pairwise(nums))
        decr = all(a >= b for a, b in pairwise(nums))
        return incr or decr


# In[41]:


#13. Roman to Integer

class Solution:
    def romanToInt(self, s: str) -> int:
        romans = {'I': 1, 'V': 5, 'X': 10, 'L': 50, 'C': 100, 'D': 500, 'M': 1000}
        ans = 0
        for i in range(len(s) - 1):
            if romans[s[i]] < romans[s[i + 1]]:
                ans -= romans[s[i]]
            else:
                ans += romans[s[i]]
        return ans + romans[s[-1]]


# In[7]:


#58. Length of Last Word

class Solution:
    def lengthOfLastWord(self, s: str) -> int:
        i = len(s) - 1
        while i >= 0 and s[i] == ' ':
            i -= 1
        j = i
        while j >= 0 and s[j] != ' ':
            j -= 1
        return i - j


# In[8]:


#709. To Lower Case

class Solution:
    def toLowerCase(self, s: str) -> str:
        return "".join([chr(ord(c) | 32) if c.isupper() else c for c in s])


# In[10]:


# Baseball game

class Solution:
    def calPoints(self, ops: list[str]) -> int:
        stk = []
        for op in ops:
            if op == '+':
                stk.append(stk[-1] + stk[-2])
            elif op == 'D':
                stk.append(stk[-1] << 1)
            elif op == 'C':
                stk.pop()
            else:
                stk.append(int(op))
        return sum(stk)


# In[11]:


#657. Robot Return to Origin

class Solution:
    def judgeCircle(self, moves: str) -> bool:
        x = y = 0
        for c in moves:
            if c == 'R':
                x += 1
            elif c == 'L':
                x -= 1
            elif c == 'U':
                y += 1
            elif c == 'D':
                y -= 1
        return x == 0 and y == 0


# In[13]:


#1275. Find Winner on a Tic Tac Toe Game

class Solution:
    def tictactoe(self, moves: list[list[int]]) -> str:
        n = len(moves)
        counter = [0] * 8
        for i in range(n - 1, -1, -2):
            row, col = moves[i][0], moves[i][1]
            counter[row] += 1
            counter[col + 3] += 1
            if row == col:
                counter[6] += 1
            if row + col == 2:
                counter[7] += 1
            if (
                counter[row] == 3
                or counter[col + 3] == 3
                or counter[6] == 3
                or counter[7] == 3
            ):
                return "A" if (i % 2) == 0 else "B"
        return "Draw" if n == 9 else "Pending"


# In[14]:


#1041. Robot Bounded In Circle

class Solution:
    def isRobotBounded(self, instructions: str) -> bool:
        cur, direction = 0, [0] * 4
        for ins in instructions:
            if ins == 'L':
                cur = (cur + 1) % 4
            elif ins == 'R':
                cur = (cur + 3) % 4
            else:
                direction[cur] += 1
        return cur != 0 or (
            direction[0] == direction[2] and direction[1] == direction[3]
        )


# In[15]:


#1672. Richest Customer Wealth
class Solution:
    def maximumWealth(self, accounts: list[list[int]]) -> int:
        return max(sum(v) for v in accounts)


# In[16]:


#1572. Matrix Diagonal Sum
class Solution:
    def diagonalSum(self, mat: list[list[int]]) -> int:
        n = len(mat)
        res = 0
        for i in range(n):
            res += mat[i][i] + (0 if n - i - 1 == i else mat[i][n - i - 1])
        return res


# In[19]:


#54. Spiral Matrix
class Solution:
    def spiralOrder(self, matrix: list[list[int]]) -> list[int]:
        m, n = len(matrix), len(matrix[0])
        ans = []
        top, bottom, left, right = 0, m - 1, 0, n - 1
        while left <= right and top <= bottom:
            ans.extend([matrix[top][j] for j in range(left, right + 1)])
            ans.extend([matrix[i][right] for i in range(top + 1, bottom + 1)])
            if left < right and top < bottom:
                ans.extend([matrix[bottom][j] for j in range(right - 1, left - 1, -1)])
                ans.extend([matrix[i][left] for i in range(bottom - 1, top, -1)])
            top, bottom, left, right = top + 1, bottom - 1, left + 1, right - 1
        return ans


# In[20]:


#73. Set Matrix Zeroes
class Solution:
    def setZeroes(self, matrix: list[list[int]]) -> None:
        m, n = len(matrix), len(matrix[0])
        rows = [0] * m
        cols = [0] * n
        for i, row in enumerate(matrix):
            for j, v in enumerate(row):
                if v == 0:
                    rows[i] = cols[j] = 1
        for i in range(m):
            for j in range(n):
                if rows[i] or cols[j]:
                    matrix[i][j] = 0


# In[21]:


#1523. Count Odd Numbers in an Interval Range
class Solution:
    def countOdds(self, low: int, high: int) -> int:
        return ((high + 1) >> 1) - (low >> 1)


# In[22]:


#1491. Average Salary Excluding the Minimum and Maximum Salary
class Solution:
    def average(self, salary: list[int]) -> float:
        s = sum(salary) - min(salary) - max(salary)
        return s / (len(salary) - 2)


# In[24]:


#860. Lemonade Change
class Solution:
    def lemonadeChange(self, bills: list[int]) -> bool:
        l=[5]
        if bills[0]==10 or bills[0]==20:
            return False
        else:
            for i in range(1,len(bills)):
                if bills[i]==5:
                    l.append(bills[i])
                elif bills[i]==10:
                    if 5 in l:
                        l.remove(5)
                        l.append(10)
                    else:
                        return False
                else:
                    if 5 in l and 10 in l:
                        l.remove(5)
                        l.remove(10)
                        l.append(20)
                    elif l.count(5)>=3:
                        l.remove(5)
                        l.remove(5)
                        l.remove(5)
                        l.append(20)
                    else:
                        print(l)
                        return False
            return True


# In[26]:


#976. Largest Perimeter Triangle
class Solution:
    def largestPerimeter(self, nums: list[int]) -> int:
        nums = sorted(nums)
        for i in range(len(nums) - 1, 1, -1):
            if nums[i - 2] + nums[i - 1] > nums[i]:
                return nums[i - 2] + nums[i - 1] + nums[i]
        return 0


# In[27]:


#1232. Check If It Is a Straight Line
class Solution:
    def checkStraightLine(self, coordinates: list[list[int]]) -> bool:
        x1, y1 = coordinates[0]
        x2, y2 = coordinates[1]
        for x, y in coordinates[2:]:
            if (x - x1) * (y2 - y1) != (y - y1) * (x2 - x1):
                return False
        return True


# In[28]:


#67. Add Binary
class Solution:
    def addBinary(self, a: str, b: str) -> str:
        return bin(int(a, 2) + int(b, 2))[2:]


# In[29]:


#43. Multiply Strings
class Solution:
    def multiply(self, num1: str, num2: str) -> str:
        if num1 == "0" or num2 == "0":
            return "0"
        m, n = len(num1), len(num2)
        arr = [0] * (m + n)
        for i in range(m - 1, -1, -1):
            a = int(num1[i])
            for j in range(n - 1, -1, -1):
                b = int(num2[j])
                arr[i + j + 1] += a * b
        for i in range(m + n - 1, 0, -1):
            arr[i - 1] += arr[i] // 10
            arr[i] %= 10
        i = 0 if arr[0] else 1
        return "".join(str(x) for x in arr[i:])


# In[30]:


#50. Pow(x, n)
class Solution:
    def myPow(self, x: float, n: int) -> float:
        def qmi(a, k):
            res = 1
            while k:
                if k & 1:
                    res *= a
                a *= a
                k >>= 1
            return res

        return qmi(x, n) if n >= 0 else 1 / qmi(x, -n)


# In[ ]:




