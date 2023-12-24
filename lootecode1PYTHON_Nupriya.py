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


# In[ ]:




