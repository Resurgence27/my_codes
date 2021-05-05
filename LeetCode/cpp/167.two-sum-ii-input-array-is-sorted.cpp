/*
 * @lc app=leetcode id=167 lang=cpp
 *
 * [167] Two Sum II - Input array is sorted
 */

// @lc code=start
#include <bits/stdc++.h>
using namespace std;
class Solution
{
public:
    vector<int> twoSum(vector<int> &numbers, int target)
    {
        int left = 0, right = numbers.size() - 1;
        vector<int> result(2);
        while (left != right)
        {
            if (numbers[left] + numbers[right] == target)
            {
                result[0] = left + 1;
                result[1] = right + 1;
                break;
            }
            else if (numbers[left] + numbers[right] > target)
                right--;
            else
                left++;
        }
        return result;
    }
};
// @lc code=end
