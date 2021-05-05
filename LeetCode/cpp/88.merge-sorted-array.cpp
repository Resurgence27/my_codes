// @before-stub-for-debug-begin
#include <vector>
#include <string>
// #include "commoncppproblem88.h"

using namespace std;
// @before-stub-for-debug-end

/*
 * @lc app=leetcode id=88 lang=cpp
 *
 * [88] Merge Sorted Array
 */

// @lc code=start
#include <bits/stdc++.h>
using namespace std;
class Solution
{
public:
    void merge(vector<int> &nums1, int m, vector<int> &nums2, int n)
    {
        // nums2 is null
        if (n == 0)
            return;
        int ptr_s1 = m - 1, ptr_s2 = n - 1;
        while (ptr_s1 >= 0 || ptr_s2 >= 0)
        {
            //nums2 all added
            if (ptr_s2 < 0)
                return;
            else if (ptr_s1 < 0) //the remain s2 is smaller than all nums1
            {
                nums1[ptr_s1 + ptr_s2 + 1] = nums2[ptr_s2];
                ptr_s2--;
            }
            else
            {
                //select the bigger one
                //from the last index to add
                //and guarantee nums2 ends before nums1
                if (nums1[ptr_s1] > nums2[ptr_s2])
                {
                    nums1[ptr_s1 + ptr_s2 + 1] = nums1[ptr_s1];
                    ptr_s1--;
                }
                else
                {
                    nums1[ptr_s1 + ptr_s2 + 1] = nums2[ptr_s2];
                    ptr_s2--;
                }
            }
        }
        return;
    }
};
// @lc code=end
