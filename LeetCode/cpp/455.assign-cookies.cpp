/*
 * @lc app=leetcode id=455 lang=cpp
 *
 * [455] Assign Cookies
 */

// @lc code=start
#include <bits/stdc++.h>
using namespace std;
class Solution
{
public:
    int findContentChildren(vector<int> &g, vector<int> &s)
    {
        int meet = 0;
        sort(s.begin(), s.end());
        sort(g.begin(), g.end());
        //try to use int to replace the iterator
        //iterator accounts for 8 bytes
        auto ptr_s = s.rbegin();
        for (auto ptr = g.rbegin(); ptr != g.rend(); ptr++)
        {
            if (ptr_s != s.rend())
            {
                if (*ptr_s >= *ptr)
                {
                    meet++;
                    ptr_s++;
                }
            }
            else
                break;
        }
        return meet;
    }
};
// @lc code=end
