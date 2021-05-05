/*
 * @lc app=leetcode id=605 lang=cpp
 *
 * [605] Can Place Flowers
 */

// @lc code=start
#include <bits/stdc++.h>
using namespace std;
class Solution
{
public:
    bool canPlaceFlowers(vector<int> &flowerbed, int n)
    {
        //handle one element
        if (flowerbed.size() == 1)
        {
            if (flowerbed[0] == 0)
                return true;
            else
            {
                if (n >= 1)
                    return false;
                return true;
            }
        }

        int count_zero, count_flower = 0;
        //handle first 0
        count_zero = flowerbed[0] == 0 ? 1 : 0;
        for (int ptr = 0; ptr < flowerbed.size(); ptr++)
        {
            if (flowerbed[ptr] == 0)
            {
                count_zero++;
                //handle end 0
                if (ptr == flowerbed.size() - 1)
                    count_zero++;
            }
            else
                count_zero = 0;
            if (count_zero >= 3)
            {
                count_flower++;
                count_zero = 1;
            }
            if (count_flower == n)
                return true;
        }
        return false;
    }
};
// @lc code=end
