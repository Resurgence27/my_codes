/*
 * @lc app=leetcode id=122 lang=cpp
 *
 * [122] Best Time to Buy and Sell Stock II
 */

// @lc code=start
#include <bits/stdc++.h>
using namespace std;
class Solution
{
public:
    // first test is 100% transcend others
    int maxProfit(vector<int> &prices)
    {
        //handle one element
        if (prices.size() == 1)
            return 0;

        int profit = 0;
        //handle the first small extreme
        //int buy_day = prices[0] < prices[1] ? 0 : -1;
        bool is_buy = prices[0] < prices[1] ? 1 : 0;
        if (is_buy)
            profit -= prices.front();

        for (int ptr = 1; ptr < prices.size() - 1; ptr++)
        {
            //find the small extreme
            if ((prices[ptr] < prices[ptr + 1] && prices[ptr] <= prices[ptr - 1]) || prices[ptr] == 0)
            {
                profit -= prices[ptr];
                is_buy = 1;
            }
            //find the big extreme
            else if (prices[ptr] >= prices[ptr + 1] && prices[ptr] > prices[ptr - 1])
            {
                profit += prices[ptr];
                is_buy = 0;
            }
        }
        //handle the ascending or descending array
        if (is_buy)
            profit += prices.back();
        return profit;
    }
};
// @lc code=end
