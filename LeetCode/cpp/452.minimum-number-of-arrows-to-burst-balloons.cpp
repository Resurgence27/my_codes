/*
 * @lc app=leetcode id=452 lang=cpp
 *
 * [452] Minimum Number of Arrows to Burst Balloons
 */

// @lc code=start
#include <bits/stdc++.h>
using namespace std;
class Solution
{
public:
    int findMinArrowShots(vector<vector<int>> &points)
    {
        //handle one element
        if (points.size() == 0)
            return 1;

        //at least 1
        int arrow_num = 1;
        sort(points.begin(), points.end(), [](vector<int> &v1, vector<int> &v2) { return (v1[1] < v2[1]); });

        int interval_r = points[0][1];
        for (int ptr = 0; ptr < points.size(); ptr++)
        {
            if (interval_r < points[ptr][0])
            {
                arrow_num++;
                interval_r = points[ptr][1];
            }
        }
        return arrow_num;
    }
};
// @lc code=end
