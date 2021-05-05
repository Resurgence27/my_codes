// @before-stub-for-debug-begin
#include <vector>
#include <string>
// #include "commoncppproblem435.h"

using namespace std;
// @before-stub-for-debug-end

/*
 * @lc app=leetcode id=435 lang=cpp
 *
 * [435] Non-overlapping Intervals
 */

// @lc code=start
#include <bits/stdc++.h>
using namespace std;
class Solution
{
public:
    int eraseOverlapIntervals(vector<vector<int>> &intervals)
    {
        if (intervals.size() == 1)
            return 0;

        int move_num = 0;

        sort(intervals.begin(), intervals.end(), [](vector<int> &v1, vector<int> &v2) { return (v1[1] < v2[1]); });
        int interval_r = intervals[0][1];
        for (int p = 1; p < intervals.size(); p++)
        {
            if (interval_r > intervals[p][0])
            {
                move_num++;
            }
            else
            {
                interval_r = intervals[p][1];
            }
        }

        return move_num;
    }
};
// @lc code=end
