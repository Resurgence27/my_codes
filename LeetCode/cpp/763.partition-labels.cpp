/*
 * @lc app=leetcode id=763 lang=cpp
 *
 * [763] Partition Labels
 */

// @lc code=start
#include <bits/stdc++.h>
using namespace std;
class Solution
{
public:
    vector<int> partitionLabels(string S)
    {
        //summary:refer to the online thought

        //handle one char
        if (S.length() == 1)
            return vector<int>(1, 1);

        //find the char when first and last appearing
        vector<int> alpha_last(26);

        for (int i = 0; i < S.length(); i++)
        {
            alpha_last[S[i] - 'a'] = i;
        }

        vector<int> result;
        int start = 0, end = 0;
        result.reserve(26);
        for (int ptr = 0; ptr < S.size(); ptr++)
        {
            end = max(end, alpha_last[S[ptr] - 'a']);
            if (ptr == end)
            {
                result.push_back(end - start + 1);
                start = end + 1;
            }
        }

        return result;
    }
};
// @lc code=end
