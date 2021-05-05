/*
 * @lc app=leetcode id=135 lang=cpp
 *
 * [135] Candy
 *
 * https://leetcode.com/problems/candy/description/
 *
 * algorithms
 * Hard (33.41%)
 * Likes:    1442
 * Dislikes: 197
 * Total Accepted:    155.5K
 * Total Submissions: 465.4K
 * Testcase Example:  '[1,0,2]'
 *
 * There are n children standing in a line. Each child is assigned a rating
 * value given in the integer array ratings.
 * 
 * You are giving candies to these children subjected to the following
 * requirements:
 * 
 * 
 * Each child must have at least one candy.
 * Children with a higher rating get more candies than their neighbors.
 * 
 * 
 * Return the minimum number of candies you need to have to distribute the
 * candies to the children.
 * 
 * 
 * Example 1:
 * 
 * 
 * Input: ratings = [1,0,2]
 * Output: 5
 * Explanation: You can allocate to the first, second and third child with 2,
 * 1, 2 candies respectively.
 * 
 * 
 * Example 2:
 * 
 * 
 * Input: ratings = [1,2,2]
 * Output: 4
 * Explanation: You can allocate to the first, second and third child with 1,
 * 2, 1 candies respectively.
 * The third child gets 1 candy because it satisfies the above two
 * conditions.
 * 
 * 
 * 
 * Constraints:
 * 
 * 
 * n == ratings.length
 * 1 <= n <= 2 * 10^4
 * 0 <= ratings[i] <= 2 * 10^4
 * 
 * 
 */

// @lc code=start
#include <bits/stdc++.h>
using namespace std;
class Solution
{
public:
    int candy(vector<int> &ratings)
    {

        //find the minest index
        auto min_it = min_element(ratings.begin(), ratings.end());
        // set the min rating is 1
        int total = 1;
        int front = total, back = total;
        auto forward_it = min_it - 1;
        auto backward_it = min_it + 1;
        auto is_1 = min_it;
        //obtain the element of second and reverse second
        while (forward_it >= ratings.begin())
        {
            if (*forward_it > *(forward_it + 1))
            {
                front++;
            }
            else
            {
                if (forward_it == ratings.begin())
                {
                    total += 1;
                    break;
                }

                if (*forward_it > *(forward_it - 1))
                {
                    if (*forward_it < *(forward_it + 1))
                    {
                        if (front <= 2)
                        {
                            for (auto ptr = forward_it + 1; ptr < is_1; ptr++)
                            {
                                total += 1;
                            }
                        }
                        front = 2;
                    }
                    else
                        front = 1;
                }
                else
                {
                    is_1 = forward_it;
                    front = 1;
                }
            }
            total += front;
            forward_it--;
        }

        while (backward_it <= ratings.end() - 1)
        {
            if (*backward_it > *(backward_it - 1))
            {
                back++;
            }
            else
            {
                if (backward_it == ratings.end() - 1)
                {
                    total += 1;
                    break;
                }

                if (*backward_it > *(backward_it + 1))
                {
                    if (*backward_it < *(backward_it - 1))
                    {
                        if (back <= 2)
                        {
                            for (auto ptr = is_1; ptr < backward_it; ptr++)
                            {
                                total += 1;
                            }
                        }
                        back = 2;
                    }
                    else
                        back = 1;
                }
                else
                {
                    is_1 = backward_it;
                    back = 1;
                }
            }
            total += back;
            backward_it++;
        }
        return total;
    }
}

;
// @lc code=end
