/*
 * @lc app=leetcode id=142 lang=cpp
 *
 * [142] Linked List Cycle II
 */

// @lc code=start
/**
 * Definition for singly-linked list.
 * struct ListNode {
 *     int val;
 *     ListNode *next;
 *     ListNode(int x) : val(x), next(NULL) {}
 * };
 */
#include <bits/stdc++.h>
using namespace std;

struct ListNode
{
    int val;
    ListNode *next;
    ListNode(int x) : val(x), next(NULL) {}
};

class Solution
{
public:
    ListNode *detectCycle(ListNode *head)
    {
        //as one element
        if (head->next == NULL)
            return NULL;

        ListNode *fast = head->next;
        while (fast != head || head != NULL)
        {
            if (fast->next == NULL)
                return NULL;
            fast = fast->next;
        }
    }
};
// @lc code=end
