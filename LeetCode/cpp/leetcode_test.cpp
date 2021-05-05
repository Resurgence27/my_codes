#include <bits/stdc++.h>
using namespace std;

bool cmp_vec(vector<int> &v1, vector<int> &v2)
{
    return ((v1[0] < v2[0]) || (v1[0] == v2[0] && v1[1] > v2[1]));
}

int main()
{
    string S = "ababcbacadefegdehijhklij";

    // clock_t start, end;
    // map<char, vector<int>> count_char;
    // start = clock();

    // for (int i = 0; i < S.length(); i++)
    // {
    //     if (count_char.find(S[i]) == count_char.end())
    //         count_char[S[i]] = vector<int>{i, i};
    //     else
    //         count_char[S[i]][1] = i;
    // }
    // end = clock();

    // for (auto p : count_char)
    // {
    //     //cout << p.first << " " << p.second[0] << " " << p.second[1] << endl;
    // }

    // cout << "map size is " << sizeof(count_char) << endl;
    // cout << "run time is " << double(end - start) / CLOCKS_PER_SEC << endl;

    vector<int> alpha_last(26);

    for (int i = 0; i < S.length(); i++)
    {
        alpha_last[S[i] - 'a'] = i;
    }

    vector<int> result;
    int start = 0, end = 0;
    result.reserve(26);
    for (int ptr = 0; ptr < alpha_last.size(); ptr++)
    {
        end = max(end, alpha_last[S[ptr] - 'a']);
        if (ptr == end)
        {
            result.push_back(end - start + 1);
            start = end + 1;
        }
    }

    return 0;
}