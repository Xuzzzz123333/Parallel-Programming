#include "PCFG.h"
#include <fstream>
#include <cctype>
#include <algorithm>
#include <pthread.h>
#include <vector>

#ifndef NUM_THREADS
#define NUM_THREADS 4
#endif

// 这个文件里面的各函数你都不需要完全理解，甚至根本不需要看
// 从学术价值上讲，加速模型的训练过程是一个没什么价值的问题，因为我们一般假定统计学模型的训练成本较低
// 但是，假如你是一个投稿时顶着ddl做实验的倒霉研究生/实习生，提高训练速度就可以大幅节省你的时间了
// 所以如果你愿意，也可以尝试用多线程加速训练过程

/**
 * 怎么加速PCFG训练过程？据助教所知，没有公开文献提出过有效的加速方法（因为这么做基本无学术价值）
 * 
 * 但是统计学模型好就好在其数据是可加的。例如，假如我把数据集拆分成4个部分，并行训练4个不同的模型。
 * 然后我可以直接将四个模型的统计数据进行简单加和，就得到了和串行训练相同的模型了。
 * 
 * 说起来容易，做起来不一定容易，你可能会碰到一系列具体的工程问题。如果你决定加速训练过程，祝你好运！
 * 
 */
struct TrainThreadArg
{
    const vector<string> *passwords;
    int start;
    int end;
    model *local_model;
};

void *TrainThreadFunc(void *arg)
{
    TrainThreadArg *p = (TrainThreadArg *)arg;

    for (int i = p->start; i < p->end; i += 1)
    {
        p->local_model->parse((*(p->passwords))[i]);
    }

    pthread_exit(NULL);
}
// 训练的wrapper，实际上就是读取训练集
void model::train(string path)
{
    cout << "Training..." << endl;
    cout << "Training phase 1: reading passwords..." << endl;

    vector<string> passwords;
    passwords.reserve(3100000);

    string pw;
    ifstream train_set(path);

    int lines = 0;

    while (train_set >> pw)
    {
        lines += 1;

        if (lines % 10000 == 0)
        {
            // 为了减少 IO 对训练时间的影响，这里不再每 10000 行都输出进度
            // cout << "Lines processed: " << lines << endl;

            // 保持原始代码的读取上限逻辑
            if (lines > 3000000)
            {
                break;
            }
        }

        passwords.emplace_back(pw);
    }

    cout << "Training passwords loaded: " << passwords.size() << endl;
    cout << "Training phase 1: parallel parsing passwords..." << endl;

    // 清空当前模型，防止重复 train 时叠加旧数据
    preterminals.clear();
    preterm_freq.clear();
    letters.clear();
    letters_freq.clear();
    digits.clear();
    digits_freq.clear();
    symbols.clear();
    symbols_freq.clear();
    ordered_pts.clear();
    total_preterm = 0;

    int thread_count = NUM_THREADS;

    if (thread_count < 1)
    {
        thread_count = 1;
    }

    if (thread_count > (int)passwords.size())
    {
        thread_count = passwords.size();
    }

    vector<model> local_models(thread_count);
    vector<pthread_t> handles(thread_count);
    vector<TrainThreadArg> args(thread_count);

    int n = passwords.size();

    for (int t = 0; t < thread_count; t += 1)
    {
        int start = n * t / thread_count;
        int end = n * (t + 1) / thread_count;

        args[t].passwords = &passwords;
        args[t].start = start;
        args[t].end = end;
        args[t].local_model = &local_models[t];

        pthread_create(&handles[t], NULL, TrainThreadFunc, &args[t]);
    }

    for (int t = 0; t < thread_count; t += 1)
    {
        pthread_join(handles[t], NULL);
    }

    cout << "Training phase 1: merging local models..." << endl;

    // 合并每个线程训练出的局部模型
    for (int t = 0; t < thread_count; t += 1)
    {
        model &lm = local_models[t];

        // 合并 preterminal 统计
        total_preterm += lm.total_preterm;

        for (int i = 0; i < (int)lm.preterminals.size(); i += 1)
        {
            PT pt = lm.preterminals[i];

            int id = FindPT(pt);

            if (id == -1)
            {
                int new_id = preterminals.size();
                preterminals.emplace_back(pt);
                preterm_freq[new_id] = lm.preterm_freq[i];
            }
            else
            {
                preterm_freq[id] += lm.preterm_freq[i];
            }
        }

        // 合并 letter segment 统计
        for (int i = 0; i < (int)lm.letters.size(); i += 1)
        {
            segment src = lm.letters[i];

            int id = FindLetter(src);

            if (id == -1)
            {
                segment new_seg(src.type, src.length);
                int new_id = letters.size();

                letters.emplace_back(new_seg);
                letters_freq[new_id] = 0;
                id = new_id;
            }

            letters_freq[id] += lm.letters_freq[i];

            for (auto kv : src.values)
            {
                const string &value = kv.first;
                int src_value_id = kv.second;
                int freq = src.freqs.at(src_value_id);

                if (letters[id].values.find(value) == letters[id].values.end())
                {
                    int dst_value_id = letters[id].values.size();
                    letters[id].values[value] = dst_value_id;
                    letters[id].freqs[dst_value_id] = freq;
                }
                else
                {
                    int dst_value_id = letters[id].values[value];
                    letters[id].freqs[dst_value_id] += freq;
                }
            }
        }

        // 合并 digit segment 统计
        for (int i = 0; i < (int)lm.digits.size(); i += 1)
        {
            segment src = lm.digits[i];

            int id = FindDigit(src);

            if (id == -1)
            {
                segment new_seg(src.type, src.length);
                int new_id = digits.size();

                digits.emplace_back(new_seg);
                digits_freq[new_id] = 0;
                id = new_id;
            }

            digits_freq[id] += lm.digits_freq[i];

            for (auto kv : src.values)
            {
                const string &value = kv.first;
                int src_value_id = kv.second;
                int freq = src.freqs.at(src_value_id);

                if (digits[id].values.find(value) == digits[id].values.end())
                {
                    int dst_value_id = digits[id].values.size();
                    digits[id].values[value] = dst_value_id;
                    digits[id].freqs[dst_value_id] = freq;
                }
                else
                {
                    int dst_value_id = digits[id].values[value];
                    digits[id].freqs[dst_value_id] += freq;
                }
            }
        }

        // 合并 symbol segment 统计
        for (int i = 0; i < (int)lm.symbols.size(); i += 1)
        {
            segment src = lm.symbols[i];

            int id = FindSymbol(src);

            if (id == -1)
            {
                segment new_seg(src.type, src.length);
                int new_id = symbols.size();

                symbols.emplace_back(new_seg);
                symbols_freq[new_id] = 0;
                id = new_id;
            }

            symbols_freq[id] += lm.symbols_freq[i];

            for (auto kv : src.values)
            {
                const string &value = kv.first;
                int src_value_id = kv.second;
                int freq = src.freqs.at(src_value_id);

                if (symbols[id].values.find(value) == symbols[id].values.end())
                {
                    int dst_value_id = symbols[id].values.size();
                    symbols[id].values[value] = dst_value_id;
                    symbols[id].freqs[dst_value_id] = freq;
                }
                else
                {
                    int dst_value_id = symbols[id].values[value];
                    symbols[id].freqs[dst_value_id] += freq;
                }
            }
        }
    }

    cout << "Training phase 1 finished." << endl;
}

/// @brief 在模型中找到一个PT的统计数据
/// @param pt 需要查找的PT
/// @return 目标PT在模型中的对应下标
int model::FindPT(PT pt)
{
    for (int id = 0; id < preterminals.size(); id += 1)
    {
        if (preterminals[id].content.size() != pt.content.size())
        {
            continue;
        }
        else
        {
            bool equal_flag = true;
            for (int idx = 0; idx < preterminals[id].content.size(); idx += 1)
            {
                if (preterminals[id].content[idx].type != pt.content[idx].type || preterminals[id].content[idx].length != pt.content[idx].length)
                {
                    equal_flag = false;
                    break;
                }
            }
            if (equal_flag == true)
            {
                return id;
            }
        }
    }
    return -1;
}

/// @brief 在模型中找到一个letter segment的统计数据
/// @param seg 要找的letter segment
/// @return 目标letter segment的对应下标
int model::FindLetter(segment seg)
{
    for (int id = 0; id < letters.size(); id += 1)
    {
        if (letters[id].length == seg.length)
        {
            return id;
        }
    }
    return -1;
}

/// @brief 在模型中找到一个digit segment的统计数据
/// @param seg 要找的digit segment
/// @return 目标digit segment的对应下标
int model::FindDigit(segment seg)
{
    for (int id = 0; id < digits.size(); id += 1)
    {
        if (digits[id].length == seg.length)
        {
            return id;
        }
    }
    return -1;
}

int model::FindSymbol(segment seg)
{
    for (int id = 0; id < symbols.size(); id += 1)
    {
        if (symbols[id].length == seg.length)
        {
            return id;
        }
    }
    return -1;
}

void PT::insert(segment seg)
{
    content.emplace_back(seg);
}

void segment::insert(string value)
{
    if (values.find(value) == values.end())
    {
        values[value] = values.size();
        freqs[values[value]] = 1;
    }
    else
    {
        freqs[values[value]] += 1;
    }
}


void segment::order()
{
    for (pair<string, int> value : values)
    {
        ordered_values.emplace_back(value.first);
    }
    // cout << "value size:" << ordered_values.size() << endl;
    std::sort(ordered_values.begin(), ordered_values.end(),
              [this](const std::string &a, const std::string &b)
              {
                  return freqs.at(values[a]) > freqs.at(values[b]);
              });

    // 将排序后的频率存入 ordered_freqs 并计算 total_freq
    for (const std::string &val : ordered_values)
    {
        ordered_freqs.emplace_back(freqs.at(values[val]));
        total_freq += freqs.at(values[val]);
    }
    for (string val : ordered_values)
    {
        ordered_freqs.emplace_back(freqs.at(values[val]));
        total_freq += freqs.at(values[val]);
    }
}

void model::parse(string pw)
{
    PT pt;
    string curr_part = "";
    int curr_type = 0; // 0: 未设置, 1: 字母, 2: 数字, 3: 特殊字符
    // 请学会使用这种方式写for循环：for (auto it : iterable)
    // 相信我，以后你会用上的。You're welcome :)
    for (char ch : pw)
    {
        if (isalpha(ch))
        {
            if (curr_type != 1)
            {
                if (curr_type == 2)
                {
                    segment seg(curr_type, curr_part.length());
                    if (FindDigit(seg) == -1)
                    {
                        int id = GetNextDigitsID();
                        digits.emplace_back(seg);
                        digits[id].insert(curr_part);
                        digits_freq[id] = 1;
                    }
                    else
                    {
                        int id = FindDigit(seg);
                        digits_freq[id] += 1;
                        digits[id].insert(curr_part);
                    }
                    curr_part.clear();
                    pt.insert(seg);
                }
                else if (curr_type == 3)
                {
                    segment seg(curr_type, curr_part.length());
                    if (FindSymbol(seg) == -1)
                    {
                        int id = GetNextSymbolsID();
                        symbols.emplace_back(seg);
                        symbols_freq[id] = 1;
                        symbols[id].insert(curr_part);
                    }
                    else
                    {
                        int id = FindSymbol(seg);
                        symbols_freq[id] += 1;
                        symbols[id].insert(curr_part);
                    }
                    curr_part.clear();
                    pt.insert(seg);
                }
            }
            curr_type = 1;
            curr_part += ch;
        }
        else if (isdigit(ch))
        {
            if (curr_type != 2)
            {
                if (curr_type == 1)
                {
                    segment seg(curr_type, curr_part.length());
                    if (FindLetter(seg) == -1)
                    {
                        int id = GetNextLettersID();
                        letters.emplace_back(seg);
                        letters_freq[id] = 1;
                        letters[id].insert(curr_part);
                    }
                    else
                    {
                        int id = FindLetter(seg);
                        letters_freq[id] += 1;
                        letters[id].insert(curr_part);
                    }
                    curr_part.clear();
                    pt.insert(seg);
                }
                else if (curr_type == 3)
                {
                    segment seg(curr_type, curr_part.length());
                    if (FindSymbol(seg) == -1)
                    {
                        int id = GetNextSymbolsID();
                        symbols.emplace_back(seg);
                        symbols_freq[id] = 1;
                        symbols[id].insert(curr_part);
                    }
                    else
                    {
                        int id = FindSymbol(seg);
                        symbols_freq[id] += 1;
                        symbols[id].insert(curr_part);
                    }
                    curr_part.clear();
                    pt.insert(seg);
                }
            }
            curr_type = 2;
            curr_part += ch;
        }
        else
        {
            if (curr_type != 3)
            {
                if (curr_type == 1)
                {
                    segment seg(curr_type, curr_part.length());
                    if (FindLetter(seg) == -1)
                    {
                        int id = GetNextLettersID();
                        letters.emplace_back(seg);
                        letters_freq[id] = 1;
                        letters[id].insert(curr_part);
                    }
                    else
                    {
                        int id = FindLetter(seg);
                        letters_freq[id] += 1;
                        letters[id].insert(curr_part);
                    }
                    curr_part.clear();
                    pt.insert(seg);
                }
                else if (curr_type == 2)
                {
                    segment seg(curr_type, curr_part.length());
                    if (FindDigit(seg) == -1)
                    {
                        int id = GetNextDigitsID();
                        digits.emplace_back(seg);
                        digits_freq[id] = 1;
                        digits[id].insert(curr_part);
                    }
                    else
                    {
                        int id = FindDigit(seg);
                        digits_freq[id] += 1;
                        digits[id].insert(curr_part);
                    }
                    curr_part.clear();
                    pt.insert(seg);
                }
            }
            curr_type = 3;
            curr_part += ch;
        }
    }
    if (!curr_part.empty())
    {
        if (curr_type == 1)
        {
            segment seg(curr_type, curr_part.length());
            if (FindLetter(seg) == -1)
            {
                int id = GetNextLettersID();
                letters.emplace_back(seg);
                letters_freq[id] = 1;
                letters[id].insert(curr_part);
            }
            else
            {
                int id = FindLetter(seg);
                letters_freq[id] += 1;
                letters[id].insert(curr_part);
            }
            curr_part.clear();
            pt.insert(seg);
        }
        else if (curr_type == 2)
        {
            segment seg(curr_type, curr_part.length());
            if (FindDigit(seg) == -1)
            {
                int id = GetNextDigitsID();
                digits.emplace_back(seg);
                digits_freq[id] = 1;
                digits[id].insert(curr_part);
            }
            else
            {
                int id = FindDigit(seg);
                digits_freq[id] += 1;
                digits[id].insert(curr_part);
            }
            curr_part.clear();
            pt.insert(seg);
        }
        else
        {
            segment seg(curr_type, curr_part.length());
            if (FindSymbol(seg) == -1)
            {
                int id = GetNextSymbolsID();
                symbols.emplace_back(seg);
                symbols_freq[id] = 1;
                symbols[id].insert(curr_part);
            }
            else
            {
                int id = FindSymbol(seg);
                symbols_freq[id] += 1;
                symbols[id].insert(curr_part);
            }
            curr_part.clear();
            pt.insert(seg);
        }
    }
    // pt.PrintPT();
    // cout<<endl;
    // cout << FindPT(pt) << endl;
    total_preterm += 1;
    if (FindPT(pt) == -1)
    {
        for (int i = 0; i < pt.content.size(); i += 1)
        {
            pt.curr_indices.emplace_back(0);
        }
        int id = GetNextPretermID();
        // cout << id << endl;
        preterminals.emplace_back(pt);
        preterm_freq[id] = 1;
    }
    else
    {
        int id = FindPT(pt);
        // cout << id << endl;
        preterm_freq[id] += 1;
    }
}

void segment::PrintSeg()
{
    if (type == 1)
    {
        cout << "L" << length;
    }
    if (type == 2)
    {
        cout << "D" << length;
    }
    if (type == 3)
    {
        cout << "S" << length;
    }
}

void segment::PrintValues()
{
    // order();
    for (string iter : ordered_values)
    {
        cout << iter << " freq:" << freqs[values[iter]] << endl;
    }
}

void PT::PrintPT()
{
    for (auto iter : content)
    {
        iter.PrintSeg();
    }
}

void model::print()
{
    cout << "preterminals:" << endl;
    for (int i = 0; i < preterminals.size(); i += 1)
    {
        preterminals[i].PrintPT();
        // cout << preterminals[i].curr_indices.size() << endl;
        cout << " freq:" << preterm_freq[i];
        cout << endl;
    }
    // order();
    for (auto iter : ordered_pts)
    {
        iter.PrintPT();
        cout << " freq:" << preterm_freq[FindPT(iter)];
        cout << endl;
    }
    cout << "segments:" << endl;
    for (int i = 0; i < letters.size(); i += 1)
    {
        letters[i].PrintSeg();
        // letters[i].PrintValues();
        cout << " freq:" << letters_freq[i];
        cout << endl;
    }
    for (int i = 0; i < digits.size(); i += 1)
    {
        digits[i].PrintSeg();
        // digits[i].PrintValues();
        cout << " freq:" << digits_freq[i];
        cout << endl;
    }
    for (int i = 0; i < symbols.size(); i += 1)
    {
        symbols[i].PrintSeg();
        // symbols[i].PrintValues();
        cout << " freq:" << symbols_freq[i];
        cout << endl;
    }
}

bool compareByPretermProb(const PT& a, const PT& b) {
    return a.preterm_prob > b.preterm_prob;  // 降序排序
}

void model::order()
{
    cout << "Training phase 2: Ordering segment values and PTs..." << endl;
    for (PT pt : preterminals)
    {
        pt.preterm_prob = float(preterm_freq[FindPT(pt)]) / total_preterm;
        ordered_pts.emplace_back(pt);
    }
    bool swapped;
    cout << "total pts" << ordered_pts.size() << endl;
    std::sort(ordered_pts.begin(), ordered_pts.end(), compareByPretermProb);
    cout << "Ordering letters" << endl;
    // cout << "total letters" << endl;
    for (int i = 0; i < letters.size(); i += 1)
    {
        // cout << i << endl;
        letters[i].order();
    }
    cout << "Ordering digits" << endl;
    // cout << "total letters" << endl;
    for (int i = 0; i < digits.size(); i += 1)
    {
        digits[i].order();
    }
    cout << "ordering symbols" << endl;
    // cout << "total letters" << endl;
    for (int i = 0; i < symbols.size(); i += 1)
    {
        symbols[i].order();
    }
}