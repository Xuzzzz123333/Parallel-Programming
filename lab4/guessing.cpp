#include "PCFG.h"
#include <pthread.h>
#include <chrono>

using namespace std;

#ifndef NUM_THREADS
#define NUM_THREADS 4
#endif

#ifndef PTHREAD_THRESHOLD
#define PTHREAD_THRESHOLD 200000
#endif

double time_generate_only = 0.0;
double time_priority_only = 0.0;
void PriorityQueue::CalProb(PT &pt)
{
    // 计算PriorityQueue里面一个PT的流程如下：
    // 1. 首先需要计算一个PT本身的概率。例如，L6S1的概率为0.15
    // 2. 需要注意的是，Queue里面的PT不是“纯粹的”PT，而是除了最后一个segment以外，全部被value实例化的PT
    // 3. 所以，对于L6S1而言，其在Queue里面的实际PT可能是123456S1，其中“123456”为L6的一个具体value。
    // 4. 这个时候就需要计算123456在L6中出现的概率了。假设123456在所有L6 segment中的概率为0.1，那么123456S1的概率就是0.1*0.15

    // 计算一个PT本身的概率。后续所有具体segment value的概率，直接累乘在这个初始概率值上
    pt.prob = pt.preterm_prob;

    // index: 标注当前segment在PT中的位置
    int index = 0;


    for (int idx : pt.curr_indices)
    {
        // pt.content[index].PrintSeg();
        if (pt.content[index].type == 1)
        {
            // 下面这行代码的意义：
            // pt.content[index]：目前需要计算概率的segment
            // m.FindLetter(seg): 找到一个letter segment在模型中的对应下标
            // m.letters[m.FindLetter(seg)]：一个letter segment在模型中对应的所有统计数据
            // m.letters[m.FindLetter(seg)].ordered_values：一个letter segment在模型中，所有value的总数目
            pt.prob *= m.letters[m.FindLetter(pt.content[index])].ordered_freqs[idx];
            pt.prob /= m.letters[m.FindLetter(pt.content[index])].total_freq;
            // cout << m.letters[m.FindLetter(pt.content[index])].ordered_freqs[idx] << endl;
            // cout << m.letters[m.FindLetter(pt.content[index])].total_freq << endl;
        }
        if (pt.content[index].type == 2)
        {
            pt.prob *= m.digits[m.FindDigit(pt.content[index])].ordered_freqs[idx];
            pt.prob /= m.digits[m.FindDigit(pt.content[index])].total_freq;
            // cout << m.digits[m.FindDigit(pt.content[index])].ordered_freqs[idx] << endl;
            // cout << m.digits[m.FindDigit(pt.content[index])].total_freq << endl;
        }
        if (pt.content[index].type == 3)
        {
            pt.prob *= m.symbols[m.FindSymbol(pt.content[index])].ordered_freqs[idx];
            pt.prob /= m.symbols[m.FindSymbol(pt.content[index])].total_freq;
            // cout << m.symbols[m.FindSymbol(pt.content[index])].ordered_freqs[idx] << endl;
            // cout << m.symbols[m.FindSymbol(pt.content[index])].total_freq << endl;
        }
        index += 1;
    }
    // cout << pt.prob << endl;
}

void PriorityQueue::init()
{
    // cout << m.ordered_pts.size() << endl;
    // 用所有可能的PT，按概率填入优先队列
    for (PT pt : m.ordered_pts)
    {
        for (segment seg : pt.content)
        {
            if (seg.type == 1)
            {
                // max_indices 用来表示 PT 中各个 segment 的可能数目。
                // 例如 L6S1 中，假设模型统计到了 100 个 L6，
                // 那么 L6 对应的 max_indices 就是 100。
                pt.max_indices.emplace_back(
                    m.letters[m.FindLetter(seg)].ordered_values.size()
                );
            }

            if (seg.type == 2)
            {
                pt.max_indices.emplace_back(
                    m.digits[m.FindDigit(seg)].ordered_values.size()
                );
            }

            if (seg.type == 3)
            {
                pt.max_indices.emplace_back(
                    m.symbols[m.FindSymbol(seg)].ordered_values.size()
                );
            }
        }

        pt.preterm_prob = float(m.preterm_freq[m.FindPT(pt)]) / m.total_preterm;

        // 计算当前 PT 的概率
        CalProb(pt);

        // 将 PT 放入真正的堆式优先队列
        priority.push(pt);
    }

    // cout << "priority size:" << priority.size() << endl;
}

void PriorityQueue::PopNext()
{
    auto start_priority_1 = std::chrono::system_clock::now();

    PT curr_pt = priority.top();
    priority.pop();

    auto end_priority_1 = std::chrono::system_clock::now();

    auto start_generate = std::chrono::system_clock::now();

    Generate(curr_pt);

    auto end_generate = std::chrono::system_clock::now();

    auto start_priority_2 = std::chrono::system_clock::now();

    vector<PT> new_pts = curr_pt.NewPTs();

    for (PT pt : new_pts)
    {
        CalProb(pt);
        priority.push(pt);
    }

    auto end_priority_2 = std::chrono::system_clock::now();

    auto duration_generate = std::chrono::duration_cast<std::chrono::microseconds>(
        end_generate - start_generate
    );

    auto duration_priority_1 = std::chrono::duration_cast<std::chrono::microseconds>(
        end_priority_1 - start_priority_1
    );

    auto duration_priority_2 = std::chrono::duration_cast<std::chrono::microseconds>(
        end_priority_2 - start_priority_2
    );

    time_generate_only += double(duration_generate.count()) / 1000000.0;
    time_priority_only += double(duration_priority_1.count() + duration_priority_2.count()) / 1000000.0;
}
// 这个函数你就算看不懂，对并行算法的实现影响也不大
// 当然如果你想做一个基于多优先队列的并行算法，可能得稍微看一看了
vector<PT> PT::NewPTs()
{
    // 存储生成的新PT
    vector<PT> res;

    // 假如这个PT只有一个segment
    // 那么这个segment的所有value在出队前就已经被遍历完毕，并作为猜测输出
    // 因此，所有这个PT可能对应的口令猜测已经遍历完成，无需生成新的PT
    if (content.size() == 1)
    {
        return res;
    }
    else
    {
        // 最初的pivot值。我们将更改位置下标大于等于这个pivot值的segment的值（最后一个segment除外），并且一次只更改一个segment
        // 上面这句话里是不是有没看懂的地方？接着往下看你应该会更明白
        int init_pivot = pivot;

        // 开始遍历所有位置值大于等于init_pivot值的segment
        // 注意i < curr_indices.size() - 1，也就是除去了最后一个segment（这个segment的赋值预留给并行环节）
        for (int i = pivot; i < curr_indices.size() - 1; i += 1)
        {
            // curr_indices: 标记各segment目前的value在模型里对应的下标
            curr_indices[i] += 1;

            // max_indices：标记各segment在模型中一共有多少个value
            if (curr_indices[i] < max_indices[i])
            {
                // 更新pivot值
                pivot = i;
                res.emplace_back(*this);
            }

            // 这个步骤对于你理解pivot的作用、新PT生成的过程而言，至关重要
            curr_indices[i] -= 1;
        }
        pivot = init_pivot;
        return res;
    }

    return res;
}

struct GenerateRefThreadArg
{
    int start;
    int end;
    size_t base;
    const string *prefix;
    const vector<string> *values;
    vector<GuessRef> *guess_refs;
    bool has_prefix;
};

void *GenerateRefThreadFunc(void *arg)
{
    GenerateRefThreadArg *p = (GenerateRefThreadArg *)arg;
    vector<GuessRef> &refs = *(p->guess_refs);

    for (int i = p->start; i < p->end; i += 1)
    {
        GuessRef &ref = refs[p->base + i];

        ref.prefix = p->prefix;
        ref.values = p->values;
        ref.value_idx = i;
        ref.has_prefix = p->has_prefix;
    }

    pthread_exit(NULL);
}
// 这个函数是PCFG并行化算法的主要载体
// 尽量看懂，然后进行并行实现

void PriorityQueue::Generate(PT pt)
{
    CalProb(pt);

    // 情况一：PT 只有一个 segment
    if (pt.content.size() == 1)
    {
        segment *a = nullptr;

        if (pt.content[0].type == 1)
        {
            a = &m.letters[m.FindLetter(pt.content[0])];
        }
        else if (pt.content[0].type == 2)
        {
            a = &m.digits[m.FindDigit(pt.content[0])];
        }
        else if (pt.content[0].type == 3)
        {
            a = &m.symbols[m.FindSymbol(pt.content[0])];
        }

        int n = pt.max_indices[0];
        const vector<string> *values = &(a->ordered_values);

        size_t base = guess_refs.size();
        guess_refs.resize(base + n);

        if (n < PTHREAD_THRESHOLD)
        {
            for (int i = 0; i < n; i += 1)
            {
                GuessRef &ref = guess_refs[base + i];

                ref.prefix = nullptr;
                ref.values = values;
                ref.value_idx = i;
                ref.has_prefix = false;
            }
        }
        else
        {
            int thread_count = NUM_THREADS;

            if (thread_count > n)
            {
                thread_count = n;
            }

            vector<pthread_t> handles(thread_count);
            vector<GenerateRefThreadArg> args(thread_count);

            for (int t = 0; t < thread_count; t += 1)
            {
                int start = n * t / thread_count;
                int end = n * (t + 1) / thread_count;

                args[t].start = start;
                args[t].end = end;
                args[t].base = base;
                args[t].prefix = nullptr;
                args[t].values = values;
                args[t].guess_refs = &guess_refs;
                args[t].has_prefix = false;

                pthread_create(&handles[t], NULL, GenerateRefThreadFunc, &args[t]);
            }

            for (int t = 0; t < thread_count; t += 1)
            {
                pthread_join(handles[t], NULL);
            }
        }

        total_guesses += n;
    }

    // 情况二：PT 有多个 segment
    else
    {
        string prefix;
        int seg_idx = 0;

        // 构造最后一个 segment 之前的固定 prefix
        for (int idx : pt.curr_indices)
        {
            if (pt.content[seg_idx].type == 1)
            {
                prefix += m.letters[m.FindLetter(pt.content[seg_idx])].ordered_values[idx];
            }
            else if (pt.content[seg_idx].type == 2)
            {
                prefix += m.digits[m.FindDigit(pt.content[seg_idx])].ordered_values[idx];
            }
            else if (pt.content[seg_idx].type == 3)
            {
                prefix += m.symbols[m.FindSymbol(pt.content[seg_idx])].ordered_values[idx];
            }

            seg_idx += 1;

            if (seg_idx == (int)pt.content.size() - 1)
            {
                break;
            }
        }

        int last_idx = (int)pt.content.size() - 1;
        segment *a = nullptr;

        if (pt.content[last_idx].type == 1)
        {
            a = &m.letters[m.FindLetter(pt.content[last_idx])];
        }
        else if (pt.content[last_idx].type == 2)
        {
            a = &m.digits[m.FindDigit(pt.content[last_idx])];
        }
        else if (pt.content[last_idx].type == 3)
        {
            a = &m.symbols[m.FindSymbol(pt.content[last_idx])];
        }

        int n = pt.max_indices[last_idx];
        const vector<string> *values = &(a->ordered_values);

        // prefix 是局部变量，不能保存它的地址；
        // 放入 prefix_storage 后，再保存 prefix_storage 中字符串的地址。
        prefix_storage.emplace_back(std::move(prefix));
        const string *prefix_ptr = &prefix_storage.back();

        size_t base = guess_refs.size();
        guess_refs.resize(base + n);

        if (n < PTHREAD_THRESHOLD)
        {
            for (int i = 0; i < n; i += 1)
            {
                GuessRef &ref = guess_refs[base + i];

                ref.prefix = prefix_ptr;
                ref.values = values;
                ref.value_idx = i;
                ref.has_prefix = true;
            }
        }
        else
        {
            int thread_count = NUM_THREADS;

            if (thread_count > n)
            {
                thread_count = n;
            }

            vector<pthread_t> handles(thread_count);
            vector<GenerateRefThreadArg> args(thread_count);

            for (int t = 0; t < thread_count; t += 1)
            {
                int start = n * t / thread_count;
                int end = n * (t + 1) / thread_count;

                args[t].start = start;
                args[t].end = end;
                args[t].base = base;
                args[t].prefix = prefix_ptr;
                args[t].values = values;
                args[t].guess_refs = &guess_refs;
                args[t].has_prefix = true;

                pthread_create(&handles[t], NULL, GenerateRefThreadFunc, &args[t]);
            }

            for (int t = 0; t < thread_count; t += 1)
            {
                pthread_join(handles[t], NULL);
            }
        }

        total_guesses += n;
    }
}