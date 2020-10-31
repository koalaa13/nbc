import os
import math
import matplotlib.pyplot

CLASS_CNT = 2
PARTS_CNT = 10


# 1 = legit message
def get_type(file_name):
    return int("legit" in file_name)


def calc_cnts(data):
    cnts = [{} for _ in range(CLASS_CNT)]
    for msg in data:
        for word in set(msg[1]):
            cnts[msg[0]].setdefault(word, 0)
            cnts[msg[0]][word] += 1
    return cnts


def calc_class_cnts(data):
    class_cnts = [0 for _ in range(CLASS_CNT)]
    for msg in data:
        class_cnts[msg[0]] += 1
    return class_cnts


def calc_all_word(data):
    all_words = set()
    for msg in data:
        all_words = all_words.union(set(msg[1]))
    return all_words


def calc_p(alpha, cnts, class_cnts, all_words):
    p = [{} for _ in range(CLASS_CNT)]
    for i in range(CLASS_CNT):
        for word in all_words:
            p[i].setdefault(word, 0)
            if word in cnts[i]:
                p[i][word] = (cnts[i][word] + alpha) / (class_cnts[i] + 2 * alpha)
            else:
                p[i][word] = alpha / (class_cnts[i] + 2 * alpha)
    return p


def get_ans(msg, lambda_legit, class_cnts, all_words, p, n):
    cur_ps = []
    cur_words = set(msg[1])
    for cls in range(CLASS_CNT):
        res = math.log(class_cnts[cls] / n)
        for word in all_words:
            if word in cur_words:
                res += math.log(p[cls][word])
            else:
                res += math.log(1 - p[cls][word])
        if cls == 1:
            res += math.log(lambda_legit)
        cur_ps.append(res)
    return max(range(len(cur_ps)), key=lambda i: cur_ps[i])


def get_ans_with_w(msg, lambda_legit, class_cnts, all_words, p, n):
    cur_ps = []
    cur_words = set(msg[1])
    for cls in range(CLASS_CNT):
        res = math.log(class_cnts[cls] / n)
        for word in all_words:
            if word in cur_words:
                res += math.log(p[cls][word])
            else:
                res += math.log(1 - p[cls][word])
        if cls == 1:
            res += math.log(lambda_legit)
        cur_ps.append(res)
    return cur_ps[0] / (cur_ps[0] + cur_ps[1]) - 0.5


def gen_gram_for_message(n, msg_words):
    new_msg_word = []
    for i in range(len(msg_words) - n + 1):
        cur_word = ""
        for j in range(i, i + n):
            cur_word += msg_words[j]
        new_msg_word.append(cur_word)
    return new_msg_word


def gen_grams(n, part):
    new_part = []
    for msg in part:
        new_part.append([msg[0], gen_gram_for_message(n, msg[1])])
    return new_part


def calc_accuracy(lambda_legit, alpha, parts):
    legit_as_spam = 0
    ac = 0
    count = 0
    for i in range(PARTS_CNT):
        learning_part = []
        for j in range(PARTS_CNT):
            if i == j:
                continue
            learning_part += parts[j]
        n = len(learning_part)
        cnts = calc_cnts(learning_part)
        class_cnts = calc_class_cnts(learning_part)
        all_words = calc_all_word(learning_part)
        p = calc_p(alpha, cnts, class_cnts, all_words)
        id = 0
        for msg in parts[i]:
            ans = get_ans(msg, lambda_legit, class_cnts, all_words, p, n)
            count += 1
            if ans == msg[0]:
                ac += 1
            elif msg[0] == 1:
                legit_as_spam += 1
            id += 1
        print("finished iteration#" + str(i) + " of cross-validation")
    return [ac / count, legit_as_spam]


def build_fs(lambda_legit, alpha, parts):
    fs = []
    m_minus = 0
    m_plus = 0
    for i in range(PARTS_CNT):
        learning_part = []
        for j in range(PARTS_CNT):
            if i == j:
                continue
            learning_part += parts[j]
        n = len(learning_part)
        cnts = calc_cnts(learning_part)
        class_cnts = calc_class_cnts(learning_part)
        all_words = calc_all_word(learning_part)
        p = calc_p(alpha, cnts, class_cnts, all_words)
        id = 0
        for msg in parts[i]:
            ans = get_ans_with_w(msg, lambda_legit, class_cnts, all_words, p, n)
            fs.append([ans, msg[0]])
            if msg[0] == 0:
                m_minus += 1
            else:
                m_plus += 1
            id += 1
        print("finished iteration#" + str(i) + " of cross-validation")
    fs.sort()
    with open("fs.txt", 'w') as f:
        f.write(str(m_minus) + '\n')
        f.write(str(m_plus) + '\n')
        f.write(str(fs))


def calc_ROC(file_name):
    with open(file_name, 'r') as f:
        m_minus = int(f.readline())
        m_plus = int(f.readline())
        data = f.readline().split(sep=', ')
        fs = []
        for i in reversed(range(len(data))):
            if i % 2 == 1:
                fs.append(int(data[i]))
        x = 0
        y = 0
        for v in fs:
            if v == 0:
                cur_x = x + 1. / m_minus
                cur_y = y
            else:
                cur_x = x
                cur_y = y + 1. / m_plus
            matplotlib.pyplot.plot([x, cur_x], [y, cur_y], "k-")
            x = cur_x
            y = cur_y
        matplotlib.pyplot.savefig("ROC.png")
        matplotlib.pyplot.clf()


parts = []
for i in range(PARTS_CNT):
    cur_part = []
    folder_name = "part" + str(i + 1)
    files = os.listdir(folder_name)
    for file in files:
        with open(folder_name + '/' + file) as f:
            title = f.readline().rstrip('\n').split(sep=' ')
            trash = f.readline()
            message = f.readline().rstrip('\n').split(sep=' ')
            cur_part.append([get_type(file), [*(str(x) + '$' for x in title[1:])] + [*(str(x) + '_' for x in message)]])
    parts.append(cur_part)

best_alpha = 1e-8
best_lambda = 1e200
# calc_ROC("fs.txt")
# parts_2_gramma = []
# for part in parts:
#     parts_2_gramma.append(gen_grams(2, part))
# parts_3_gramma = []
# for part in parts:
#     parts_3_gramma.append(gen_grams(3, part))
# build_fs(best_lambda, best_alpha, parts_3_gramma)
# lambdas = [1, 10, 100, 1e3, 1e6, 1e8, 1e15, 1e200]
# xs = [1, 10, 100, 1e3, 1e6, 1e8, 1e15, 1e200]
# ys = [0.955045871559633, 0.955045871559633, 0.955045871559633, 0.955045871559633,
#       0.9522935779816514, 0.9495412844036697, 0.9440366972477064, 0.8733944954128441]
# for i in range(len(xs)):
#     matplotlib.pyplot.plot(xs[i], ys[i], "go")
# for i in range(len(xs) - 1):
#     matplotlib.pyplot.plot([xs[i], xs[i + 1]], [ys[i], ys[i + 1]], "k-")
# matplotlib.pyplot.savefig("lambda_to_accuracy.png")
# matplotlib.pyplot.clf()
# alphas = [1, 1e-2, 1e-4, 1e-6, 1e-8]
# with open("1_gramma_info.txt", 'w') as f:
#     for alpha in alphas:
#         for lambda_legit in lambdas:
#             res = calc_accuracy(lambda_legit, alpha, parts)
#             f.write("alpha = " + str(alpha) + '\n')
#             f.write("lambda = " + str(lambda_legit) + '\n')
#             f.write("accuracy =  " + str(res[0]) + '\n')
#             f.write("legit as spam = " + str(res[1]) + '\n')
#             print("calculated alpha = " + str(alpha) + " lambda = " + str(lambda_legit))
# print("finished n = 1")
# with open("2_gramma_info.txt", 'w') as f:
#     for alpha in alphas:
#         for lambda_legit in lambdas:
#             res = calc_accuracy(lambda_legit, alpha, parts_2_gramma)
#             f.write("alpha = " + str(alpha) + '\n')
#             f.write("lambda = " + str(lambda_legit) + '\n')
#             f.write("accuracy =  " + str(res[0]) + '\n')
#             f.write("legit as spam = " + str(res[1]) + '\n')
#             print("calculated alpha = " + str(alpha) + " lambda = " + str(lambda_legit))
# print("finished n = 2")
# with open("3_gramma_info.txt", 'w') as f:
#     for alpha in alphas:
#         for lambda_legit in lambdas:
#             res = calc_accuracy(lambda_legit, alpha, parts_3_gramma)
#             f.write("alpha = " + str(alpha) + '\n')
#             f.write("lambda = " + str(lambda_legit) + '\n')
#             f.write("accuracy =  " + str(res[0]) + '\n')
#             f.write("legit as spam = " + str(res[1]) + '\n')
#             print("calculated alpha = " + str(alpha) + " lambda = " + str(lambda_legit))
# print("finished n = 3")
