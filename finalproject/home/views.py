from django.shortcuts import render
from django.http import HttpResponse
from django.template import loader
from django.template import context
from .models import Input
from collections import Counter
import nltk
import simplejson
from nltk.corpus import stopwords
from django.shortcuts import render_to_response
from django.template import RequestContext
from django.views.decorators.csrf import csrf_protect
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from nltk.corpus import state_union
from nltk.tokenize import PunktSentenceTokenizer
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from nltk.tokenize import PunktSentenceTokenizer
from nltk.stem.porter import *
from nltk.parse import RecursiveDescentParser
from collections import defaultdict
from nltk.stem.snowball import SnowballStemmer
import numpy as np
from array import array

# Create your views here.
def home(request):
    input = Input.objects.all()
    template = loader.get_template('home/home.html')
    context = {
        'input': input,
    }
    return HttpResponse(template.render(context, request))


def summarizer(request):
    input = Input.objects.all()
    template = loader.get_template('home/summ1.html')
    context = {
        'input': input,
    }
    return HttpResponse(template.render(context, request))


def summarizer(request):
    input = Input.objects.all()
    template = loader.get_template('home/summ1.html')
    context = {
        'input': input,
    }
    return HttpResponse(template.render(context, request))

def summarizer1(request):
    input = Input.objects.all()
    template = loader.get_template('home/summ2.html')
    context = {
        'input': input,
    }
    return HttpResponse(template.render(context, request))



def result(request):

    file = request.FILES['input_file']
    text = file.read()
    text=str(text)
    punctuation = [".", ",", "-", "'", "]", "[", "{", "}", "?", "!", ";", "(", ")", ":"]
    sent_list = []
    stemmer = SnowballStemmer("english", ignore_stopwords=True)
    stop = set(stopwords.words('english'))
    summary = []
    array = []


    def tokenize_sentence(text):
        sent_list = []
        sent_list = sent_tokenize(text)
        return sent_list


    sent_list = tokenize_sentence(text)


    def numb_of_sent():
        count = len(sent_list)
        return count


    def numb_of_words(sentence):
        counter = 0
        for w in sentence:
            counter = counter + 1
        return counter


    count = numb_of_sent()


    def total_numb_of_words():
        i = 0
        counter = 0
        for w in range(count):
            all_words = nltk.word_tokenize(sent_list[i])
            for x in all_words:
                if x not in punctuation:
                    counter = counter + 1
        return counter


    wordcount = total_numb_of_words()
    scores = [[] for i in range(count)]

    sum_length = int(count / 3)
    word_score = [[] for i in range(count)]
    imp_words = [[] for i in range(count)]
    imp_words1 = []


    def init_wordscore():
        i = 0
        for w in range(count):
            word_score[i] = 0
            i = i + 1
        return


    init_wordscore()
    output = []
    p = ""
    i = 0
    wordscore = []

    all_sents = nltk.sent_tokenize(text)


    ##tagged=nltk.pos_tag(all_words)

    def remove_braces(summary):
        all_sent = nltk.sent_tokenize(summary)
        flt_words = []
        i = 0
        for w in all_sent:
            all_words = nltk.word_tokenize(all_sent[i])
            sentence = ""
            found = 0
            for y in all_sent[i]:
                if y == "(":
                    found = 1
                if found == 0:
                    sentence = sentence + y
                if y == ")":
                    found = 0
            all_sent[i] = sentence
            i = i + 1
        summary = ''.join(all_sent)
        return summary


    def remove_square_brackets(summary):
        all_sent = nltk.sent_tokenize(summary)
        flt_words = []
        i = 0
        for w in all_sent:
            all_words = nltk.word_tokenize(all_sent[i])
            sentence = ""
            found = 0
            for y in all_sent[i]:
                if y == "[":
                    found = 1
                if found == 0:
                    sentence = sentence + y
                if y == "]":
                    found = 0
            all_sent[i] = sentence
            i = i + 1
        summary = ''.join(all_sent)
        return summary


    def find_name():
        all_names = []
        for m in all_sents:
            sent_words = nltk.word_tokenize(m)
            tagged = nltk.pos_tag(sent_words)
            ##                print(tagged)
            namechunkvar = []
            finalchunk = []

            def make_title(n):
                len_tagged = len(tagged)
                for u in tagged[n:len_tagged]:
                    if (((u[1] == 'VBZ') | (u[1] == 'VBD') | (u[0] ==
                                                                  ',') | (u[1] == 'VBN') | (u[1] == 'VBP') | (u[1] ==
                                                                                                                  'VBG') | (
                                u[1] == 'VB')) & (u[0][0].isupper() ==
                                                      False)):
                        all_names.append(name)
                        topic = name
                        break;
                    else:
                        namechunkvar.append(u[0])
                        name = ' '.join(namechunkvar)

                return

            for i in tagged:
                y = tagged.index(i)
                make_title(y)
                break;
        return (all_names)


    repeating_chunks = []
    all_names = find_name()
    repeating_chunks = ([k for k, v in Counter(all_names).items() if v > 1])
    print(repeating_chunks)
    length = len(all_names)


    def topic(text1):
        all_words = nltk.word_tokenize(text1)
        flt_words = []
        counter = 0
        i = 0
        p = nltk.tag.pos_tag(all_words)
        for w in all_words:
            if w not in punctuation:
                if w not in stop:
                    w = w.lower()
                    if ((p[i][1] != 'DT') & (p[i][1] != 'PRP') & (p[i][1] != 'PRP$')):
                        SnowballStemmer("english").stem(w)
                        flt_words.append(w)
            i = i + 1
        flt_words = nltk.FreqDist(flt_words)
        main_topic1 = flt_words.most_common(count)
        i = 0
        for w in main_topic1:
            word_score[i] = main_topic1[i][1]
            imp_words[i] = main_topic1[i][0]
            print(imp_words[i])
            i = i + 1
        all_words = nltk.sent_tokenize(text1)
        i = 0
        j = 0
        p = nltk.tag.pos_tag(imp_words)
        all_sent = nltk.sent_tokenize(text)
        for w in imp_words:
            if ((p[i][1] == 'NN') | (p[i][1] == 'NNS')):
                word_score[i] = word_score[i] * 5
            if ((p[i][1] == 'NNP') | (p[i][1] == 'NNPS')):
                word_score[i] = word_score[i] * 10
            j = 0
            for x in all_sent:
                all_words = nltk.word_tokenize(all_sent[j])
                y = all_words[0]
                if w.upper() == y.upper():
                    word_score[i] = word_score[i] * 50
                    print("word is :" + w)
                    print(word_score[i])
                j = j + 1
            i = i + 1

        i = 0
        length = len(repeating_chunks)
        for y in imp_words:
            j = 0
            for x in range(length):
                all_words = nltk.word_tokenize(repeating_chunks[j])
                for z in all_words:
                    if y.upper() == z.upper():
                        word_score[i] = word_score[i] * 100
                        print(y)
                        print(word_score[i])
                        print(" ")
                j = j + 1
            i = i + 1

        sorted_list = [i[0] for i in sorted(enumerate(word_score), key=lambda x: x[1])]

        j = 0
        for x in range(length):
            all_words = nltk.word_tokenize(repeating_chunks[j])
            check = 0
            for z in all_words:
                print(z.upper())
                if imp_words[sorted_list[count - 1]].upper() == z.upper():
                    topic1 = ' '.join(all_words)
                    x = length + 5
                    check = 1
                    break;
            if check == 1:
                break;
            j = j + 1

        if length == 0:
            topic1 = all_names[0]
            print(all_names[0])
        else:
            if check == 0:
                topic1 = imp_words[sorted_list[count - 1]]

        return topic1


    text1 = remove_square_brackets(text)
    text1 = remove_braces(text)
    main_topic2 = topic(text)
    main_topic2 = str(main_topic2)


    def sentence_score(j):
        all_words = nltk.word_tokenize(sent_list[j])
        flt_words = []
        counter = 1
        multiplier = 1
        sent_score = 0
        pos_value = 0
        check = 0
        for w in all_words:
            # if w not in stop_words:
            if w not in punctuation:
                flt_words.append(w)
        p = nltk.tag.pos_tag(flt_words)
        i = 0
        for w in flt_words:
            if ((p[i][1] == 'NN') | (p[i][1] == 'NNPS') | (p[i][1] == 'NNS') | (p[i][1] == 'NNP')):
                pos_value = pos_value + 1
            if ((p[i][1] == 'JJ') | (p[i][1] == 'JJS') | (p[i][1] == 'JJR')):
                pos_value = pos_value + 0.5
            if ((p[i][1] == 'RB') | (p[i][1] == 'RBS') | (p[i][1] == 'RBR')):
                pos_value = pos_value + 0.25
            if w in stop:
                counter = counter + 1
            for y in imp_words:
                if w == y:
                    multiplier = multiplier + 0.1
                if check == 0:
                    w = w.lower()
                    if w == y:
                        multiplier = multiplier + 1
            check = check + 1
            i = i + 1
        sent_length = len(sent_list[j])
        sentscore = int((sent_length)-((j+1)%count))
        sentscore = (sentscore / (counter * 0.1))
        sentscore = sentscore + pos_value
        sentscore = sentscore * multiplier
        return sentscore


    i = 0
    j = 0
    for x in range(count):
        scores[i] = sentence_score(j)
        i = i + 1
        j = j + 1

    ind = np.argpartition(scores, -sum_length)[-sum_length:]
    ind.sort()
    main_topic2 = remove_braces(main_topic2)
    main_topic2 = remove_square_brackets(main_topic2)


    def summ_print(ind, x):
        summary = sent_list[ind[x]]
        return summary


    i = 0

    for x in range(sum_length):
        s = summ_print(ind, i)
        summary.append(s)
        i = i + 1

    summary = ''.join(summary)

    summary = remove_braces(summary)

    summary = remove_square_brackets(summary)
    output = summary
    scores = simplejson.dumps(scores)
    template = loader.get_template('home/sum_result.html')
    context = {
        'output': output,
        'count': sum_length,
        'scores': scores,
        'topic': main_topic2,
        'word_score': word_score
    }
    return HttpResponse(template.render(context, request))






def result1(request):
    text = request.POST['input_text']
    punctuation = [".", ",", "-","'","]","[","{","}","?","!",";","(",")",":"]
    sent_list = []
    stemmer = SnowballStemmer("english", ignore_stopwords=True)
    stop = set(stopwords.words('english'))
    summary = []
    array = []


    def tokenize_sentence(text):
        sent_list = []
        sent_list = sent_tokenize(text)
        return sent_list

    sent_list = tokenize_sentence(text)

    def numb_of_sent():
        count = len(sent_list)
        return count

    def numb_of_words(sentence):
        counter=0
        for w in sentence:
            counter=counter+1
        return counter

    count = numb_of_sent()

    def total_numb_of_words():
        i=0
        counter=0
        for w in range(count):
            all_words=nltk.word_tokenize(sent_list[i])
            for x in all_words:
                if x not in punctuation:
                    counter=counter+1
        return counter

    wordcount=total_numb_of_words()
    scores = [[] for i in range(count)]


    sum_length = int(count / 3)
    word_score = [[] for i in range(count)]
    imp_words = [[] for i in range(count)]
    imp_words1 = []

    def init_wordscore():
        i=0
        for w in range(count):
            word_score[i]=0
            i=i+1
        return
    init_wordscore()
    output = []
    p=""
    i=0
    wordscore=[]

    all_sents = nltk.sent_tokenize(text)

    ##tagged=nltk.pos_tag(all_words)

    def remove_braces(summary):
        all_sent=nltk.sent_tokenize(summary)
        flt_words=[]
        i=0
        for w in all_sent:
            all_words=nltk.word_tokenize(all_sent[i])
            sentence = ""
            found=0
            for y in all_sent[i]:
                if y == "(":
                    found=1
                if found == 0:
                    sentence=sentence+y
                if y == ")":
                    found=0
            all_sent[i]=sentence
            i=i+1
        summary=''.join(all_sent)
        return summary

    def remove_square_brackets(summary):
        all_sent=nltk.sent_tokenize(summary)
        flt_words=[]
        i=0
        for w in all_sent:
            all_words=nltk.word_tokenize(all_sent[i])
            sentence = ""
            found=0
            for y in all_sent[i]:
                if y == "[":
                    found=1
                if found == 0:
                    sentence=sentence+y
                if y == "]":
                    found=0
            all_sent[i]=sentence
            i=i+1
        summary=''.join(all_sent)
        return summary

    def find_name():
        all_names = []
        for m in all_sents:
            sent_words = nltk.word_tokenize(m)
            tagged = nltk.pos_tag(sent_words)
            ##                print(tagged)
            namechunkvar = []
            finalchunk = []

            def make_title(n):
                len_tagged = len(tagged)
                for u in tagged[n:len_tagged]:
                    if (((u[1] == 'VBZ') | (u[1] == 'VBD') | (u[0] ==
                                                                  ',') | (u[1] == 'VBN') | (u[1] == 'VBP') | (u[1] ==
                                                                                                                  'VBG') | (
                        u[1] == 'VB')) & (u[0][0].isupper() ==
                                              False)):
                        all_names.append(name)
                        topic = name
                        break;
                    else:
                        namechunkvar.append(u[0])
                        name = ' '.join(namechunkvar)

                return

            for i in tagged:
                y = tagged.index(i)
                make_title(y)
                break;
        return (all_names)


    repeating_chunks=[]
    all_names = find_name()
    repeating_chunks=([k for k,v in Counter(all_names).items() if v>1])
    print(repeating_chunks)
    length=len(all_names)

    def topic(text1):
        all_words = nltk.word_tokenize(text1)
        flt_words = []
        counter = 0
        i = 0
        p = nltk.tag.pos_tag(all_words)
        for w in all_words:
            if w not in punctuation:
                if w not in stop:
                    w = w.lower()
                    if ((p[i][1] != 'DT')&(p[i][1] != 'PRP')&(p[i][1] != 'PRP$')):
                        SnowballStemmer("english").stem(w)
                        flt_words.append(w)
            i = i + 1
        flt_words = nltk.FreqDist(flt_words)
        main_topic1 = flt_words.most_common(count)
        i = 0
        for w in main_topic1:
            word_score[i] = main_topic1[i][1]
            imp_words[i] = main_topic1[i][0]
            print(imp_words[i])
            i = i + 1
        all_words = nltk.sent_tokenize(text1)
        i = 0
        j = 0
        p = nltk.tag.pos_tag(imp_words)
        all_sent = nltk.sent_tokenize(text)
        for w in imp_words:
            if ((p[i][1] == 'NN') | (p[i][1] == 'NNS')):
                word_score[i] = word_score[i] * 5
            if ((p[i][1] == 'NNP') | (p[i][1] == 'NNPS')):
                word_score[i] = word_score[i] * 10
            j = 0
            for x in all_sent:
                all_words = nltk.word_tokenize(all_sent[j])
                y = all_words[0]
                if w.upper() == y.upper():
                    word_score[i] = word_score[i] * 50
                    print("word is :" + w)
                    print(word_score[i])
                j = j + 1
            i = i + 1

        i = 0
        length = len(repeating_chunks)
        for y in imp_words:
            j = 0
            for x in range(length):
                all_words = nltk.word_tokenize(repeating_chunks[j])
                for z in all_words:
                    if y.upper() == z.upper():
                        word_score[i] = word_score[i] * 100
                        print(y)
                        print(word_score[i])
                        print(" ")
                j = j + 1
            i = i + 1

        sorted_list = [i[0] for i in sorted(enumerate(word_score), key=lambda x: x[1])]

        j = 0
        for x in range(length):
            all_words = nltk.word_tokenize(repeating_chunks[j])
            check = 0
            for z in all_words:
                print(z.upper())
                if imp_words[sorted_list[count - 1]].upper() == z.upper():
                    topic1 = ' '.join(all_words)
                    x = length + 5
                    check = 1
                    break;
            if check == 1:
                break;
            j = j + 1

        if length == 0:
            topic1 = all_names[0]
            print(all_names[0])
        else:
            if check == 0:
                topic1 = imp_words[sorted_list[count - 1]]

        return topic1

    text1=remove_square_brackets(text)
    text1=remove_braces(text)
    main_topic2 = topic(text)
    main_topic2 = str(main_topic2)
    def sentence_score(j):
        all_words = nltk.word_tokenize(sent_list[j])
        flt_words = []
        counter=1
        multiplier = 1
        sent_score = 0
        pos_value=0
        check=0
        for w in all_words:
            # if w not in stop_words:
            if w not in punctuation:
                flt_words.append(w)
        p=nltk.tag.pos_tag(flt_words)
        i=0
        for w in flt_words:
            if ((p[i][1] == 'NN')|(p[i][1] == 'NNPS')|(p[i][1] == 'NNS')|(p[i][1] == 'NNP')):
                pos_value=pos_value+1
            if ((p[i][1] == 'JJ')|(p[i][1] == 'JJS')|(p[i][1] == 'JJR')):
                pos_value=pos_value+0.5
            if ((p[i][1] == 'RB')|(p[i][1] == 'RBS')|(p[i][1] == 'RBR')):
                pos_value=pos_value+0.25
            if w in stop:
              counter=counter+1
            for y in imp_words:
                if w == y:
                    multiplier = multiplier+0.1
                if check == 0:
                    w=w.lower()
                    if w == y:
                        multiplier = multiplier +1
            check = check +1
            i=i+1
        sent_length = len(sent_list[j])
        sentscore = int((sent_length)+count/(j+1))
        sentscore = (sentscore/(counter*0.1))
        sentscore=sentscore+pos_value
        sentscore=sentscore*multiplier
        return sentscore

    i = 0
    j = 0
    for x in range(count):
        scores[i] = sentence_score(j)
        i = i + 1
        j = j + 1

    ind = np.argpartition(scores, -sum_length)[-sum_length:]
    ind.sort()
    main_topic2=remove_braces(main_topic2)
    main_topic2=remove_square_brackets(main_topic2)

    def summ_print(ind,x):
        summary=sent_list[ind[x]]
        return summary

    i=0

    for x in range(sum_length):
        s=summ_print(ind,i)
        summary.append(s)
        i=i+1

    summary = ''.join(summary)

    topic2=""
    for w in main_topic2:
        if w not in punctuation:
            topic2=topic2+w

    summary = remove_braces(summary)

    summary=remove_square_brackets(summary)
    output = summary
    scores=simplejson.dumps(scores)
    word_score = simplejson.dumps(word_score)
    template = loader.get_template('home/sum_result.html')
    context = {
        'output': output,
        'count':sum_length,
        'scores':scores,
        'topic': topic2,
        'word_score':word_score
    }
    return HttpResponse(template.render(context, request))
