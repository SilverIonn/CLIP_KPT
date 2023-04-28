from .scrap import get_related_words
from dassl.utils import mkdir_if_missing
import os.path as osp



# labels = [label, label, label ]
# search_words [[sublabel   ], search_inst  ]


# related score threshold 0; search words can be adjusted(list) for each label; 


def gen_labelwords(labels,search_words, cfg):
    label_words = []
    num = cfg.VERBALIZER.SIZE
    file = f'{cfg.VERBALIZER.DIR}/knowledgeable_verbalizer.txt'
    mkdir_if_missing(osp.dirname(file))
    
    for idx, label in enumerate(labels):
        
        search_inst = search_words[idx]
        
        if isinstance(search_inst, list):
            words_list = []
            num_sublabel = len(search_inst)
            for i, sub_label in enumerate(search_inst):
                subwords_dict_list = get_related_words(sub_label.lower())[:int(num/num_sublabel)]
                # subwords_dict_list = get_related_words(sub_label)[:int(num/num_sublabel)]
                # subwords_list = [i['word'] for i in subwords_dict_list if i['score'] > 0]
                subwords_list = [i['word'] for i in subwords_dict_list]
                subwords_list.insert(0,sub_label.lower())
                words_list.extend(subwords_list)
                
            words_list.insert(0,label.lower())  
                
            
        else:
        
            words_dict_list = get_related_words(search_inst.lower())[:num]
            # words_dict_list = get_related_words(search_inst)[:num]
            # words_dict_list = get_related_words(label)[:num]

            ###Single word
            # words_list = [i['word'] for i in words_dict_list if ' ' not in i['word'] and i['score'] > 0]
            ###Multiple
            # words_list = [i['word'] for i in words_dict_list if i['score'] > 0]
            words_list = [i['word'] for i in words_dict_list]
            # print(words_list)
            # print('\n')
            words_list.insert(0,label.lower())
        # print(words_list)
        # print('\n')
        label_words.append(words_list)
    # print(label_words)

    with open(file, 'w') as f:
        for line in label_words:
            for word in line[:-1]:
                f.write(f"{word},")
            f.write(f"{line[-1]}\n")




# def gen_labelwords(labels,cfg):
#     label_words = []
#     num = cfg.VERBALIZER.SIZE
#     file = f'{cfg.VERBALIZER.DIR}/knowledgeable_verbalizer.txt'
#     mkdir_if_missing(osp.dirname(file))
    
#     for idx, label in enumerate(labels):
        
#         words_dict_list = get_related_words(label.lower())[:num]
#         # words_dict_list = get_related_words(search_inst)[:num]
#         # words_dict_list = get_related_words(label)[:num]

#         ###Single word
#         # words_list = [i['word'] for i in words_dict_list if ' ' not in i['word'] and i['score'] > 0]
#         ###Multiple
#         # words_list = [i['word'] for i in words_dict_list if i['score'] > 0]
#         words_list = [i['word'] for i in words_dict_list]
#         # print(words_list)
#         # print('\n')
#         words_list.insert(0,label.lower())
#         # print(words_list)
#         # print('\n')
#         label_words.append(words_list)
#     # print(label_words)

#     with open(file, 'w') as f:
#         for line in label_words:
#             for word in line[:-1]:
#                 f.write(f"{word},")
#             f.write(f"{line[-1]}\n")