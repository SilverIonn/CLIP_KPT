import json
import torch
import torch.nn as nn
from typing import *
import numpy as np


class KnowledgeableVerbalizer(nn.Module):
    def __init__(self,num_classes):
        super().__init__()
        self.num_classes = num_classes
        self._in_on_label_words_set = False
        
    @property
    def label_words(self,):
        r'''
        Label words means the words in the vocabulary projected by the labels.
        E.g. if we want to establish a projection in sentiment classification: positive :math:`\rightarrow` {`wonderful`, `good`},
        in this case, `wonderful` and `good` are label words.
        '''
        if not hasattr(self, "_label_words"):
            raise RuntimeError("label words haven't been set.")
        return self._label_words
    
    @label_words.setter
    def label_words(self, label_words):
        if label_words is None:
            return
        self._label_words = label_words
        
        if not self._in_on_label_words_set:
            self.safe_on_label_words_set()

        
    def safe_on_label_words_set(self,):
        self._in_on_label_words_set = True
        self.on_label_words_set()
        self._in_on_label_words_set = False
        
        
    def on_label_words_set(self):
        self.label_words = self.delete_common_words(self.label_words)
        
        
#     def delete_common_words(self, d):
#         word_count = {}
#         for d_perclass in d:
#             for w in d_perclass:
#                 if w not in word_count:
#                     word_count[w]=1
#                 else:
#                     word_count[w]+=1
#         for w in word_count:
#             if word_count[w]>=2:
#                 for d_perclass in d:
#                     if w in d_perclass[1:]:
#                         findidx = d_perclass[1:].index(w)
#                         d_perclass.pop(findidx+1)
#         return d
    
    def delete_common_words(self, d):
        from collections import Counter
        d_flat = [word for words in d for word in words]
        counts = Counter(d_flat)
        common_words = set([word for word, count in counts.items() if count > 1])
        # Remove duplicates
        d_unique = []
        for words in d:
            new_words = []
            for idx, word in enumerate(words):
                if idx and word in common_words:
                    continue
                new_words.append(word)
            d_unique.append(new_words)
        return d_unique
    
    def from_file(self,
                  path: str,
                  choice: Optional[int] = 0 ):
        r"""Load the predefined label words from verbalizer file.
        Currently support three types of file format:
        1. a .jsonl or .json file, in which is a single verbalizer
        in dict format.
        2. a .jsonal or .json file, in which is a list of verbalizers in dict format
        3.  a .txt or a .csv file, in which is the label words of a class are listed in line,
        separated by commas. Begin a new verbalizer by an empty line.
        This format is recommended when you don't know the name of each class.
        The details of verbalizer format can be seen in :ref:`How_to_write_a_verbalizer`.
        Args:
            path (:obj:`str`): The path of the local template file.
            choice (:obj:`int`): The choice of verbalizer in a file containing
                             multiple verbalizers.
        Returns:
            Template : `self` object
        """
        if path.endswith(".txt") or path.endswith(".csv"):
            with open(path, 'r') as f:
                lines = f.readlines()
                label_words_all = []
                label_words_single_group = []
                for line in lines:
                    line = line.strip().strip(" ")
                    if line == "":
                        if len(label_words_single_group)>0:
                            label_words_all.append(label_words_single_group)
                        label_words_single_group = []
                    else:
                        label_words_single_group.append(line)
                if len(label_words_single_group) > 0: # if no empty line in the last
                    label_words_all.append(label_words_single_group)
                if choice >= len(label_words_all):
                    raise RuntimeError("choice {} exceed the number of verbalizers {}"
                                .format(choice, len(label_words_all)))

                label_words = label_words_all[choice]
                label_words = [label_words_per_label.strip().split(",") \
                            for label_words_per_label in label_words]

        elif path.endswith(".jsonl") or path.endswith(".json"):
            with open(path, "r") as f:
                label_words_all = json.load(f)
                # if it is a file containing multiple verbalizers
                if isinstance(label_words_all, list):
                    if choice >= len(label_words_all):
                        raise RuntimeError("choice {} exceed the number of verbalizers {}"
                                .format(choice, len(label_words_all)))
                    label_words = label_words_all[choice]
                elif isinstance(label_words_all, dict):
                    label_words = label_words_all
                    if choice>0:
                        logger.warning("Choice of verbalizer is 1, but the file  \
                        only contains one verbalizer.")

        self.label_words = label_words
        if self.num_classes is not None:
            num_classes = len(self.label_words)
            assert num_classes==self.num_classes, 'number of classes in the verbalizer file\
                                            does not match the predefined num_classes.'
        return self