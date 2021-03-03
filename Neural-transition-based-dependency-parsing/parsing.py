import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class PartialParse(object):
    """docstring forPartialParse."""

    def __init__(self, sentence):
        """sentence is list of strings"""
        super(PartialParse, self).__init__()
        self.sentence = sentence
        self.stack = ['ROOT']
        self.buffer = sentence.copy()
        self.dependencies = []

    def parse_step(self, transition):
        """transition takes values 'LA', 'RA', 'S' for left arc, right arc and shift
        respectively"""

        if transition == 'S':
            if len(self.buffer) != 0:
                self.stack.append(self.buffer[0])
                self.buffer.pop(0)

        if transition == 'LA':
            if len(self.stack) != 1 :
                self.dependencies.append((self.stack[-1], self.stack[-2]))
                self.stack.pop(-2)

        else:
            if len(self.stack) != 1:
                self.dependencies.append(self.stack[-2], self.stack[-1])
                self.stack,pop(-1)

    def parse(self, transitions):
        """apply the parse_step for all the given transitions"""

        for transition in transitions:
            self.parse_step(transition)

        return self.dependencies

def minibatch_parse(sentences, model, batch_size):
    partial_parses = [PartialParse(s) for s in sentences]
    unfinished_parses = partial_parses[:]
    while len(unfinished_parses) !=0:
        mini_batch = unfinished_parses[0:batch_size]
        while mini_batch:
            transitions = model.predict(mini_batch)
            for i, transition in enumerate(transitions):
                mini_batch[i].parse_step(transition)

            for temp in mini_batch:
                if len(temp.buffer) == 0 and len(temp.stack) == 1:
                    mini_batch.remove(temp)

        unfinished_parses = unfinished_parses[batch_size:]
    dependencies = [pp.dependencies for pp in partial_parses]

    return dependencies
