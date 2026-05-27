# Copyright (c) Microsoft Corporation. 
# Licensed under the MIT license.
import logging
import sys
import json
import numpy as np

def read_answers(filename):
    answers = {}
    with open(filename) as f:
        for line in f:
            line = line.strip()
            js = json.loads(line)
            idx = js['idx']
            if idx not in answers:
                answers[idx] = []
            answers[idx].append(js['target'])
    return answers


def read_predictions(filename):
    predictions = {}
    with open(filename) as f:
        for line in f:
            line = line.strip()
            idx, label = line.split()
            idx = int(idx)
            if idx not in predictions:
                predictions[idx] = []
            predictions[idx].append(int(label))
    return predictions


def calculate_scores(answers, predictions):
    # Per-sample accuracy
    Acc = []
    for key in answers:
        if key not in predictions:
            logging.error("Missing prediction for index {}.".format(key))
            sys.exit()
        if len(answers[key]) != len(predictions[key]):
            logging.error("Mismatched number of samples for index {}: answers={}, predictions={}".format(
                key, len(answers[key]), len(predictions[key])))
            sys.exit()
        for a, p in zip(answers[key], predictions[key]):
            Acc.append(a == p)

    scores = {}
    scores['Acc'] = np.mean(Acc)

    # Pair accuracy: for idx values that appear exactly twice (a fixed + CVE pair),
    # a pair is correct only if BOTH samples are predicted correctly.
    pair_correct = 0
    pair_total = 0
    for key in answers:
        if len(answers[key]) == 2:
            pair_total += 1
            if all(answers[key][i] == predictions[key][i] for i in range(2)):
                pair_correct += 1

    if pair_total > 0:
        scores['PairAcc'] = pair_correct / pair_total

    return scores

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Evaluate leaderboard predictions for Defect Detection dataset.')
    parser.add_argument('--answers', '-a',help="filename of the labels, in txt format.")
    parser.add_argument('--predictions', '-p',help="filename of the leaderboard predictions, in txt format.")
    

    args = parser.parse_args()
    answers=read_answers(args.answers)
    predictions=read_predictions(args.predictions)
    scores=calculate_scores(answers,predictions)
    print(scores)

if __name__ == '__main__':
    main()
