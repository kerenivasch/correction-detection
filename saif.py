
import os
import math
import wave
import pickle
import random
import librosa
import soundfile
import webrtcvad
import contextlib
import numpy as np
import pandas as pd
import tensorflow as tf
import skipthoughts.skipthoughts as st

vad = webrtcvad.Vad()
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

kfold = 5
test_frac = 1/kfold
categories = 3
voice_feat = 600
transcript_feat = 9600
minibatch_size = 128
graph_seed = 1234
graph_keep_prob = 0.5


def get_activations(clf, X):

    hidden_layer_sizes = clf.hidden_layer_sizes
    if not hasattr(hidden_layer_sizes, "__iter__"):
        hidden_layer_sizes = [hidden_layer_sizes]
    hidden_layer_sizes = list(hidden_layer_sizes)
    layer_units = [np.shape(X)[1]] + hidden_layer_sizes +  [clf.n_outputs_]
    activations = [X]
    for i in range(clf.n_layers_ - 1):
        activations.append(np.empty((np.shape(X)[0], layer_units[i + 1])))
    clf._forward_pass(activations)
    return activations[-2]

def extract_feature(file_name, mfcc, chroma, mel):
    """From DataFlair Team Speech Emotion Recognition with librosa mini project"""
    with soundfile.SoundFile(file_name) as sound_file:

        X = sound_file.read(dtype="float32")
        sample_rate=sound_file.samplerate

        if chroma:
            stft=np.abs(librosa.stft(X))
        result=np.array([])
        if mfcc:
            mfccs=np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
            result=np.hstack((result, mfccs))
        if chroma:
            chroma=np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,axis=0)
            result=np.hstack((result, chroma))
        if mel:
            mel=np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T,axis=0)
            result=np.hstack((result, mel))

    return result

def get_emotions_vec(wav_name):
    return extract_feature("./wav/" + wav_name, True, True, True)

def count_frames_with_speech(wav_name):
    offset = 0
    speech_counter = 0
    frame_size = 30

    with contextlib.closing(wave.open("./wav/" + wav_name, 'rb')) as wf:
        num_channels = wf.getnchannels()
        assert num_channels == 1
        sample_width = wf.getsampwidth()
        assert sample_width == 2
        rate = wf.getframerate()
        assert rate in (8000, 16000, 32000, 48000)
        sig = wf.readframes(wf.getnframes())

    n = int(rate * (frame_size / 1000.0) * 2)
    while offset + n < len(sig):
        if vad.is_speech(sig[offset:offset + n], rate):
            speech_counter += 1

        offset += n

    return speech_counter

def load_data():
    ret_data = []

    mapping = pd.ExcelFile("transcriptVoiceMap.xlsx").parse(0)

    #load skipthoughts encoder
    model = st.load_model()
    encoder = st.Encoder(model)
    all_fst_transcript_vectors = encoder.encode([str(x) for x in mapping["fst transcript"]])
    all_snd_transcript_vectors = encoder.encode([str(x) for x in mapping["snd transcript"]])

    #load voice emotions model
    emotions_model = pickle.load(open("emotions_model.sav", 'rb'))
    all_fst_wav_vectors = get_activations(emotions_model, np.array([get_emotions_vec(x) for x in mapping["fst wav name"]]))
    all_snd_wav_vectors = get_activations(emotions_model, np.array([get_emotions_vec(x) for x in mapping["snd wav name"]]))

    for idx in range(len(mapping)):

        transcript_vec = all_fst_transcript_vectors[idx]
        snd_transcript_vec = all_snd_transcript_vectors[idx]
        transcript_vec = np.concatenate((np.abs(transcript_vec - snd_transcript_vec), transcript_vec * snd_transcript_vec))

        output = mapping["output"][idx]

        emotions_vec = all_fst_wav_vectors[idx]
        emotions_vec = np.concatenate((emotions_vec, all_snd_wav_vectors[idx]))

        fst_wav = mapping["fst wav name"][idx]
        snd_wav = mapping["snd wav name"][idx]
        frames_with_speech = count_frames_with_speech(fst_wav)
        frames_with_speech = count_frames_with_speech(snd_wav) - frames_with_speech

        label_idx = mapping["label"][idx]
        label = np.zeros(categories)
        label[label_idx] = 1

        ret_data.append((transcript_vec, [output], emotions_vec, [frames_with_speech], label))

    return ret_data

def split_data(k, all):
    train = []
    test = []
    split_point = int(len(all) * test_frac)

    for idx in range(0, len(all)):
        if idx in range(k * split_point, (k + 1) * split_point):
            test.append((all[idx]))
        else:
            train.append((all[idx]))

    return train, test

def print_test_error(conf_matrix):
    for category in range(categories):
        print("category: ", category)

        next = (category+1) % categories
        next_next = (next+1) % categories

        true_positive = conf_matrix[category][category]
        false_negative = conf_matrix[category][next] + conf_matrix[category][next_next]
        false_positive = conf_matrix[next][category] + conf_matrix[next_next][category]
        recall = true_positive / (true_positive + false_negative)
        print("recall: ", recall)
        precision = true_positive / (true_positive + false_positive)
        print("precision: ", precision)
        print("f1: ", (2 * precision * recall) / (precision + recall))


all = load_data()
random.seed(17)
random.shuffle(all)

tot_train_acc = 0
tot_test_acc = 0
conf_matrix = np.zeros([categories, categories], dtype=int)
for k in range(0, kfold):

    train, test = split_data(k, all)

    tf.reset_default_graph()

    x1 = tf.placeholder(dtype=tf.float32, shape=[None, transcript_feat])
    x2 = tf.placeholder(dtype=tf.float32, shape=[None, 1])
    x3 = tf.placeholder(dtype=tf.float32, shape=[None, voice_feat])
    x4 = tf.placeholder(dtype=tf.float32, shape=[None, 1])

    y_ = tf.placeholder(dtype=tf.float32, shape=[None, categories])

    #transcript + exe
    hid1_size = 30
    W1 = tf.Variable(tf.truncated_normal([transcript_feat+1, hid1_size], stddev=0.1, seed=graph_seed))
    b1 = tf.Variable(tf.constant(0.1, shape=[hid1_size]))
    z1 = tf.nn.relu(tf.matmul(tf.concat([x1, x2], 1), W1) + b1)

    #voice + vad
    hid2_size = 30
    W2 = tf.Variable(tf.truncated_normal([voice_feat+1, hid2_size], stddev=0.1, seed=graph_seed))
    b2 = tf.Variable(tf.constant(0.1, shape=[hid2_size]))
    z2 = tf.nn.relu(tf.matmul(tf.concat([x3, x4], 1), W2) + b2)

    #Third HL
    hid3_size = 30
    W3 = tf.Variable(tf.truncated_normal([hid1_size + hid2_size, hid3_size], stddev=0.1, seed=graph_seed))
    b3 = tf.Variable(tf.constant(0.1, shape=[hid3_size]))
    z3 = tf.nn.relu(tf.matmul(tf.concat([z1, z2], 1), W3) + b3)
    keep_prob = tf.placeholder(tf.float32)
    z3_drop = tf.nn.dropout(z3, keep_prob, seed=graph_seed)

    W = tf.Variable(tf.truncated_normal([hid3_size, categories], stddev=0.1, seed=graph_seed))
    b = tf.Variable(tf.constant(0.0,shape=[categories]))
    z = tf.matmul(z3_drop, W) + b

    y = tf.nn.softmax(z)
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=z, labels=y_))
    update = tf.train.AdamOptimizer(0.001).minimize(loss)
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    for i in range(10000):

        numMiniBatches = math.ceil(len(train) / minibatch_size)
        idx = (i % numMiniBatches) * minibatch_size
        lastIdx = idx + minibatch_size
        if (lastIdx > len(train)):
            lastIdx = len(train)

        xtranscript = [item[0] for item in train[idx:lastIdx]]
        xoutput = [item[1] for item in train[idx:lastIdx]]
        xvoice = [item[2] for item in train[idx:lastIdx]]
        xvad = [item[3] for item in train[idx:lastIdx]]

        ylabel = [item[4] for item in train[idx:lastIdx]]

        [_, acc, loss_val] = sess.run([update, accuracy, loss], feed_dict={x1:xtranscript, x2:xoutput, x3:xvoice,
                                                                    x4:xvad, y_: ylabel, keep_prob:graph_keep_prob})
        if acc > 0.995:
            break

    tot_train_acc += acc

    actuals = [item[4] for item in test]
    test_acc, predictions = sess.run([accuracy, y], feed_dict={x1: [item[0] for item in test],
                        x2: [item[1] for item in test], x3: [item[2] for item in test], x4: [item[3] for item in test],
                        y_: actuals, keep_prob:graph_keep_prob})

    print("fold ", k, ": accuracy on test ", test_acc)
    tot_test_acc += test_acc

    for p in range(len(predictions)):
        actual = np.argmax(actuals[p])
        prediction = np.argmax(predictions[p])
        conf_matrix[actual][prediction] += 1

    print("cumulative confusion matrix: ")
    print(conf_matrix)

print("final train accuracy: ", tot_train_acc/kfold)
print("final test accuracy: ", tot_test_acc/kfold)

print_test_error(conf_matrix)