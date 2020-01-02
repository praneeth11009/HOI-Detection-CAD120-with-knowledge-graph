import tensorflow as tf
import numpy as np
import ipdb

def masked_softmax_cross_entropy(preds, labels, mask):
    """Softmax cross-entropy loss with masking."""
    loss = tf.nn.softmax_cross_entropy_with_logits(logits=preds, labels=labels)
    mask = tf.cast(mask, dtype=tf.float32)
    mask /= tf.reduce_mean(mask)
    loss *= mask
    return tf.reduce_mean(loss)

def masked_accuracy(preds, labels, mask):
    """Accuracy with masking."""
    mask = tf.cast(mask, dtype=tf.float32)
    mask /= tf.reduce_mean(mask)

    labels *= mask 
    preds *= mask
    
    correct_prediction = tf.equal(tf.argmax(preds, 1), tf.argmax(labels, 1))
    accuracy_all = tf.cast(correct_prediction, tf.float32)
    return tf.reduce_mean(accuracy_all)

def masked_sigmoid_cross_entropy(preds, labels, mask):
    """Sigmoid cross-entropy loss with masking."""
    loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=preds, labels=labels)
    mask = tf.cast(mask, dtype=tf.float32)
    mask /= tf.reduce_mean(mask)
    loss *= mask
    return tf.reduce_mean(loss)

def mask_mse_loss(preds, labels, mask):
    """MSE loss with masking."""
    mask = tf.cast(mask, dtype=tf.float32)
    mask /= tf.reduce_mean(mask)

    labels *= mask 
    preds *= mask

    loss = tf.nn.l2_loss(tf.subtract(labels, preds))
    
    return loss

def sigmoid_cross_entropy(preds, labels): 
    loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=preds, labels=labels) 
    return tf.reduce_mean(loss)

def mse_loss(preds, labels):
    loss = tf.nn.l2_loss(tf.subtract(labels, preds))
    return loss

def softmax_cross_entropy(preds, labels):
    loss = tf.nn.softmax_cross_entropy_with_logits(logits=preds, labels=labels)
    return tf.reduce_mean(loss)

def accuracy(preds, labels):
    correct_prediction = tf.equal(tf.argmax(preds, 1), tf.argmax(labels, 1))
    accuracy_all = tf.cast(correct_prediction, tf.float32)
    return tf.reduce_mean(accuracy_all)

def CosineEmbeddingLoss(margin=0.1):
    def _cosine_similarity(x1, x2):
        #return tf.reduce_sum(x1 * x2, axis=-1) / (tf.norm(x1, axis=-1) * tf.norm(x2, axis=-1) + 1e-12) 
        return tf.reduce_sum(x1 * x2, axis=-1) # x1, x2 is already L2-norm
    def _cosine_embedding_loss_fn(input_one, input_two, target):
        similarity = _cosine_similarity(input_one, input_two)
        return tf.reduce_mean(tf.where(tf.equal(target, 1), 1. - similarity, tf.maximum(tf.zeros_like(similarity), similarity - margin))) 

def cosine_similarity(x1, x2):
    #return tf.reduce_sum(x1 * x2, axis=-1) / (tf.norm(x1, axis=-1) * tf.norm(x2, axis=-1) + 1e-12)
    return tf.reduce_sum(x1 * x2, axis=-1) # x1, x2 is already L2-norm

def sampling_neg(target, P, rows, cols):
    mask = np.zeros((rows, cols))
    cnt = 0
    for i in range(rows*cols):
        row = np.random.choice(range(rows))
        col  = np.random.choice(range(cols))
        if target[row, col]==0:
            mask[row, col] = 1
            cnt +=1
        if cnt>=P: break

def contrastive_loss(margin, x1, x2, target):
    sim_all = tf.reduce_sum(x1 * x2, axis=-1) 
    
    idx  = tf.equal(target, 1)
    mask_pos = tf.to_float(idx)
    sim_pos = tf.multiply(sim_all, mask_pos)
    match_loss = tf.reduce_sum(1. - sim_pos) / tf.reduce_sum(mask_pos)
    
    # negative sampling
    P = 16.0
    mask_neg = sampling_neg(target, P, 16, 26) 
    sim_neg = tf.multiply(sim_all, mask_neg)
    mismatch_loss = tf.reduce_sum(tf.maximum(tf.zeros_like(sim_neg), sim_neg - margin)) / P
                       
    return match_loss + mismatch_loss

def mse_loss(margin, x1, x2, target):
    sub = tf.reduce_sum(tf.subtract(x1, x2), axis=-1) 
    sim_all = tf.square(sub)/2
    
    idx  = tf.equal(target, 1)
    mask_pos = tf.to_float(idx)
    sim_pos = tf.multiply(sim_all, mask_pos)
    match_loss = tf.reduce_sum(tf.maximum(tf.zeros_like(sim_pos), sim_pos - margin)) / tf.reduce_sum(mask_pos) 
    
    # negative sampling
    P = 16.0
    mask_neg = sampling_neg(target, P, 16, 26) 
    sim_neg = tf.multiply(sim_all, mask_neg)
    mismatch_loss = tf.reduce_sum(1 - sim_neg) / P 
                       
    return match_loss + mismatch_loss

def cos_loss(x1, x2, target):
    idx = tf.equal(target, 1)
    mask = tf.to_float(idx)
    cos_sim = tf.multiply(tf.reduce_sum(x1 * x2, axis=-1), mask)
    return tf.reduce_sum(1. - cos_sim) / tf.reduce_sum(mask)

def mean_square_loss(x1, x2, target):
    idx = tf.where(tf.equal(target, 1))
    # import pdb
    # pdb.set_trace()
    # input1 = tf.gather(x1, idx)
    # input2 = tf.gather(x2, idx)
    input1 = tf.gather(x1, idx, axis=-1)
    input2 = tf.gather(x2, idx, axis=-1)
    l2_loss = tf.nn.l2_loss(tf.subtract(input1, input2)) # one number sum(x**2)/2
    return l2_loss
