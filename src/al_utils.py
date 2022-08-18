"""
Use this file for the querying strategies
"""

import numpy as np
from scipy.stats import entropy
from sklearn import preprocessing
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from clf_utils import ContentTypeData

class ALUtil:
    """
    Active learning querying strategies
    """
    def __init__(self, top_k=100):
        self.top_k = top_k
        self.model = SentenceTransformer('paraphrase-distilroberta-base-v1')
        self.ct_data = ContentTypeData()

    def similarity_between_docs(self, doc1, doc2):
        v1 = np.reshape(doc1, (1, -1))
        v2 = np.reshape(doc2, (1, -1))
        return cosine_similarity(v1, v2)[0][0]

    def return_cent(self, cents, emb):
        sims = {idx:self.similarity_between_docs(emb, i) for idx, i in enumerate(cents)}
        mx, mx_id = 0, 0
        for k, v in sims.items():
            if v > mx:
                mx = v
                mx_id = k
        return mx_id

    def margin_sampling(self, preds):
        """
        Returns the top k messages in the list of messages based on uncertainity 
        :params: 
            preds: list of predictions (probabilities of belonging to each class, remember its for multi-label problem)
        :return:
            messages_to_label_indices: indices of indices of messages to label
            messages_left_indices: indices of messages not to label
        """
        mean_uncertainity = np.abs(preds - 0.5).mean(axis=1)
        mean_uncertainity_indices = list(np.argsort(mean_uncertainity))
        return mean_uncertainity_indices[:self.top_k], mean_uncertainity_indices[self.top_k:]

    def qbc_sampling(self, preds_by_classifier):
        """
        Returns the top k messages in the list of messages based on query by committee
        :param: 
            preds_by_classifier: list of list of predictions (1 or 0, depending on which of the classes sample belongs to; 
                                    again, its a multi-label problem)
                                    each list consists of predictions from a classifier (committee memeber)
        :return:
            messages_to_label_indices: indices of indices of messages to label
            messages_left_indices: indices of messages not to label
        """
        preds_by_sample = np.array(preds_by_classifier).swapaxes(0, 1)
        sample_entropy = [np.array([entropy(sample[:, idx1], base=2) for idx1, _ in enumerate(range(sample.shape[1]))]).mean() for sample in preds_by_sample]
        sample_entropy_sorted_indices = list(np.argsort(sample_entropy)[::-1])
        return sample_entropy_sorted_indices[:self.top_k], sample_entropy_sorted_indices[self.top_k:]

    def cluster_sampling(self, preds_by_classifier, messages, cluster_centers):
        """
        Returns the top k messages in the list of messages based on cluster-based active learning
        :param: 
            preds_by_classifier: list of predictions (probabilities of belonging to each class, remember its for multi-label problem)
        :return:
            messages_to_label_indices: indices of indices of messages to label
            messages_left_indices: indices of messages not to label

        process:
            1. cluster the messages
            2. for each cluster, find the top k messages
        """
        preds_by_sample = np.array(preds_by_classifier).swapaxes(0, 1)
        sample_entropy = [np.array([entropy(sample[:, idx1], base=2) for idx1, _ in enumerate(range(sample.shape[1]))]).mean() for sample in preds_by_sample]
        sample_entropy_sorted_indices = list(np.argsort(sample_entropy)[::-1])
        # print(sample_entropy_sorted_indices)
        abs_messages = self.ct_data.abstract_sents(messages)
        encoded_messages = self.model.encode(abs_messages)
        encoded_messages = preprocessing.normalize(encoded_messages)
        messages_cluster_labels = [self.return_cent(cluster_centers, i) for i in encoded_messages]
        # print(messages_cluster_labels)
        messages_in_clusters = {idx: [] for idx, _ in enumerate(range(len(cluster_centers)))}
        for idx, label in enumerate(messages_cluster_labels):
            messages_in_clusters[label].append(sample_entropy_sorted_indices[idx])
        # print(messages_in_clusters)
        messages_to_label_indices, messages_left_indices = [], []
        for idx, cluster in messages_in_clusters.items():
            if len(cluster) > 0:
                messages_to_label_indices.extend(cluster[:1])
                messages_left_indices.extend(cluster[1:])
        return messages_to_label_indices, messages_left_indices
