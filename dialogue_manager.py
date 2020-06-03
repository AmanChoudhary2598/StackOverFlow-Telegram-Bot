import os
from sklearn.metrics.pairwise import pairwise_distances_argmin
from utils import *
from sklearn.metrics.pairwise import cosine_similarity
import torch.nn as nn
import torch

from chit-chat-bot import *
USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")

class ThreadRanker(object):
    def __init__(self, paths):
        self.word_embeddings, self.embeddings_dim = load_embeddings("starspace_embedding.tsv")
        self.thread_embeddings_folder = paths['THREAD_EMBEDDINGS_FOLDER']

    def __load_embeddings_by_tag(self, tag_name):
        embeddings_path = os.path.join(self.thread_embeddings_folder, tag_name + ".pkl")
        thread_ids, thread_embeddings = unpickle_file(embeddings_path)
        return thread_ids, thread_embeddings

    def get_best_thread(self, question, tag_name):
        thread_ids, thread_embeddings = self.__load_embeddings_by_tag(tag_name)

        question_vec = np.array(question_to_vec(question=question, embeddings=self.word_embeddings, dim=self.embeddings_dim))
        #print(question_vec.shape)
        #print(thread_embeddings.shape)
        question_vec = question_vec.reshape(-1,100)
        siml = pairwise_distances_argmin(question_vec,thread_embeddings)
        best_thread = siml[0]

        return thread_ids[best_thread]


class DialogueManager(object):
    def __init__(self, paths):
        print("Loading resources...")

        # Intent recognition:
        self.tfidf_vectorizer = unpickle_file(RESOURCE_PATH['TFIDF_VECTORIZER'])
        self.intent_recognizer = unpickle_file(RESOURCE_PATH['INTENT_RECOGNIZER'])

        self.ANSWER_TEMPLATE = 'I think its about %s\nThis thread might help you: https://stackoverflow.com/questions/%s'

        # Goal-oriented part:
        self.tag_classifier = unpickle_file(RESOURCE_PATH['TAG_CLASSIFIER'])
        self.thread_ranker = ThreadRanker(RESOURCE_PATH)
        self.create_chitchat_bot()

    def create_chitchat_bot(self):
        # Configure models
        model_name = 'cb_model'
        attn_model = 'dot'
        hidden_size = 500
        encoder_n_layers = 2
        decoder_n_layers = 2
        dropout = 0.1
        corpus_name = "cornell movie-dialogs corpus"

        loadFilename = 'ConversationalModel/data/save/cb_model/cornell movie-dialogs corpus/2-2_500/30000_checkpoint.tar'

        checkpoint = torch.load(loadFilename)
        encoder_sd = checkpoint['en']
        decoder_sd = checkpoint['de']
        embedding_sd = checkpoint['embedding']
        voc = Voc(corpus_name)
        voc.__dict__ = checkpoint['voc_dict']


        print('Building encoder and decoder ...')
        # Initialize word embeddings
        embedding = nn.Embedding(voc.num_words, hidden_size)
        embedding.load_state_dict(embedding_sd)

        # Initialize encoder & decoder models
        encoder = EncoderRNN(hidden_size, embedding, encoder_n_layers, dropout)
        decoder = LuongAttnDecoderRNN(attn_model, embedding, hidden_size, voc.num_words, decoder_n_layers, dropout)
        encoder.load_state_dict(encoder_sd)
        decoder.load_state_dict(decoder_sd)

        # Use appropriate device
        encoder = encoder.to(device)
        decoder = decoder.to(device)
        encoder.eval()
        decoder.eval()
        print('Models built and ready to go!')

        self.encoder = encoder
        self.decoder = decoder
        self.voc = voc
        self.searcher = GreedySearchDecoder(self.encoder, self.decoder)

    def evaluate(self,encoder, decoder, searcher, voc, sentence, max_length=10):

        ### Format input sentence as a batch
        # words -> indexes
        indexes_batch = [indexesFromSentence(voc, sentence)]
        # Create lengths tensor
        lengths = torch.tensor([len(indexes) for indexes in indexes_batch])
        # Transpose dimensions of batch to match models' expectations
        input_batch = torch.LongTensor(indexes_batch).transpose(0, 1)
        # Use appropriate device
        input_batch = input_batch.to(device)
        lengths = lengths.to(device)
        # Decode sentence with searcher
        tokens, scores = searcher(input_batch, lengths, max_length)
        # indexes -> words
        decoded_words = [voc.index2word[token.item()] for token in tokens]
        return decoded_words

    def generate_answer(self, question):

        prepared_question = text_prepare(question)
        features = self.tfidf_vectorizer.transform([prepared_question])
        intent = self.intent_recognizer.predict(features)[0]

        # Chit-chat part:
        if intent == 'dialogue':
            input_sentence = normalizeString(question)
            # Evaluate sentence
            output_words = self.evaluate(self.encoder, self.decoder, self.searcher, self.voc, input_sentence)
            # Format and print response sentence
            output_words[:] = [x for x in output_words if not (x == 'EOS' or x == 'PAD')]
            response =  ' '.join(output_words)
            return response

        else:
            tag = self.tag_classifier.predict(features)[0]

            thread_id = self.thread_ranker.get_best_thread(prepared_question,tag)

            return self.ANSWER_TEMPLATE % (tag, thread_id)
