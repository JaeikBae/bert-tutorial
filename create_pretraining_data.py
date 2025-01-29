# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Create masked LM/next sentence masked_lm TF examples for BERT."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import random
import tokenization
import tensorflow as tf
from absl import flags
FLAGS = flags.FLAGS 

flags.DEFINE_string("input_file", None,
                    "Input raw text file (or comma-separated list of files).")

flags.DEFINE_string(
    "output_file", None,
    "Output TF example file (or comma-separated list of files).")

flags.DEFINE_string("vocab_file", None,
                    "The vocabulary file that the BERT model was trained on.")

flags.DEFINE_bool(
    "do_lower_case", True,
    "Whether to lower case the input text. Should be True for uncased "
    "models and False for cased models.")

flags.DEFINE_bool(
    "do_whole_word_mask", False,
    "Whether to use whole word masking rather than per-WordPiece masking.")

flags.DEFINE_integer("max_seq_length", 128, "Maximum sequence length.")

flags.DEFINE_integer("max_predictions_per_seq", 20,
                     "Maximum number of masked LM predictions per sequence.")

flags.DEFINE_integer("random_seed", 12345, "Random seed for data generation.")

flags.DEFINE_integer(
    "dupe_factor", 10,
    "Number of times to duplicate the input data (with different masks).")

flags.DEFINE_float("masked_lm_prob", 0.15, "Masked LM probability.")

flags.DEFINE_float(
    "short_seq_prob", 0.1,
    "Probability of creating sequences which are shorter than the "
    "maximum length.")


class TrainingInstance(object):
  """A single training instance (sentence pair)."""

  def __init__(self, tokens, segment_ids, masked_lm_positions, masked_lm_labels,
               is_random_next):
    self.tokens = tokens
    self.segment_ids = segment_ids
    self.is_random_next = is_random_next
    self.masked_lm_positions = masked_lm_positions
    self.masked_lm_labels = masked_lm_labels

  def __str__(self):
    s = ""
    s += "tokens: %s\n" % (" ".join(
        [tokenization.printable_text(x) for x in self.tokens]))
    s += "segment_ids: %s\n" % (" ".join([str(x) for x in self.segment_ids]))
    s += "is_random_next: %s\n" % self.is_random_next
    s += "masked_lm_positions: %s\n" % (" ".join(
        [str(x) for x in self.masked_lm_positions]))
    s += "masked_lm_labels: %s\n" % (" ".join(
        [tokenization.printable_text(x) for x in self.masked_lm_labels]))
    s += "\n"
    return s

  def __repr__(self):
    return self.__str__()


def write_instance_to_example_files(instances, tokenizer, max_seq_length,
                                    max_predictions_per_seq, output_files):
  """Create TF example files from `TrainingInstance`s."""
  writers = []
  for output_file in output_files:
    writers.append(tf.python_io.TFRecordWriter(output_file))

  writer_index = 0

  total_written = 0
  for (inst_index, instance) in enumerate(instances):
    input_ids = tokenizer.convert_tokens_to_ids(instance.tokens)
    input_mask = [1] * len(input_ids)
    segment_ids = list(instance.segment_ids)
    assert len(input_ids) <= max_seq_length

    while len(input_ids) < max_seq_length:
      input_ids.append(0)
      input_mask.append(0)
      segment_ids.append(0)

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length

    masked_lm_positions = list(instance.masked_lm_positions)
    masked_lm_ids = tokenizer.convert_tokens_to_ids(instance.masked_lm_labels)
    masked_lm_weights = [1.0] * len(masked_lm_ids)

    while len(masked_lm_positions) < max_predictions_per_seq:
      masked_lm_positions.append(0)
      masked_lm_ids.append(0)
      masked_lm_weights.append(0.0)

    next_sentence_label = 1 if instance.is_random_next else 0

    features = collections.OrderedDict()
    features["input_ids"] = create_int_feature(input_ids)
    features["input_mask"] = create_int_feature(input_mask)
    features["segment_ids"] = create_int_feature(segment_ids)
    features["masked_lm_positions"] = create_int_feature(masked_lm_positions)
    features["masked_lm_ids"] = create_int_feature(masked_lm_ids)
    features["masked_lm_weights"] = create_float_feature(masked_lm_weights)
    features["next_sentence_labels"] = create_int_feature([next_sentence_label])

    tf_example = tf.train.Example(features=tf.train.Features(feature=features))

    writers[writer_index].write(tf_example.SerializeToString())
    writer_index = (writer_index + 1) % len(writers)

    total_written += 1

    if inst_index < 20:
      tf.logging.info("*** Example ***")
      tf.logging.info("tokens: %s" % " ".join(
          [tokenization.printable_text(x) for x in instance.tokens]))

      for feature_name in features.keys():
        feature = features[feature_name]
        values = []
        if feature.int64_list.value:
          values = feature.int64_list.value
        elif feature.float_list.value:
          values = feature.float_list.value
        tf.logging.info(
            "%s: %s" % (feature_name, " ".join([str(x) for x in values])))

  for writer in writers:
    writer.close()

  tf.logging.info("Wrote %d total instances", total_written)


def create_int_feature(values):
  feature = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
  return feature


def create_float_feature(values):
  feature = tf.train.Feature(float_list=tf.train.FloatList(value=list(values)))
  return feature


def create_training_instances(input_files, tokenizer, max_seq_length,
                              dupe_factor, short_seq_prob, masked_lm_prob,
                              max_predictions_per_seq, rng):
  """Create `TrainingInstance`s from raw text."""
  all_documents = [[]]

  # Input file format:
  # (1) One sentence per line. These should ideally be actual sentences, not
  # entire paragraphs or arbitrary spans of text. (Because we use the
  # sentence boundaries for the "next sentence prediction" task).
  # (2) Blank lines between documents. Document boundaries are needed so
  # that the "next sentence prediction" task doesn't span between documents.
  for input_file in input_files:
    with tf.io.gfile.GFile(input_file, "r") as reader:
      while True:
        line = tokenization.convert_to_unicode(reader.readline())
        if not line:
          break
        line = line.strip()

        # Empty lines are used as document delimiters
        if not line:
          all_documents.append([])
        tokens = tokenizer.tokenize(line)
        if tokens:
          all_documents[-1].append(tokens)

  # Remove empty documents
  all_documents = [x for x in all_documents if x]
  rng.shuffle(all_documents)

  vocab_words = list(tokenizer.vocab.keys())
  instances = []
  for _ in range(dupe_factor):
    for document_index in range(len(all_documents)): #전체 문서에서
      instances.extend( # 문서별로 학습 데이터 생성
          create_instances_from_document(
              all_documents, document_index, max_seq_length, short_seq_prob,
              masked_lm_prob, max_predictions_per_seq, vocab_words, rng))

  rng.shuffle(instances) # 문서 섞기
  return instances # 전처리 완료된 학습 데이터 반환


def create_instances_from_document(
    all_documents, document_index, max_seq_length, short_seq_prob,
    masked_lm_prob, max_predictions_per_seq, vocab_words, rng):
  """Creates `TrainingInstance`s for a single document."""
  document = all_documents[document_index] # 전체 문서에서 현재 문서를 가져옴

  # Account for [CLS], [SEP], [SEP]
  max_num_tokens = max_seq_length - 3 # 전체 토큰 수 제한할건데 특수 토큰 자리도 포함해서

  # We *usually* want to fill up the entire sequence since we are padding
  # to `max_seq_length` anyways, so short sequences are generally wasted
  # computation. However, we *sometimes*
  # (i.e., short_seq_prob == 0.1 == 10% of the time) want to use shorter
  # sequences to minimize the mismatch between pre-training and fine-tuning.
  # The `target_seq_length` is just a rough target however, whereas
  # `max_seq_length` is a hard limit.
  target_seq_length = max_num_tokens # 타겟 시퀀스 길이를 설정
  if rng.random() < short_seq_prob: # 대략 10%는 짧은 문장을 생성
    target_seq_length = rng.randint(2, max_num_tokens)

  # We DON'T just concatenate all of the tokens from a document into a long
  # sequence and choose an arbitrary split point because this would make the
  # next sentence prediction task too easy. Instead, we split the input into
  # segments "A" and "B" based on the actual "sentences" provided by the user
  # input.
  instances = []
  current_chunk = []
  current_length = 0
  i = 0 # 문서내 토큰 위치
  while i < len(document): # 문서를 순환하면서
    segment = document[i] # 문장 가져오기
    current_chunk.append(segment) # 현재 청크에 붙이기
    current_length += len(segment) # 길이 체크
    if i == len(document) - 1 or current_length >= target_seq_length: # 끝까지 도달했거나, 타겟 길이를 달성한 경우
      if current_chunk: # 현재 청크가 있으면. 어떤 이유로 청크가 없으면 패스
        # `a_end` is how many segments from `current_chunk` go into the `A`
        # (first) sentence.
        a_end = 1
        if len(current_chunk) >= 2: # 현재 청크가 문장 2개 이상으로 구성된 경우
          a_end = rng.randint(1, len(current_chunk) - 1) # 문장 A의 끝을 랜덤으로 정하기

        tokens_a = []
        for j in range(a_end): # a_end만큼 문장 A 나누기
          tokens_a.extend(current_chunk[j])

        tokens_b = [] 
        # Random next
        is_random_next = False # 50% 랜덤 가져오기 하지 않음
        if len(current_chunk) == 1 or rng.random() < 0.5: # 현재 청크 길이가 1이거나, 50% 확률로 랜덤 문장 가져오기
          is_random_next = True # 랜덤으로 가져왔다고 표시
          target_b_length = target_seq_length - len(tokens_a) # B 길이 계산

          # This should rarely go for more than one iteration for large
          # corpora. However, just to be careful, we try to make sure that
          # the random document is not the same as the document
          # we're processing.
          for _ in range(10): # 현재와 다른 문서 가져오기 10번 시도
            random_document_index = rng.randint(0, len(all_documents) - 1)
            if random_document_index != document_index:
              break

          random_document = all_documents[random_document_index] # B 문서 가져오기
          random_start = rng.randint(0, len(random_document) - 1) # 시작위치 지정
          for j in range(random_start, len(random_document)):
            tokens_b.extend(random_document[j]) # B 가져오기
            if len(tokens_b) >= target_b_length: # 길이만큼 가져오기
              break
          # We didn't actually use these segments so we "put them back" so
          # they don't go to waste.
          num_unused_segments = len(current_chunk) - a_end # 안쓴 길이 계산
          i -= num_unused_segments # 안쓴 만큼 뒤로 가기
        # Actual next
        else:
          is_random_next = False # B 문장을 그대로 쓰는경우
          for j in range(a_end, len(current_chunk)): # 현재 문서에서 문장 가져오기
            tokens_b.extend(current_chunk[j])
        truncate_seq_pair(tokens_a, tokens_b, max_num_tokens, rng) # max_num_tokens 길이 만큼 자르기

        assert len(tokens_a) >= 1 # 길이 검사
        assert len(tokens_b) >= 1 # 길이 검사

        tokens = [] # 학습 데이터 생성 시작
        segment_ids = []
        tokens.append("[CLS]") # 맨 앞에 CLS 토큰
        segment_ids.append(0) # 문장 번호 입력. A = 0, B = 1
        for token in tokens_a:
          tokens.append(token) # A에 있는 토큰 불러오기
          segment_ids.append(0) # A를 0으로 표시

        tokens.append("[SEP]") # 문장 구분자 추가
        segment_ids.append(0) # 구분자도 0

        for token in tokens_b: 
          tokens.append(token) # B에 있는 토큰 불러오기
          segment_ids.append(1) # B를 1로 표시
        tokens.append("[SEP]") # 문장 구분자 추가
        segment_ids.append(1) # B문장 구분자는 1

        (tokens, masked_lm_positions,
         masked_lm_labels) = create_masked_lm_predictions( # 마스킹 만들기
             tokens, masked_lm_prob, max_predictions_per_seq, vocab_words, rng)
        
        instance = TrainingInstance( # 생성된 데이터로 인스턴스 객체 생성
            tokens=tokens,
            segment_ids=segment_ids,
            is_random_next=is_random_next,
            masked_lm_positions=masked_lm_positions,
            masked_lm_labels=masked_lm_labels)
        instances.append(instance) # 학습데이터 모음 생성
      current_chunk = [] # 다시 초기화
      current_length = 0
    i += 1
  # 문서 끝나면 끝
  return instances # 인덱스 문서내 데이터 처리완료


MaskedLmInstance = collections.namedtuple("MaskedLmInstance",
                                          ["index", "label"])


def create_masked_lm_predictions(tokens, masked_lm_prob,
                                 max_predictions_per_seq, vocab_words, rng, do_whole_word_mask=False):
  """Creates the predictions for the masked LM objective."""

  cand_indexes = []
  for (i, token) in enumerate(tokens): # 전체 문장 돌면서
    if token == "[CLS]" or token == "[SEP]": # 특수 토큰은 패스
      continue
    # Whole Word Masking means that if we mask all of the wordpieces
    # corresponding to an original word. When a word has been split into
    # WordPieces, the first token does not have any marker and any subsequence
    # tokens are prefixed with ##. So whenever we see the ## token, we
    # append it to the previous set of word indexes.
    #
    # Note that Whole Word Masking does *not* change the training code
    # at all -- we still predict each WordPiece independently, softmaxed
    # over the entire vocabulary.
    if (do_whole_word_mask and len(cand_indexes) >= 1 and
        token.startswith("##")): # ##로 시작하는 토큰은 이전 토큰과 합치기 ex. play, ##ing -> playing
      cand_indexes[-1].append(i) # 이전 토큰에 같이 추가
    else:
      cand_indexes.append([i]) # ##으로 시작하지 않으면 그냥 추가

  rng.shuffle(cand_indexes) # 마스킹 후보들을 랜덤으로 섞기

  output_tokens = list(tokens) # 전체 토큰 불러와서

  num_to_predict = min(max_predictions_per_seq,
                       max(1, int(round(len(tokens) * masked_lm_prob)))) # 15%로 마스킹하지만 20개는 넘지 않도록

  masked_lms = [] # 마스킹된 토큰 기록
  covered_indexes = set() # 마스킹된 인덱스 기록
  for index_set in cand_indexes: # 후보들 전체 순환하면서
    if len(masked_lms) >= num_to_predict: # 설정한 마스크 개수 만큼 실행
      break
    # If adding a whole-word mask would exceed the maximum number of
    # predictions, then just skip this candidate.
    if len(masked_lms) + len(index_set) > num_to_predict: # ##때문에 한번에 2개가 들어있는거도 있음. 그런 경우 길이가 넘으면 패스
      continue
    is_any_index_covered = False # 중복 마스킹 방지
    for index in index_set:
      if index in covered_indexes:
        is_any_index_covered = True
        break
    if is_any_index_covered: # 중복 마스킹으로 선택되었으면 패스
      continue
    for index in index_set:
      covered_indexes.add(index) # 현재 인덱스를 가렸다고 저장

      masked_token = None
      # 80% of the time, replace with [MASK]
      if rng.random() < 0.8: # 80%로 MASK 토큰
        masked_token = "[MASK]"
      else:
        # 10% of the time, keep original
        if rng.random() < 0.5: # 20%의 50% = 10%는 그대로 유지
          masked_token = tokens[index]
        # 10% of the time, replace with random word
        else: # 나머지 10%는 아무 단어나 가져오기
          masked_token = vocab_words[rng.randint(0, len(vocab_words) - 1)]

      output_tokens[index] = masked_token # 토큰 마스킹. 바꿔치기

      masked_lms.append(MaskedLmInstance(index=index, label=tokens[index])) # 마스킹된 위치에 대한 데이터 추가
  assert len(masked_lms) <= num_to_predict # 위에서 break할때 확인했지만 다시 추가 확인
  masked_lms = sorted(masked_lms, key=lambda x: x.index) # 인덱스 기준으로 다시 정리

  masked_lm_positions = []
  masked_lm_labels = []
  for p in masked_lms: # 마스킹 된 데이터 두개로 정리
    masked_lm_positions.append(p.index) # 인덱스만
    masked_lm_labels.append(p.label) # 데이터만

  return (output_tokens, masked_lm_positions, masked_lm_labels) # 마스킹된 데이터, 마스킹된 위치, 마스킹된 단어들


def truncate_seq_pair(tokens_a, tokens_b, max_num_tokens, rng):
  """Truncates a pair of sequences to a maximum sequence length."""
  while True:
    total_length = len(tokens_a) + len(tokens_b) # 전체 길이 확인
    if total_length <= max_num_tokens: # 최대치 안넘으면 그냥 반환
      break

    trunc_tokens = tokens_a if len(tokens_a) > len(tokens_b) else tokens_b # A나 B 중에서 선택
    assert len(trunc_tokens) >= 1

    # We want to sometimes truncate from the front and sometimes from the
    # back to add more randomness and avoid biases.
    if rng.random() < 0.5: # 50%로 앞에서 빼거나, 50%로 뒤에서 빼기
      del trunc_tokens[0]
    else:
      trunc_tokens.pop()


def main(_):
  tf.logging.set_verbosity(tf.logging.INFO)

  tokenizer = tokenization.FullTokenizer(
      vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)

  input_files = []
  for input_pattern in FLAGS.input_file.split(","):
    input_files.extend(tf.gfile.Glob(input_pattern))

  tf.logging.info("*** Reading from input files ***")
  for input_file in input_files:
    tf.logging.info("  %s", input_file)

  rng = random.Random(FLAGS.random_seed)
  instances = create_training_instances(
      input_files, tokenizer, FLAGS.max_seq_length, FLAGS.dupe_factor,
      FLAGS.short_seq_prob, FLAGS.masked_lm_prob, FLAGS.max_predictions_per_seq,
      rng)

  output_files = FLAGS.output_file.split(",")
  tf.logging.info("*** Writing to output files ***")
  for output_file in output_files:
    tf.logging.info("  %s", output_file)

  write_instance_to_example_files(instances, tokenizer, FLAGS.max_seq_length,
                                  FLAGS.max_predictions_per_seq, output_files)


if __name__ == "__main__":
  flags.mark_flag_as_required("input_file")
  flags.mark_flag_as_required("output_file")
  flags.mark_flag_as_required("vocab_file")
  tf.app.run()
