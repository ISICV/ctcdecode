#include "scorer.h"

#include <unistd.h>
#include <iostream>
#include <vector>
#include <map>
#include <unordered_map>

#include "lm/config.hh"
#include "lm/model.hh"
#include "lm/state.hh"
#include "util/string_piece.hh"
#include "util/tokenize_piece.hh"

#include "decoder_utils.h"

using namespace lm::ngram;

Scorer::Scorer(double alpha,
               double beta,
               const std::string& lm_path,
               const std::vector<std::string>& char_list,
               const std::vector<std::string>& tokenization_char_list) {
  this->alpha = alpha;
  this->beta = beta;

  dictionary = nullptr;
  is_character_based_ = true;
  language_model_ = nullptr;

  max_order_ = 0;
  dict_size_ = 0;
  
  setup(lm_path, char_list, tokenization_char_list);
}

Scorer::~Scorer() {
  if (language_model_ != nullptr) {
    delete static_cast<lm::base::Model*>(language_model_);
  }
  if (dictionary != nullptr) {
    delete static_cast<fst::StdVectorFst*>(dictionary);
  }
}

void Scorer::setup(const std::string& lm_path,
                   const std::vector<std::string>& char_list,
                   const std::vector<std::string>& tokenization_char_list) {
  // load language model
  load_lm(lm_path);
  // set char map for scorer
  set_char_map(char_list);
  // set tokenization symbol set based on char_map
  set_tokenization_char_map(tokenization_char_list);
  // fill the dictionary for FST
  if (!is_character_based()) {
    fill_dictionary();
  }
}

void Scorer::load_lm(const std::string& lm_path) {
  const char* filename = lm_path.c_str();
  VALID_CHECK_EQ(access(filename, F_OK), 0, "Invalid language model path");

  RetriveStrEnumerateVocab enumerate;
  lm::ngram::Config config;
  config.enumerate_vocab = &enumerate;
  language_model_ = lm::ngram::LoadVirtual(filename, config);
  max_order_ = static_cast<lm::base::Model*>(language_model_)->Order();
  vocabulary_ = enumerate.vocabulary;
  for (size_t i = 0; i < vocabulary_.size(); ++i) {
    if (is_character_based_ && vocabulary_[i] != UNK_TOKEN &&
        vocabulary_[i] != START_TOKEN && vocabulary_[i] != END_TOKEN &&
        enumerate.vocabulary[i].length() > 5) { // assuming a single char is of form UXXXX
      is_character_based_ = false;
      break;
    }
  }
}

double Scorer::get_log_cond_prob(const std::vector<std::string>& words) {
  lm::base::Model* model = static_cast<lm::base::Model*>(language_model_);
  double cond_prob;
  lm::ngram::State state, tmp_state, out_state;
  // avoid to inserting <s> in begin
  model->NullContextWrite(&state);
  for (size_t i = 0; i < words.size(); ++i) {
    lm::WordIndex word_index = model->BaseVocabulary().Index(words[i]);
    // encounter OOV
    if (word_index == 0) {
      return OOV_SCORE;
    }
    cond_prob = model->BaseScore(&state, word_index, &out_state);
    tmp_state = state;
    state = out_state;
    out_state = tmp_state;
  }
  // return loge prob
  return cond_prob/NUM_FLT_LOGE;
}

double Scorer::get_sent_log_prob(const std::vector<std::string>& words) {
  std::vector<std::string> sentence;
  if (words.size() == 0) {
    for (size_t i = 0; i < max_order_; ++i) {
      sentence.push_back(START_TOKEN);
    }
  } else {
    for (size_t i = 0; i < max_order_ - 1; ++i) {
      sentence.push_back(START_TOKEN);
    }
    sentence.insert(sentence.end(), words.begin(), words.end());
  }
  sentence.push_back(END_TOKEN);
  return get_log_prob(sentence);
}

double Scorer::get_log_prob(const std::vector<std::string>& words) {
  assert(words.size() > max_order_);
  double score = 0.0;
  for (size_t i = 0; i < words.size() - max_order_ + 1; ++i) {
    std::vector<std::string> ngram(words.begin() + i,
                                   words.begin() + i + max_order_);
    score += get_log_cond_prob(ngram);
  }
  return score;
}

void Scorer::reset_params(float alpha, float beta) {
  this->alpha = alpha;
  this->beta = beta;
}

std::string Scorer::vec2str(const std::vector<int>& input) {
  // todo: might need to write logic to input space before and after a stop_symbol token eg: " u0020 "
  std::string word;
  //---- testing purpose only!!
  if(input.size() && input[0] == -1)
    return "_ROOT";
  //-----
  for (int i = 0; i < input.size(); i++) {
    if(i != 0)
      word += "_";
    word += char_list_[input[i]];
  }
  return word;
}

std::vector<std::string> Scorer::split_labels(const std::vector<int>& labels) {
  if (labels.empty()) return {};
  
  if (is_character_based_) {
    std::cerr<<"splitting char model not defined."<<std::endl;
    std::exit(1);
  }
  
  std::vector<std::string> words;
  std::string current_string;
  bool start_of_new_word = true;
  for(auto label: labels){
    if(tokenization_char_map_.find(label) != tokenization_char_map_.end()){
      if(current_string.length() > 0) {
        words.push_back(current_string);
      }
      words.push_back(char_list_[label]);
      start_of_new_word = true;
      current_string = "";
    }
    else{
      if(!start_of_new_word) {
        current_string += '_';
      }
      current_string += char_list_[label];
      start_of_new_word = false;
    }
  }
  if(current_string.length() > 0) {
    words.push_back(current_string);
  }
  return words;
}

void Scorer::set_char_map(const std::vector<std::string>& char_list) {
  char_list_ = char_list;
  char_map_.clear();
  for (size_t i = 0; i < char_list_.size(); i++) {
      char_map_[char_list_[i]] = i;
  }
}

void Scorer::set_tokenization_char_map(const std::vector<std::string>& tokenization_char_list){
  for(auto character: tokenization_char_list){
    std::string uxxxx_char = character;
    if (char_map_.find(uxxxx_char) != char_map_.end()){
       int pos = char_map_[uxxxx_char];
       tokenization_char_map_[pos] = uxxxx_char;
    }
    else{
      std::cerr<<"tokenization char:"<<uxxxx_char<<" not defined in vocabulary"<<std::endl;
      std::exit(1);
    }
  }
}

std::vector<std::string> Scorer::make_ngram(PathTrie* prefix) {
  std::vector<std::string> ngram;
  PathTrie* current_node = prefix;
  PathTrie* new_node = nullptr;
  
  for (int order = 0; order < max_order_; order++) {
    std::vector<int> prefix_vec;
    std::vector<int> prefix_steps;

    if (is_character_based_) {
      std::cerr<<"undefined behaviour when using tokenization_char_map_ with character model."<<std::endl;
      std::exit(1);
      new_node = current_node->get_path_vec(prefix_vec, prefix_steps, tokenization_char_map_, 1);
      current_node = new_node;
    } else {
      if(tokenization_char_map_.find(current_node->character) != tokenization_char_map_.end()){
        // logic to push back stop_symbols into the ngram as seperate tokens
        prefix_vec.push_back(current_node->character);
        prefix_steps.push_back(current_node->timestep);
        new_node = current_node->parent;
      }
      else{
        // read till stop symbol.
        new_node = current_node->get_path_vec(prefix_vec, prefix_steps, tokenization_char_map_);
      }
      current_node = new_node;
    }

    // reconstruct word
    std::string word = vec2str(prefix_vec);
    if (word.length() > 0)
      ngram.push_back(word);

    if (new_node == nullptr || new_node->character == -1) {
      // if new_node is at ROOT_
      // No more spaces, but still need order
      for (int i = 0; i < max_order_ - order - 1; i++) {
        ngram.push_back(START_TOKEN);
      }
      break;
    }
  }
  std::reverse(ngram.begin(), ngram.end());
  return ngram;
}

void Scorer::fill_dictionary() {
  fst::StdVectorFst dictionary;

  // For each unigram convert to ints and put in trie
  int dict_size = 0;
  for (const auto& word : vocabulary_) {
    bool added = add_word_to_dictionary(word, char_map_, &dictionary);
    dict_size += added ? 1 : 0;
  }

  dict_size_ = dict_size;

  /* Simplify FST

   * This gets rid of "epsilon" transitions in the FST.
   * These are transitions that don't require a string input to be taken.
   * Getting rid of them is necessary to make the FST determinisitc, but
   * can greatly increase the size of the FST
   */
  fst::RmEpsilon(&dictionary);
  fst::StdVectorFst* new_dict = new fst::StdVectorFst;

  /* This makes the FST deterministic, meaning for any string input there's
   * only one possible state the FST could be in.  It is assumed our
   * dictionary is deterministic when using it.
   * (lest we'd have to check for multiple transitions at each state)
   */
  fst::Determinize(dictionary, new_dict);

  /* Finds the simplest equivalent fst. This is unnecessary but decreases
   * memory usage of the dictionary
   */
  fst::Minimize(new_dict);
  this->dictionary = new_dict;
}
