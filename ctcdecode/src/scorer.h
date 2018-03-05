#ifndef SCORER_H_
#define SCORER_H_

#include <memory>
#include <string>
#include <unordered_map>
#include <set>
#include <vector>

#include "lm/enumerate_vocab.hh"
#include "lm/virtual_interface.hh"
#include "lm/word_index.hh"
#include "util/string_piece.hh"

#include "path_trie.h"

const double OOV_SCORE = -1000.0;
const std::string START_TOKEN = "<s>";
const std::string UNK_TOKEN = "<unk>";
const std::string END_TOKEN = "</s>";

// Todo: Remove hardcoded values and use unicode categories
const std::set<std::string> UXXXX_PUNCTUATIONS = {"u0020", "u002e", "u002c", "u003b", "u0027", "u0022", "u002f", "u0021", "u0028",
                                                  "u0029", "u005b", "u005d","u003f", "u003c", "u003e", "u002d", "u005f", "u007b",
                                                  "u007d", "u0024", "u0025", "u0023", "u0026", "u002a"};
const std::set<std::string> UXXXX_DIGITS = {"u0030", "u0031", "u0032", "u0033", "u0034", "u0035", "u0036", "u0037",
                                            "u0038", "u0039"};

// Implement a callback to retrive the dictionary of language model.
class RetriveStrEnumerateVocab : public lm::EnumerateVocab {
public:
  RetriveStrEnumerateVocab() {}

  void Add(lm::WordIndex index, const StringPiece &str) {
    vocabulary.push_back(std::string(str.data(), str.length()));
  }

  std::vector<std::string> vocabulary;
};

/* External scorer to query score for n-gram or sentence, including language
 * model scoring and word insertion.
 *
 * Example:
 *     Scorer scorer(alpha, beta, "path_of_language_model");
 *     scorer.get_log_cond_prob({ "WORD1", "WORD2", "WORD3" });
 *     scorer.get_sent_log_prob({ "WORD1", "WORD2", "WORD3" });
 */
class Scorer {
public:
  Scorer(double alpha,
         double beta,
         const std::string &lm_path,
         const std::vector<std::string> &vocabulary);
  ~Scorer();

  double get_log_cond_prob(const std::vector<std::string> &words);

  double get_sent_log_prob(const std::vector<std::string> &words);

  // return the max order
  size_t get_max_order() const { return max_order_; }

  // return the dictionary size of language model
  size_t get_dict_size() const { return dict_size_; }

  // retrun true if the language model is character based
  bool is_character_based() const { return is_character_based_; }

  // reset params alpha & beta
  void reset_params(float alpha, float beta);

  // make ngram for a given prefix
  std::vector<std::string> make_ngram(PathTrie *prefix);

  // trransform the labels in index to the vector of words (word based lm) or
  // the vector of characters (character based lm)
  std::vector<std::string> split_labels(const std::vector<int> &labels);

  // language model weight
  double alpha;
  // word insertion weight
  double beta;
  // char list
  std::vector<std::string> char_list_;
  // stop symbols defined for tokenization logic
  std::unordered_map<int, std::string> stop_symbol_map_;
  // pointer to the dictionary of FST
  void *dictionary;

protected:
  // necessary setup: load language model, set char map, fill FST's dictionary
  void setup(const std::string &lm_path,
             const std::vector<std::string> &vocab_list);

  // load language model from given path
  void load_lm(const std::string &lm_path);

  // fill dictionary for FST
  void fill_dictionary();

  // set char map
  void set_char_map(const std::vector<std::string> &char_list);

  // set tokenization symbols map
  // maps int positions of punctuation/digits to act as tokenizing points.
  void set_stop_symbol_map();
  
  double get_log_prob(const std::vector<std::string> &words);

  // translate the vector in index to string
  std::string vec2str(const std::vector<int> &input);

private:
  void *language_model_;
  bool is_character_based_;
  size_t max_order_;
  size_t dict_size_;


  std::unordered_map<std::string, int> char_map_;
  std::vector<std::string> vocabulary_;
};

#endif  // SCORER_H_
