#include "path_trie.h"

#include <string>
#include <algorithm>
#include <iostream>
#include <limits>
#include <memory>
#include <utility>
#include <vector>
#include <map>
#include <unordered_map>

#include "decoder_utils.h"
#include "scorer.h"

PathTrie::PathTrie() {
  log_prob_b_prev = -NUM_FLT_INF;
  log_prob_nb_prev = -NUM_FLT_INF;
  log_prob_b_cur = -NUM_FLT_INF;
  log_prob_nb_cur = -NUM_FLT_INF;
  score = -NUM_FLT_INF;

  ROOT_ = -1;
  character = ROOT_;
  timestep = 0;
  exists_ = true;
  parent = nullptr;

  dictionary_ = nullptr;
  dictionary_state_ = 0;
  has_dictionary_ = false;

  matcher_ = nullptr;
}

PathTrie::~PathTrie() {
  for (auto child : children_) {
    delete child.second;
  }
}

PathTrie* PathTrie::get_path_trie(int new_char, int new_timestep, bool ignore_tokenization_symbol, bool reset) {
  auto child = children_.begin();
  for (child = children_.begin(); child != children_.end(); ++child) {
    if (child->first == new_char) {
      break;
    }
  }
  if (child != children_.end()) {
    if (!child->second->exists_) {
      child->second->exists_ = true;
      child->second->log_prob_b_prev = -NUM_FLT_INF;
      child->second->log_prob_nb_prev = -NUM_FLT_INF;
      child->second->log_prob_b_cur = -NUM_FLT_INF;
      child->second->log_prob_nb_cur = -NUM_FLT_INF;
    }
    return (child->second);
  } else {
    if (has_dictionary_) {
      // note to self:
      // when using tokenization symbols we need to make sure that matching is not done.
      // Also, if it is a tokenization symbol then reset the dictionary to start for next set of matching.
      if (ignore_tokenization_symbol){
        PathTrie* new_path = new PathTrie;
        // reset dictionary state
        dictionary_state_ = dictionary_->Start();
        new_path->character = new_char;
        new_path->timestep = new_timestep;
        new_path->parent = this;
        new_path->dictionary_ = dictionary_;
        new_path->has_dictionary_ = true;
        new_path->matcher_ = matcher_;
        children_.push_back(std::make_pair(new_char, new_path));
        return new_path;
      }
      matcher_->SetState(dictionary_state_);
      bool found = matcher_->Find(new_char);
      if (!found) {
        // Adding this character causes word outside dictionary
        auto FSTZERO = fst::TropicalWeight::Zero();
        auto final_weight = dictionary_->Final(dictionary_state_);
        bool is_final = (final_weight != FSTZERO);
        if (is_final && reset) {
          dictionary_state_ = dictionary_->Start();
        }
        return nullptr;
      } else {
        PathTrie* new_path = new PathTrie;
        new_path->character = new_char;
        new_path->timestep = new_timestep;
        new_path->parent = this;
        new_path->dictionary_ = dictionary_;
        new_path->dictionary_state_ = matcher_->Value().nextstate;
        new_path->has_dictionary_ = true;
        new_path->matcher_ = matcher_;
        children_.push_back(std::make_pair(new_char, new_path));
        return new_path;
      }
    } else {
      PathTrie* new_path = new PathTrie;
      new_path->character = new_char;
      new_path->timestep = new_timestep;
      new_path->parent = this;
      children_.push_back(std::make_pair(new_char, new_path));
      return new_path;
    }
  }
}

PathTrie* PathTrie::get_path_vec(std::vector<int>& output, std::vector<int>& timesteps) {
  // a fake map to interface with get_path_vec function.
  std::unordered_map<int, std::string> _root_stop;
  return get_path_vec(output, timesteps, _root_stop);
}

PathTrie* PathTrie::get_path_vec(std::vector<int>& output,
                                 std::vector<int>& timesteps,
                                 std::unordered_map<int, std::string>& tokenization_symbol_map,
                                 size_t max_steps) {
  if ((!tokenization_symbol_map.empty() && tokenization_symbol_map.find(character) != tokenization_symbol_map.end()) ||
      character == ROOT_ ||
      output.size() == max_steps) {
    std::reverse(output.begin(), output.end());
    std::reverse(timesteps.begin(), timesteps.end());
    return this;
  } else {
    output.push_back(character);
    timesteps.push_back(timestep);
    return parent->get_path_vec(output, timesteps, tokenization_symbol_map, max_steps);
  }
}

void PathTrie::iterate_to_vec(std::vector<PathTrie*>& output) {
  if (exists_) {
    log_prob_b_prev = log_prob_b_cur;
    log_prob_nb_prev = log_prob_nb_cur;

    log_prob_b_cur = -NUM_FLT_INF;
    log_prob_nb_cur = -NUM_FLT_INF;

    score = log_sum_exp(log_prob_b_prev, log_prob_nb_prev);
    output.push_back(this);
  }
  for (auto child : children_) {
    child.second->iterate_to_vec(output);
  }
}

void PathTrie::remove() {
  exists_ = false;

  if (children_.size() == 0) {
    auto child = parent->children_.begin();
    for (child = parent->children_.begin(); child != parent->children_.end();
         ++child) {
      if (child->first == character) {
        parent->children_.erase(child);
        break;
      }
    }

    if (parent->children_.size() == 0 && !parent->exists_) {
      parent->remove();
    }

    delete this;
  }
}

void PathTrie::set_dictionary(fst::StdVectorFst* dictionary) {
  dictionary_ = dictionary;
  dictionary_state_ = dictionary->Start();
  has_dictionary_ = true;
}

using FSTMATCH = fst::SortedMatcher<fst::StdVectorFst>;
void PathTrie::set_matcher(std::shared_ptr<FSTMATCH> matcher) {
  matcher_ = matcher;
}
