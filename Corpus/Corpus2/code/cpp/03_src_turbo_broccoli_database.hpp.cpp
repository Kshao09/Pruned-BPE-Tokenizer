/*
 * database.hpp
 *
 *  Created on: Sep 22, 2016
 *      Author: dan
 */

#ifndef SRC_TURBO_BROCCOLI_DATABASE_HPP_
#define SRC_TURBO_BROCCOLI_DATABASE_HPP_

#include <boost/filesystem.hpp>

#include <turbo_broccoli/type/key.hpp>
#include <turbo_broccoli/type/value.hpp>
#include <turbo_broccoli/type/blob.hpp>
#include <turbo_broccoli/type/tags.hpp>
#include <turbo_broccoli/type/tagged_records.hpp>
#include <turbo_broccoli/type/result_find.hpp>
#include <turbo_broccoli/type/result_key.hpp>

#include <turbo_broccoli/detail/utils.hpp>


namespace turbo_broccoli {

using types::blob;
using types::db_key;
using types::result_key;
using types::result_find;

struct database {

  database(const std::string& path) : path_(path) {
    namespace fs = boost::filesystem;

    if(!fs::exists(path_) ) {
      if(!fs::create_directories(path_)) {
        throw std::runtime_error("cannot open db, cannot create directory: " + path_.generic_string());
      }
    }
    else {
      if(!fs::is_directory(path_)) {
        throw std::runtime_error("cannot open db, is not a directory: " + path_.generic_string());
      }
    }
  }


  result_find find(const std::string& key) {
    return find(detail::calculate_key(key));
  }

  result_find find(const db_key& key) {

    result_find result{};
    result.success = false;

    if(!record_exists(key)) {
      std::cout << "no record with key" << types::to_string(key) << std::endl;
      return result;
    }

    auto record = read_record(key);


    if( is_blob(record)) {
      result.success = true;
      result.results.push_back(detail::deserialize<blob>(record.data));
    }

    if(is_tag_list(record)) {
      auto records = detail::deserialize<types::tagged_records>(record.data);
      for(auto& t : records.keys) {
        auto k = types::string_to_key(t);
        if(record_exists(k)) {

          auto r = read_record(k);
          if( is_blob(r)) {
            result.success = true;
            result.results.push_back(detail::deserialize<blob>(r.data));
          }
          else {
            std::cout << "inconsistent: record is not blob " << t << std::endl;
          }
        }
        else {
          std::cout << "inconsistent no record from tag list " << t << std::endl;
        }
      }
    }

    return result;
  }


  result_key store(const blob& new_blob) {

    static const result_key failed_result{false, turbo_broccoli::types::nil_key() };

    if(record_exists(new_blob)) {
      /*
       * read all tags and update them!
       */
      auto r = read_record(new_blob.key_hash());
      auto old_blob = detail::deserialize<blob>(r.data);

      types::tag_list::list_type to_delete = diff( old_blob.tags().tags, new_blob.tags().tags);
      types::tag_list::list_type to_add    = diff( new_blob.tags().tags, old_blob.tags().tags);

      for(auto& t : to_add ) {
        update_tag_add(t, types::to_string(new_blob.key_hash()));
      }
      for(auto& t : to_delete ) {
        update_tag_remove(t, types::to_string(new_blob.key_hash()));
      }

    }
    else {
      detail::create_folder(path_, new_blob.key_hash());
      for(auto& t : new_blob.tags().tags ) {
        update_tag_add(t, types::to_string(new_blob.key_hash()));
      }
    }
    write_blob(new_blob);
    return {true, new_blob.key_hash()};



    return failed_result;
  }



private:



  inline bool record_exists(const blob& b) {
    namespace fs = boost::filesystem;
    return fs::exists(detail::to_filename(path_, b.key_hash()));
  }

  inline bool record_exists(const db_key& k) {
    namespace fs = boost::filesystem;
    return fs::exists(detail::to_filename(path_, k));
  }

  inline void write_blob(const blob& b) {
    namespace fs = boost::filesystem;
    types::value_t v;
    v.data = detail::serialize(b);
    v.reccord_type = types::value_type::blob;
    v.key = b.key();

    detail::create_folder(path_, b.key_hash());
    detail::write_file(detail::to_filename(path_, b.key_hash()).generic_string(), detail::serialize(v));



  }

  inline types::value_t read_record(const db_key& k) {
    namespace fs = boost::filesystem;
    auto tmp = detail::read_file(detail::to_filename(path_, k).generic_string() );
    return detail::deserialize<types::value_t>(tmp);
  }

  inline void update_tag_add(const std::string& tag_name, const std::string& record_key) {
    auto tag_key = detail::calculate_key(tag_name);
    types::value_t v;
    types::tagged_records records;

    if(record_exists(tag_key)) {
      v = read_record(tag_key);
      if(types::is_tag_list(v)) {
        records = detail::deserialize<types::tagged_records>(v.data);
        for(auto& r : records.keys) {
          if(record_key.compare(r) == 0) {
            return;
          }
        }
        records.keys.push_back(record_key);
      }
      else {
        throw std::runtime_error("record exissts and is not a tagged_list: " + tag_name);
      }
    }
    else {
      records.keys.push_back(record_key);
      v.key           = tag_name;
      v.reccord_type  = types::value_type::tag_list;
      v.data          = detail::serialize(records);
      detail::create_folder(path_, tag_key);
    }
    v.data          = detail::serialize(records);
    detail::write_file(detail::to_filename(path_, tag_key).generic_string(), detail::serialize(v));
  }

  inline void update_tag_remove(const std::string& tag_name, const std::string& record_key) {
    auto tag_key = detail::calculate_key(tag_name);
    types::value_t v = read_record(tag_key);

    if(types::is_tag_list(v)) {
      types::tagged_records records = detail::deserialize<types::tagged_records>(v.data);
      records.keys.erase(std::remove(records.keys.begin(), records.keys.end(), record_key), records.keys.end());

      v.data = detail::serialize(records);
      detail::write_file(detail::to_filename(path_, tag_key).generic_string(), detail::serialize(v));
    }
  }


  /*
   * \brief return list of all elements that are only in a
   * a{0, 1, 2, 3, 4}
   * b{3, 4, 5, 6, 7}
   * d{0, 1, 2}
   */
  inline std::vector<std::string> diff(const std::vector<std::string>& a, const std::vector<std::string>& b) {
    std::vector<std::string> d;
    for(auto& a_i : a) {
       bool contains_b_i{false};
       for(auto& b_i : b) {
         if(a_i.compare(b_i) == 0) {
           contains_b_i = true;
           break;
         }
       }

       if(!contains_b_i) {
         d.push_back(a_i);
       }
    }
    return d;
  }

  using path_t = boost::filesystem::path;
  path_t path_;

};


}




#endif /* SRC_TURBO_BROCCOLI_DATABASE_HPP_ */
