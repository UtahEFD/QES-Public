#pragma once

#include <unordered_map>
#include <any>

/**
 * \brief QES Data Transport dictionary container.
 *
 * This class can be used to contain data in a name, value pair
 * system. The intended use would be to transport data between
 * components in way that is decoupled from the rest of the implementation.
 */
class QESDataTransport
{
private:
  // Name value pair storage that uses std::any to safely store
  // generic data, rather than using void* casting mechanisms.
  // - See https://en.cppreference.com/w/cpp/utility/any
  std::unordered_map<std::string, std::any> dataStorageContainer;

public:
  /**
   * /brief Adds a key value pair to the container.
   *
   * Given a key (as a string), and a value of any type, this will store
   * the data in the container for retrieval later.
   *
   * @param key   The key for the data.
   * @param value The value stored with the key.
   */
  template<typename T>
  void put(const std::string &key, T value)
  {
    dataStorageContainer[key] = value;// should this be a move?
  }

  /**
   * /brief Obtains the value associated with a key and returns it.  The template
   * type is required when called. If the type is incorrect, it will throw
   * a bad cast exception.
   *
   * Retrieves the value associate with the key. The value will be cast to the
   * type specified in the template function call. If the cast is unsuccessful,
   * a runtime_error exception will be thrown to indicate the type was not correct.
   *
   * If the key is not in the container, a runtime_error exception will be thrown to
   * indicate the key was not found.
   *
   * @param key   The key of the data to retrieve.
   * @return The value associated with the key, cast to the specified type.
   * @throws std::bad_any_cast if the value is not of the expected type.
   */
  template<typename T>
  T get(const std::string &key) const
  {
    if (dataStorageContainer.find(key) == dataStorageContainer.end()) {
      throw std::runtime_error("Key not found: " + key);
    }

    try {
      return std::any_cast<T>(dataStorageContainer.at(key));
    } catch (const std::bad_any_cast &e) {
      throw std::runtime_error("Type mismatch for key: " + key);
    }
  }

  /**
   * /brief Obtains the reference to the value associated with a key and returns it.
   * The template type is required when called. If the type is incorrect,
   * it will throw a bad cast exception.
   *
   * Retrieves the value associate with the key. The value will be cast to the
   * type specified in the template function call. If the cast is unsuccessful,
   * a runtime_error exception will be thrown to indicate the type was not correct.
   *
   * If the key is not in the container, a runtime_error exception will be thrown to
   * indicate the key was not found.
   *
   * @param key   The key of the data to retrieve.
   * @return The reference to the value associated with the key, cast to the specified type.
   * @throws std::bad_any_cast if the value is not of the expected type.
   */
  template<typename T>
  T &get_ref(const std::string &key)
  {
    if (dataStorageContainer.find(key) == dataStorageContainer.end()) {
      throw std::runtime_error("Key not found: " + key);
    }

    try {
      return std::any_cast<T &>(dataStorageContainer.at(key));
    } catch (const std::bad_any_cast &e) {
      throw std::runtime_error("Type mismatch for key: " + key);
    }
  }

  /**
   * /brief Function to help see if the container contains a key.
   *
   * This function will query the container to determine if it contains the
   * key. Useful in situations where parameters might be optional.
   * if (container.contains("aKey") == true) {
   *    // call the associated container.get function to retrieve the value
   * }
   *
   * @param key   The key to check.
   * @return True if the key exists, false otherwise.
   */
  bool contains(const std::string &key) const
  {
    return dataStorageContainer.find(key) != dataStorageContainer.end();
  }
};
