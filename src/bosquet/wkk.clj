(ns bosquet.wkk)

;;
;; Well Known Keys to refer to various concepts properties,
;; system components, etc.
;;

(def llm-config
  "Key to reference LLM service components in props when making `generate` calls."
  :bosquet.llm/llm-config)

(def default-llm
  "Key referencing default LLM service in a `system.edn` system config."
  :llm/default-llm)

(def service
  "Key to reference LLM service name in gen call parameters"
  :bosquet.llm/service)

(def model-parameters
  "Key to reference LLM model parameters in gen call parameters"
  :bosquet.llm.model/parameters)

(def output-format
  "Type of generation output format: json, xml, text, etc"
  :bosquet.llm.output/format)

(def cache
  :bosquet.llm/cache)

(def default-gen-var-name
  "`gen` template tag is allowed to not specify `var` name parameter.
   In this case this name will be used"
  :gen)

(def gen-var-name
  "Key to reference LLM generation var name in gen call parameters"
  :bosquet.llm/gen-var-name)

;;
;; Memory
;;

(def memory-config
  "Key to reference memory in configuration"
  :bosquet.memory/config)

(def recall-free :memory.recall/free)
(def recall-sequential :memory.recall/sequential)
(def recall-cue :memory.recall/cue)

(def simple-short-term-memory :memory/simple-short-term)

(def recall-function
  "Function to retrieve memory using passed in memory type"
  :bosquet.memory/recall-function)

(def memory-parameters
  "Memory parameters to be used when creating and retrieving a memory"
  :bosquet.memory/parameters)

(def memory-type
  :bosquet.memory/type)
