(ns bosquet.agent.react-agent
  (:require
   [bosquet.agent.graph :refer [defgraph defnode defagent]]
   [bosquet.llm.generator :as g]
   [bosquet.llm.tools :as tools]
   [bosquet.llm.wkk :as wkk]
   [cheshire.core]
   [clojure.string :as str]
   [taoensso.timbre :as timbre]))


#_(defn create-react-prompt
  "Create ReAct prompt with examples and current state"
  [{:keys [examples task step reasoning-trace observation tool-descriptions]}]
  (let [base-prompt (str (when examples (str "EXAMPLES:\n" examples "\n\nCURRENT TASK:\n"))
                        "You have access to the following tools: " tool-descriptions "\n\n"
                        "Use actual tool names with quoted strings: get-current-weather[\"San Francisco\"] or add[15, 27]\n"
                        "CRITICAL: Before taking any action, write down a step-by-step plan of how to solve the problem."
                        "CRITICAL: Stop after generating ONE Action. Do not continue with Observation or next Thought.\n"
                        "CRITICAL: Generate ONLY one Thought and one Action per response. DO NOT generate observations.\n"
                        "CRITICAL: You MUST terminate with a Finish[...] action once the original Question is fully answered.\n"
                        "   - DO NOT invent new problems.\n"
                        "   - DO NOT continue reasoning after producing the requested final answer.\n"
                        "   - The only valid final output is:\n" 
                        "     Action <step>: Finish[<final answer>]\n" 
                        "CRITICAL: You MUST continue the numbering sequence from the current step =" step ".\n"
                        "   - Do not restart at Thought 1 or Action 1.\n"
                        "   - The only valid numbers are" step "for the current pair, then " (inc step) " for the next after system response.\n"
                        "Format your response EXACTLY as:\n"
                        "Thought " step ": [your reasoning]\n"
                        "Action " step ": tool-name[parameters] OR Finish[answer]\n"
                        "STOP HERE. Wait for system response.\n\n"
                        "Question: " task "\n")]
    (if reasoning-trace
      (str base-prompt reasoning-trace 
           (when observation (str "Observation " step ": " observation "\n")))
      base-prompt)))

 (defn create-react-prompt
  "Create ReAct prompt with examples and current state.
   Rules:
   - Generate a step-by-step plan (Thought) before each action.
   - Generate only one Action per response.
   - Each Thought/Action must reference only the Question or previous Observations.
   - Do not repeat the same tool call.
   - LLM may choose Finish[answer] when enough info is available.
   - Step numbers increment continuously."
  [{:keys [examples task step reasoning-trace observation tool-descriptions]}]
  (let [base-prompt
        (str
          (when examples
            (str "EXAMPLES:\n" examples "\n\nCURRENT TASK:\n"))
          "You have access to the following tools: " tool-descriptions "\n\n"
          "Use actual tool names with quoted strings: get-current-weather[\"San Francisco\"] or add[15, 27]\n\n"

          ;; Step-by-step and single action
          "CRITICAL:\n"
          "1. Before taking any action, write a step-by-step plan in Thought.\n"
          "2. Generate ONLY one Thought and one Action per response. DO NOT generate Observations\n"
          "3. Each Thought/Action must use only arguments from the Question or previous Observations.\n"
          "4. Do not repeat the same tool call with identical arguments.\n"
          "5. You may emit Finish[answer] when you believe you have enough information to answer the Question.\n"
          "6. Do not invent new numbers, cities, or entities unrelated to the Question.\n"
          "7. Continue step numbering from current step = " step ".\n"
          "   - Thought numbers must match the step number.\n"
          "   - Action numbers must match the step number.\n\n"

          ;; Format enforcement
          "Format your response EXACTLY as:\n"
          "Thought " step ": [your reasoning]\n"
          "Action " step ": tool-name[parameters] OR Finish[answer]\n"
          "STOP HERE. Wait for system response.\n\n"

          "Question: " task "\n")]
    (if reasoning-trace
      (str base-prompt reasoning-trace
           (when observation (str "Observation " step ": " observation "\n")))
      base-prompt)))

(defn prepare-tools
  "Create a comprehensive tool map with symbols, descriptions, and metadata"
  [tool-vars]
  (into {} 
    (map (fn [tool-var]
           (let [tool-name (keyword (str/lower-case (name (:name (meta tool-var)))))
                 tool-fn (tools/tool->function tool-var)
                 func-info (get tool-fn :function)]
             [tool-name {:symbol tool-var
                         :name (:name func-info)
                         :description (:description func-info)
                         :metadata func-info}]))
         tool-vars)))

(defn get-tool-descriptions
  "Get tool descriptions from comprehensive tool map"
  [tool-map]
  (str/join ", " 
    (map (fn [[_ tool-info]]
           (str (:name tool-info) " - " (:description tool-info)))
         tool-map)))

(defn ->llm
  [llm-spec var-name & [tool-array]]
  (let [model-params (if tool-array 
                       (merge (wkk/model-params llm-spec) {wkk/tools tool-array})
                       (wkk/model-params llm-spec))]
    (g/llm (:llm llm-spec) wkk/model-params model-params wkk/var-name var-name wkk/cache false)))

(defn parse-action-response
  "Parse LLM response to extract action and parameters"
  [response]
  (when (and response (string? response))
    (cond 
     ;; Check for Final Answer first
     (re-find #"Finish" response)
     {:action :final-answer
      :full-response response
      :final-answer (when-let [match (re-find #"Finish\s*(.+)" response)]
                     (str/trim (second match)))}
     
     ;; Try Action N: format with parameters in brackets
     (re-find #"Action\s+(?:\d+|N):\s*(\S+)" response)
     (let [action-match (re-find #"Action\s+(?:\d+|N):\s*(.+)" response)
           [_ action-text] action-match
           ;; Parse tool-name[params] format  
           [tool-name params] (if-let [bracket-match (re-find #"^([^\[\s]+)\[([^\]]+)\]" action-text)]
                               (let [tool-name (second bracket-match)
                                     params-str (str "[" (nth bracket-match 2) "]")]
                                 [tool-name 
                                  (try
                                    (clojure.edn/read-string params-str)
                                    (catch Exception _ nil))])
                               ;; No brackets - tool name only
                               [action-text nil])]
       {:action (keyword (str/lower-case tool-name))
        :parameters params
        :full-response response})
     
     :else nil)))

(defn execute-tool
  "Execute a tool with parameters and return the result"
  [tool-fn tool-name parameters]
  (try
    (let [result (apply tool-fn parameters)
          clean-result (str result)]
      (timbre/debugf "REACT-ACT - Direct tool call: %s with params: %s -> %s" tool-name parameters result)
      {:success true :result clean-result})
    (catch Exception e
      (let [error-msg (str "Error calling " tool-name ": " (.getMessage e))]
        (timbre/errorf "Tool execution error: %s" error-msg)
        {:success false :result error-msg}))))


(defn handle-tool-execution
  "Handle tool execution with parameters"
  [tool-info tool-name action-data llm-spec state]
  (let [tool-fn (:symbol tool-info)
        parameters (:parameters action-data)]
    (timbre/debugf "HANDLE-TOOL-EXECUTION - tool: %s, parameters: %s, action-data: %s" tool-name parameters action-data)
    (if parameters
      ;; Direct tool invocation with extracted parameters
      (let [exec-result (execute-tool tool-fn tool-name parameters)]
        (timbre/debugf "HANDLE-TOOL-EXECUTION - Using direct execution")
        {:observation (:result exec-result)
         :finished? false})
      ;; No parameters extracted - treat as error
      (do
        (timbre/debugf "HANDLE-TOOL-EXECUTION - No parameters found, treating as error")
        {:observation (str "No parameters provided for tool '" tool-name "'")
         :finished? false}))))

 #_(defn- create-result-state
  "Helper to create consistent result state"
  [base-state & {:keys [observation finished? result reasoning-trace step completion]}]
  (merge base-state
         (when observation {:observation observation})
         (when (some? finished?) {:finished? finished?})
         (when result {:result result})
         (when reasoning-trace {:reasoning-trace reasoning-trace})
         (when step {:step step})
         (when completion {:completion completion})))            

(defnode react-think
  [{:keys [llm-spec examples tool-map task step reasoning-trace observation max-iterations] :as state}]
  (if (>= step max-iterations)
    {:completion "Maximum iterations reached"
     :finished? true
     :result "Failed to solve within maximum iterations"}
    (let [tool-descriptions (get-tool-descriptions tool-map)
          prompt (create-react-prompt (assoc state :tool-descriptions tool-descriptions))
          system-msg (get state :system-prompt "You are a helpful assistant that can use tools.")
          completion (g/generate 
                      [[:system system-msg]
                       [:user prompt]
                       [:assistant (->llm llm-spec :react-response)]] 
                      {})
          response (get-in completion [:bosquet/completions :react-response])
          action-data (parse-action-response response)]
      {:completion completion
       :response response
       :action-data action-data})))

(defnode react-act
  [{:keys [action-data tool-map step max-iterations llm-spec] :as state}]
  (cond
    (>= step max-iterations)
    {:observation "Maximum iterations reached"
     :finished? true
     :result "Failed to solve within maximum iterations"}

    (not action-data)
    {:observation "Failed to parse action from response"
     :finished? false}

    (= (:action action-data) :final-answer)
    (let [final-answer (:final-answer action-data)
          final-trace (str (or (:reasoning-trace state) "")
                           (:response state) "\n")]
      {:observation "Task completed"
       :finished? true
       :result (or final-answer "Task completed successfully")
       :reasoning-trace final-trace
       :step (inc step)})

    :else
    (let [tool-name (:action action-data)
          tool-info (get tool-map tool-name)]
      (if tool-info
        (handle-tool-execution tool-info tool-name action-data llm-spec state)
        {:observation (str "Tool '" tool-name "' not found")
         :finished? false}))))

(defnode react-update-trace
  [{:keys [response observation step reasoning-trace] :as state}]
  (let [new-trace (str (or reasoning-trace "")
                      response "\n"
                      "Observation " step ": " observation "\n")]
    {:reasoning-trace new-trace
     :step (inc step)}))

(defgraph react-graph
  [:react-think :react-act]
  [:react-act {:continue :react-update-trace :finish :end} 
   (fn [state] (if (:finished? state) :finish :continue))]
  [:react-update-trace :react-think])

(defagent ReAct-agent react-graph
  :react-think
  {:react-think react-think
   :react-act react-act
   :react-update-trace react-update-trace})

(defn create-ReAct-agent
  "Create a ReAct agent with graph-based step control.
  
  Args:
  - tools: Vector of tool function vars with metadata
  - prompt: String prompt or map with :examples key for few-shot examples
  - max-iterations: Maximum number of reasoning steps (default: 15)  
  - llm: LLM specification for bosquet.llm.generator
  
  Returns a function that executes the ReAct reasoning loop."
  [tools prompt max-iterations llm-spec]
  (let [max-steps (or max-iterations 15)
        system-prompt (cond
                        (string? prompt) prompt
                        (map? prompt) (:system prompt)
                        :else "You are a helpful reasoning agent.")
        examples (cond
                   (map? prompt) (:examples prompt)
                   (sequential? prompt) prompt
                   :else nil)]
    (fn [initial-state]
      (let [tool-map (prepare-tools tools)
            state (merge initial-state
                        {:tool-map tool-map
                         :examples examples
                         :system-prompt system-prompt
                         :max-iterations max-steps
                         :llm-spec llm-spec
                         :step 1
                         :reasoning-trace ""})]
        (ReAct-agent state)))))

(comment
  ;; Example usage:


  (defn ^{:desc "Search for information"} search-info  
    [^{:type "string" :desc "Search query"} query]
    (str "Search results for: " query " - This is a mock search result."))
  
  ;; Create LLM spec
  (def llm-spec {:llm wkk/ollama
                 wkk/model-params {:model "gemma3" :temperature 0}})

  (->llm llm-spec :test-response)  
  ;; Create ReAct agent
  (def my-agent 
    (create-ReAct-agent 
      [#'bosquet.llm.tools/add #'bosquet.llm.tools/get-current-weather #'search-info]
      {:system "You are a calculator. Give clean, concise answers"
       ; :examples "Question: Add 2 to the current temperature in London
       ;  Thought 1: To answer the question, I first need the temperature in London.
       ;  Action 1: get-current-weather[\"London\"]
       ;
       ;  Observation 1: {:location \"London\", :temperature \"65\", :unit \"fahrenheit\"}
       ;  Thought 2: Now that I know the temperature in London 65°F, I can add 2 to it.
       ;  Action 2: add[65, 2]
       ;
       ;  Observation 2: 67
       ;  Thought 3: I now have the final result, so I can stop.
       ;  Action 3: Finish[67]"
       }
      10
      llm-spec))
 
  ;; Run the agent  
  (my-agent {:task "What is 123 + 456?"})

  (my-agent {:task "Add 1000 to the current weather in san francisco."})

  (require '[bosquet.template.read :as template])
  (require '[bosquet.agent.tool :refer [search]]) 
  (require '[bosquet.agent.wikipedia :as w]) 

  (defn wiki-tool
     ^{:desc "Lookup or search wikipedia for information"}
      [^{:type "string" :desc "Search query"} query]
    (search (w/->Wikipedia) {:parameters query}))
  
  (def wiki-agent 
    (create-ReAct-agent 
      [#'wiki-tool]
        {:system "You are a able to find results in wikipedia and find answers. Give clean, concise answers"
         :examples (-> (template/load-prompt-palette-edn (clojure.java.io/file "resources/prompt-palette/agent/react-graph.edn")) :react/examples)}
      10
      llm-spec))
  (def question
    "Author David Chanoff has collaborated with a U.S. Navy admiral who served as the ambassador to the United Kingdom under which President?")
  (wiki-agent {:task question})
  (wiki-agent {:task "What city hosted the most recent super bowl and what was the GDP for that state"})

  (g/generate
    [[:system "You are a calculator so only provide the number as the answer without any explanations\nUse any tools to gather information or calculate the answer"]
     [:user "What is the result of adding 1000 to the current weather in san francisco?"]
     [:assistant (->llm llm-spec :tool-response [#'bosquet.llm.tools/get-current-weather #'bosquet.llm.tools/add])]])
  )
